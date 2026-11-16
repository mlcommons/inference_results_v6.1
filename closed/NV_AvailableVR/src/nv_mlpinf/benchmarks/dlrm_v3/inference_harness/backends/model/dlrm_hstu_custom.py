"""
Custom DLRM-HSTU model implementation for optimized inference.

Extends the base DLRM-HSTU model with custom forward passes and
optimizations for multi-task recommendation inference.
"""

import generative_recommenders.modules.dlrm_hstu as dlrm_hstu
import torch
from typing import Dict, Tuple, Optional, List
from generative_recommenders.modules.dlrm_hstu import SequenceEmbedding
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
import nvtx
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)


def _get_supervision_labels_and_weights(
    supervision_bitmasks: torch.Tensor,
    task_configs: List[TaskConfig],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Extract supervision labels and weights from bitmasks for multi-task learning.

    Args:
        supervision_bitmasks: Tensor of supervision bitmasks.
        task_configs: List of task configurations.

    Returns:
        Tuple of (supervision_labels dict, supervision_weights dict).

    Raises:
        RuntimeError: If task type is not supported.
    """
    supervision_labels: Dict[str, torch.Tensor] = {}
    supervision_weights: Dict[str, torch.Tensor] = {}
    for task in task_configs:
        if task.task_type == MultitaskTaskType.BINARY_CLASSIFICATION:
            supervision_labels[task.task_name] = (
                torch.bitwise_and(supervision_bitmasks, task.task_weight) > 0
            ).to(torch.float32)
        else:
            raise RuntimeError("Unsupported MultitaskTaskType")
    return supervision_labels, supervision_weights


class DlrmHSTUCustom(dlrm_hstu.DlrmHSTU):
    """
    Custom DLRM-HSTU implementation with optimized inference paths.

    Extends the base DlrmHSTU model with custom forward methods tailored
    for high-throughput inference in distributed serving environments.
    """

    def __init__(
            self,
            hstu_configs: dlrm_hstu.DlrmHSTUConfig,
            embedding_tables: Dict[str, dlrm_hstu.EmbeddingConfig],
            is_dense: bool = False,
            is_inference: bool = True):
        """
        Initialize custom DLRM-HSTU model.

        Args:
            hstu_configs: HSTU model configuration.
            embedding_tables: Embedding table configurations.
            is_dense: Whether to use dense-only mode (no sparse embeddings).
            is_inference: Whether model is in inference mode.
        """
        super().__init__(hstu_configs=hstu_configs, embedding_tables=embedding_tables, is_inference=is_inference, is_dense=is_dense, bf16_training=True)

    def main_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:

        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if uih_feature_name in seq_embeddings:
                seq_embeddings[uih_feature_name] = SequenceEmbedding(
                    lengths=uih_seq_lengths + num_candidates,
                    embedding=concat_2D_jagged(
                        max_seq_len=max_uih_len + max_num_candidates,
                        max_len_left=max_uih_len,
                        offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            uih_seq_lengths
                        ),
                        values_left=seq_embeddings[uih_feature_name].embedding,
                        max_len_right=max_num_candidates,
                        offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            num_candidates
                        ),
                        values_right=seq_embeddings[candidate_feature_name].embedding,
                        kernel=self.hammer_kernel(),
                    ),
                )
        with nvtx.annotate(f"DlrmHSTUCustom - item_forward", color="green"):
            candidates_item_embeddings = self._item_forward(
                seq_embeddings,
            )

        with nvtx.annotate(f"DlrmHSTUCustom - user_forward", color="blue"):
            # import pdb; pdb.set_trace()
            candidates_user_embeddings = self._user_forward(
                max_uih_len=max_uih_len,
                max_candidates=max_num_candidates,
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                num_candidates=num_candidates,
            )
        with nvtx.annotate(f"DlrmHSTUCustom - multitask_module", color="purple"):
            supervision_labels, supervision_weights = (
                _get_supervision_labels_and_weights(
                    supervision_bitmasks=payload_features[
                        self._hstu_configs.candidates_weight_feature_name
                    ],
                    task_configs=self._multitask_configs,
                )
            )
            mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
                self._multitask_module(
                    encoded_user_embeddings=candidates_user_embeddings,
                    item_embeddings=candidates_item_embeddings,
                    supervision_labels=supervision_labels,
                    supervision_weights=supervision_weights,
                )
            )

        assert mt_target_preds is not None
        # Use pinned CPU buffers + non_blocking copy for async D2H
        mt_target_preds = mt_target_preds.detach()
        preds_cpu = torch.empty_like(mt_target_preds, device="cpu", pin_memory=True)
        preds_cpu.copy_(mt_target_preds, non_blocking=True)

        if mt_target_labels is not None:
            mt_target_labels = mt_target_labels.detach()
            labels_cpu = torch.empty_like(mt_target_labels, device="cpu", pin_memory=True)
            labels_cpu.copy_(mt_target_labels, non_blocking=True)
        else:
            labels_cpu = None

        if mt_target_weights is not None:
            mt_target_weights = mt_target_weights.detach()
            weights_cpu = torch.empty_like(mt_target_weights, device="cpu", pin_memory=True)
            weights_cpu.copy_(mt_target_weights, non_blocking=True)
        else:
            weights_cpu = None

        # Ensure async copies complete before returning CPU tensors
        torch.cuda.current_stream(mt_target_preds.device).synchronize()
        return (
            preds_cpu,
            labels_cpu,
            weights_cpu,
        )
