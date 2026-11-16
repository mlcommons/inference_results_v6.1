# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

"""
Custom STU (Sequence Transformer Unit) layer implementation for optimized inference.

Provides custom STU layers with configurable attention kernels (PyTorch, Triton, BW-HSTU)
for flexible performance tuning in recommendation model inference.
"""

# pyre-strict
from typing import Optional, Tuple
from enum import Enum, unique
from pathlib import Path
import sys
import importlib.util

import torch
import nvtx
import torch.nn.functional as F

from generative_recommenders.common import switch_to_contiguous_if_needed
from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_mha
from generative_recommenders.ops.triton.triton_hstu_attention import triton_hstu_mha

from generative_recommenders.ops.layer_norm import layer_norm
from generative_recommenders.modules.stu import STULayer, STULayerConfig
from generative_recommenders.ops.hstu_compute import hstu_compute_uqvk, hstu_compute_output
from generative_recommenders.common import HammerKernel
from torch.fx._symbolic_trace import is_fx_tracing

# Import hstu_varlen_fwd_100 from /opt/FBGEMM
# This is needed because fbgemm_gpu may already be imported from a different location
# by other dependencies (e.g., generative_recommenders), and sys.path modifications
# won't help once a module is cached in sys.modules.
# We add the path and manipulate sys.modules to ensure the submodule can be found.
FBGEMM_ROOT = Path("/opt/FBGEMM")
sys.path.insert(0, str(FBGEMM_ROOT))

# Ensure the intermediate package paths are set up in sys.modules
# so that Python can find the submodules from /opt/FBGEMM
import importlib
_fbgemm_experimental_path = FBGEMM_ROOT / "fbgemm_gpu" / "experimental"
if "fbgemm_gpu.experimental" not in sys.modules:
    # Create a module spec for fbgemm_gpu.experimental
    _exp_spec = importlib.util.spec_from_file_location(
        "fbgemm_gpu.experimental",
        _fbgemm_experimental_path / "__init__.py",
        submodule_search_locations=[str(_fbgemm_experimental_path)]
    )
    if _exp_spec:
        _exp_module = importlib.util.module_from_spec(_exp_spec)
        sys.modules["fbgemm_gpu.experimental"] = _exp_module
        _exp_spec.loader.exec_module(_exp_module)

# Now import the module normally
from fbgemm_gpu.experimental.hstu.src.hstu_blackwell import hstu_varlen_fwd_100



@unique
class HammerKernel_t(Enum):
    PYTORCH = "PYTORCH"
    TRITON = "TRITON"
    BW_HSTU = "BW_HSTU"
    CUTE_DSL = "CUTE_DSL"


def hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    sort_by_length: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:

    _, H, _ = q.shape
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(q.dim() == 3, "q must be 3-D")
        torch._assert(k.shape == q.shape, "k must be the same shape as q")
        torch._assert(v.dim() == 3, "v must be 3-D")
        torch._assert(v.shape[0] == q.shape[0], "wrong v shape[0]")
        torch._assert(v.shape[1] == H, "wrong v shape[1]")
        torch._assert(causal, "only support causal attention")

    if kernel in [HammerKernel_t.TRITON]:
        if not is_fx_tracing() and kernel == HammerKernel_t.TRITON:
            torch._assert(q.is_cuda, "q must be CUDA tensor")
            torch._assert(k.is_cuda, "k must be CUDA tensor")
            torch._assert(v.is_cuda, "v must be CUDA tensor")
            torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
            # torch._assert(dropout_pr < 1e-6, "dropout for triton path not implemented")
            torch._assert(
                min_full_attn_seq_len == 0, "min_full_attn_seq_len not implemented"
            )
        assert attn_scale is None, "attn_scale not implemented"
        q = switch_to_contiguous_if_needed(q)
        k = switch_to_contiguous_if_needed(k)
        v = switch_to_contiguous_if_needed(v)
        seq_offsets = seq_offsets.contiguous()
    if kernel == HammerKernel_t.TRITON:
        return triton_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length=sort_by_length,
            enable_tma=enable_tma,
        )
    elif kernel == HammerKernel_t.BW_HSTU:
        # cutlass kernel, not included in the docker image
        from bw_hstu.ops import bw_hstu_mha
        seq_offsets = seq_offsets.to(torch.int32)
        num_targets = num_targets.to(torch.int32)
        return bw_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            min_full_attn_seq_len=min_full_attn_seq_len,
            contextual_seq_len=contextual_seq_len,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            deterministic=False,
            sm_margin=0,
        )
    elif kernel == HammerKernel_t.CUTE_DSL:
        # CuTe DSL kernel (Blackwell optimized)
        seq_offsets = seq_offsets.to(torch.int32)
        num_targets = num_targets.to(torch.int32)
        output, _ = hstu_varlen_fwd_100(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=seq_offsets,
            cu_seqlens_k=seq_offsets,
            max_seqlen_q=max_seq_len,
            max_seqlen_k=max_seq_len,
            num_contexts=None,
            num_targets=num_targets,
            target_group_size=1,
            window_size_left=-1,
            window_size_right=0,
            alpha=alpha,
            rab=None,
            func=None,
            paged_kv=None,
            page_ids=None,
            page_indptrs=None,
        )
        return output
        
    else:
        return pytorch_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            dropout_pr=dropout_pr,
            training=training,
            num_targets=num_targets,
            attn_scale=attn_scale,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=min_full_attn_seq_len,
        )


def hstu_compute_uqvk(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normed_x = layer_norm(
        x,
        weight=norm_weight,
        bias=norm_bias,
        eps=norm_eps,
        kernel=kernel,
    )
    # use cublas gemm add bias
    uvqk = torch.addmm(uvqk_bias, normed_x, uvqk_weight)
    with nvtx.annotate("STU_custom: split_uvqk"):
        u, v, q, k = torch.split(
            uvqk,
            [
                hidden_dim * num_heads,
                hidden_dim * num_heads,
                attn_dim * num_heads,
                attn_dim * num_heads,
            ],
            dim=1,
        )
        u = F.silu(u, inplace=True)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
    return u, q, k, v


class STULayerCustom(STULayer):
    def __init__(self, config: STULayerConfig, is_inference: bool, dtype: torch.dtype = torch.bfloat16, device: str = "cuda:0"):
        super().__init__(config, is_inference=is_inference)
        self.device = device
        # Convert parameters to the specified dtype and device using .data assignment
        # This preserves them as nn.Parameter objects
        self._input_norm_weight.data = self._input_norm_weight.data.to(dtype).to(device)
        self._input_norm_bias.data = self._input_norm_bias.data.to(dtype).to(device)
        self._uvqk_weight.data = self._uvqk_weight.data.to(dtype).to(device)
        self._uvqk_beta.data = self._uvqk_beta.data.to(dtype).to(device)
        self._output_weight.data = self._output_weight.data.to(dtype).to(device)
        self._output_norm_weight.data = self._output_norm_weight.data.to(dtype).to(device)
        self._output_norm_bias.data = self._output_norm_bias.data.to(dtype).to(device)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_eps = 1e-6
        num_heads = self._num_heads
        attn_dim = self._attention_dim
        hidden_dim = self._hidden_dim
        with nvtx.annotate("STU_custom: hstu_compute_uqvk"):
            u, q, k, v = hstu_compute_uqvk(
                x=x,
                norm_weight=self._input_norm_weight,
                norm_bias=self._input_norm_bias,
                norm_eps=norm_eps,
                num_heads=num_heads,
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                uvqk_weight=self._uvqk_weight,
                uvqk_bias=self._uvqk_beta,
                kernel=HammerKernel.TRITON,
            )
            # Ensure tensors are contiguous and on the correct device
            # (hstu_compute_uqvk uses .view() which may create non-contiguous tensors)
            q = switch_to_contiguous_if_needed(q)
            k = switch_to_contiguous_if_needed(k)
            v = switch_to_contiguous_if_needed(v)
            x_offsets = x_offsets.contiguous()

        with nvtx.annotate("STU_custom: hstu_mha"):
            attn_output = hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=x_offsets,
                causal=self._causal,
                dropout_pr=0.0,
                training=False,
                num_targets=num_targets,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                min_full_attn_seq_len=0,
                sort_by_length=self._sort_by_length,
                kernel=HammerKernel_t.CUTE_DSL,
                enable_tma=False,
            )
            attn_output = attn_output.view(-1, hidden_dim * num_heads)
        with nvtx.annotate("STU_custom: hstu_compute_output"):
            output = hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight,
                norm_bias=self._output_norm_bias,
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight,
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_ux=True,
                training=False,
                kernel=HammerKernel.TRITON,
                recompute_y_in_backward=self._recompute_y,
            )
        return output
