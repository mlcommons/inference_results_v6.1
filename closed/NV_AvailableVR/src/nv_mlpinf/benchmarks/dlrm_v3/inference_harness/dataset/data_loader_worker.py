"""
Isolated data loading worker module for ProcessPoolExecutor with spawn context.
This module is imported by spawned child processes.
"""

import os

# Clear MPI-related environment variables BEFORE any other imports
# This prevents spawned child processes from trying to re-initialize MPI
# _MPI_ENV_PREFIXES = (
#     'OMPI_',      # Open MPI
#     'PMI_',       # Process Management Interface
#     'PMIX_',      # PMIx
#     'MPI_',       # Generic MPI
#     'SLURM_',     # SLURM (MPI launcher)
#     'ORTE_',      # Open RTE (Open MPI runtime)
#     'OPAL_',      # Open MPI portability layer
# )

# _mpi_vars_to_remove = [key for key in os.environ if key.startswith(_MPI_ENV_PREFIXES)]
# for var in _mpi_vars_to_remove:
#     del os.environ[var]

import csv
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import os as _os
_os.environ['MPI4PY_RC_INITIALIZE'] = '0'
_os.environ['MPI4PY_RC_FINALIZE'] = '0'
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torchrec import KeyedJaggedTensor
import json
logger = logging.getLogger(__name__)


def json_loads(
    x: str | int | List[int],
) -> List[int]:
    """
    Parse a JSON-like string into a list of integers.

    Handles multiple input formats including JSON arrays, comma-separated
    strings, and single values.

    Args:
        x: Input that can be a JSON array string, a single integer,
           or already a list of integers.

    Returns:
        List of integers parsed from the input.
    """
    if isinstance(x, str):
        if x[0] != "[" and x[-1] != "]":
            x = "[" + x + "]"
        y = json.loads(x)
    else:
        y = x
    y_list = [y] if type(y) == int else list(y)
    return y_list


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    """
    Truncate a sequence if it exceeds the maximum length.

    Args:
        y: Input sequence to potentially truncate.
        max_seq_len: Maximum allowed sequence length.

    Returns:
        The input sequence, truncated to max_seq_len if necessary.
    """
    y_len = len(y)
    if y_len > max_seq_len:
        y = y[:max_seq_len]
    return y


# <offset.csv>
# line 0: [0, 100, 150, 200....] 50000 line offsets of data.csv
# line 1: [1, 100, 150, 200....] 50000
# line 2: [2, 100, 150, 200....] 50000
# ....
# line <num_files>: [] 50000

# <request_per_ts.csv>
# line 0: ts = 0, [1, 3, 5, 7 ,..... 4999999] 0.7 * 5,000,000 requests
# line 1: ts = 1, [1, 3, 5, 7 ,..... 4999999] 0.7 * 5,000,000 requests
# line 2: ts = 2, [1, 3, 5, 7 ,..... 4999999] 0.7 * 5,000,000 requests
# ....
# line 100: ts = 100, [1, 3, 5, 7 ,..... 4999999] 0.7 * 5,000,000 requests


# <requests_per_ts_offset.csv>
# line 0: [0, 12300, 1512512, 123125412]: 100 indices for each line of requests_per_ts.csv


# <users_cumsum_per_ts>
# line 0: ts = 0, [file0 total users, file 0 + file 1 total users, file 0 + file 1 + file 2 total users, ....]
# line 1: ts = 1, [file0 total users, file 0 + file 1 total users, file 0 + file 1 + file 2 total users, ....]
# line 2: ts = 2, [file0 total users, file 0 + file 1 total users, file 0 + file 1 + file 2 total users, ....]
# ....
# line 100: ts = 100, [file0 total users, file 0 + file 1 total users, file 0 + file 1 + file 2 total users, ....]

def split_combined_kjt(combined_kjt: KeyedJaggedTensor, num_uih_keys: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    """
    Split a combined KJT back into separate UIH and candidates KJTs.
    This is called in the main process after receiving from workers.
    """
    all_keys = combined_kjt.keys()
    all_lengths = combined_kjt.lengths()
    all_values = combined_kjt.values()

    # Split keys
    uih_keys = all_keys[:num_uih_keys]
    candidates_keys = all_keys[num_uih_keys:]

    # Split lengths
    uih_lengths = all_lengths[:num_uih_keys]
    candidates_lengths = all_lengths[num_uih_keys:]

    # Calculate value split point
    uih_values_len = uih_lengths.sum().item()
    uih_values = all_values[:uih_values_len]
    candidates_values = all_values[uih_values_len:]

    # Create separate KJTs
    uih_kjt = KeyedJaggedTensor(
        keys=uih_keys,
        lengths=uih_lengths,
        values=uih_values,
    )

    candidates_kjt = KeyedJaggedTensor(
        keys=candidates_keys,
        lengths=candidates_lengths,
        values=candidates_values,
    )

    return uih_kjt, candidates_kjt


def load_samples_worker(args):
    """
    Worker function for parallel data loading.
    This function is executed in a separate process.
    Returns combined KJT to reduce mmap regions during transfer.
    """
    (pid, sample_chunk, file_offset, requests_at_ts, users_per_file, process_line_config, li_config, ratings_file_path) = args

    # Now you can call methods on the dataset instance
    results = []
    user_data_file = open(ratings_file_path, "r")
    try:
        for idx, sample_idx in enumerate(sample_chunk):
            if idx % 500 == 0 or idx == len(sample_chunk) - 1:
                logger.debug(f"[rank {pid}] progress: {idx}/{len(sample_chunk)}, current sample_idx: {sample_idx}")
            # Call iloc to load the data
            data = _iloc_worker(file_offset, requests_at_ts, users_per_file, sample_idx, user_data_file, process_line_config)
            # Returns (combined_kjt, num_uih_keys) to reduce mmap pressure
            combined_kjt, num_uih_keys = _load_item_worker(data, 2048, li_config)
            results.append((sample_idx, combined_kjt, num_uih_keys))
    finally:
        user_data_file.close()

    return results


def _iloc_worker(file_offset, requests_at_ts, users_per_file, idx, user_data_file, pl_config):
    user_idx = requests_at_ts[idx]
    idx = user_idx % users_per_file
    user_data_file.seek(file_offset[idx])
    line = user_data_file.readline()
    data = _process_line_worker(line, user_idx, pl_config)
    return data


@dataclass
class ProcessLineConfig:
    ts: int
    train_ts: int
    total_ts: int
    is_eval: bool
    is_inference: bool


def _process_line_worker(line: str, user_id: int, config: ProcessLineConfig):
    reader = csv.reader([line])
    parsed_line = next(reader)
    # total ts + one more eval ts + one base ts so that uih won't be zero
    # for each ts, ordered as candidate_ids, candidate_ratings, uih_ids, uih_ratings
    assert len(parsed_line) == 4 * (config.total_ts + 2)
    uih_item_ids_list = []
    uih_ratings_list = []
    candidate_item_ids = ""
    candidate_ratings = ""
    if (not config.is_eval) and (not config.is_inference):
        assert config.ts < config.train_ts
        for i in range(config.ts + 1):
            if parsed_line[4 * i]:
                uih_item_ids_list.append(parsed_line[2 + 4 * i])
                uih_ratings_list.append(parsed_line[3 + 4 * i])
        candidate_item_ids = parsed_line[4 * (config.ts + 1)]
        candidate_ratings = parsed_line[1 + 4 * (config.ts + 1)]
    elif config.is_eval:
        for i in range(config.ts + 1):
            if parsed_line[4 * i]:
                uih_item_ids_list.append(parsed_line[2 + 4 * i])
                uih_ratings_list.append(parsed_line[3 + 4 * i])
        candidate_item_ids = parsed_line[4 * (config.ts + 1)]
        candidate_ratings = parsed_line[1 + 4 * (config.ts + 1)]
    else:
        assert config.is_inference is True
        assert config.ts >= config.train_ts
        for i in range(config.train_ts + 1):
            if parsed_line[4 * i]:
                uih_item_ids_list.append(parsed_line[2 + 4 * i])
                uih_ratings_list.append(parsed_line[3 + 4 * i])
        for i in range(config.train_ts + 2, config.ts + 2):
            if parsed_line[4 * i]:
                uih_item_ids_list.append(parsed_line[2 + 4 * i])
                uih_ratings_list.append(parsed_line[3 + 4 * i])
        candidate_item_ids = parsed_line[4 * (config.ts + 2)]
        candidate_ratings = parsed_line[1 + 4 * (config.ts + 2)]
    uih_item_ids = ",".join(uih_item_ids_list)
    uih_ratings = ",".join(uih_ratings_list)
    assert candidate_item_ids != "" and candidate_ratings != ""
    return pd.Series(
        data={
            "user_id": user_id,
            "uih_item_ids": uih_item_ids,
            "uih_ratings": uih_ratings,
            "candidate_item_ids": candidate_item_ids,
            "candidate_ratings": candidate_ratings,
        }
    )


@dataclass
class LoadItemConfig:
    max_num_candidates: int
    max_uih_len: int
    action_weights: List[int]
    contextual_feature_to_max_length: Dict[str, int]
    items_per_category: int
    uih_keys: List[str]
    candidates_keys: List[str]


def _load_item_worker(data: pd.Series, max_num_candidates: int, li_config: LoadItemConfig):
    ids_uih = json_loads(data["uih_item_ids"])
    ids_candidates = json_loads(data.candidate_item_ids)
    ratings_uih = json_loads(data.uih_ratings)
    ratings_candidates = json_loads(data.candidate_ratings)
    timestamps_uih = [1] * len(ids_uih)

    assert len(ids_uih) == len(
        timestamps_uih
    ), "history len differs from timestamp len."
    assert len(ids_uih) == len(
        ratings_uih
    ), f"history len {len(ids_uih)} differs from ratings len {len(ratings_uih)}."
    assert (
        len(ids_candidates) == len(ratings_candidates)
    ), f"candidates len {len(ids_candidates)} differs from ratings len {len(ratings_candidates)}."

    ids_uih = maybe_truncate_seq(ids_uih, li_config.max_uih_len)
    ratings_uih = maybe_truncate_seq(ratings_uih, li_config.max_uih_len)
    timestamps_uih = maybe_truncate_seq(timestamps_uih, li_config.max_uih_len)
    ids_candidates = maybe_truncate_seq(ids_candidates, max_num_candidates)
    num_candidates = len(ids_candidates)
    ratings_candidates = maybe_truncate_seq(ratings_candidates, max_num_candidates)
    action_weights_uih = [
        li_config.action_weights[int(rating) - 1] for rating in ratings_uih
    ]
    action_weights_candidates = [
        int(rating >= 3.5) for rating in ratings_candidates
    ]

    uih_kjt_values: List[int] = []
    uih_kjt_lengths: List[int] = []
    for name, length in li_config.contextual_feature_to_max_length.items():
        uih_kjt_values.append(data[name])
        uih_kjt_lengths.append(length)

    uih_seq_len = len(ids_uih)
    item_category_ids = [id // li_config.items_per_category for id in ids_uih]
    dummy_watch_times_uih = [0 for _ in range(uih_seq_len)]

    # Build UIH keys list (copy to avoid modifying the original)
    uih_keys = list(li_config.uih_keys)

    extend_uih_kjt_values: List[int] = (
        ids_uih
        + ratings_uih
        + timestamps_uih
        + action_weights_uih
        + dummy_watch_times_uih
        + item_category_ids
    )

    uih_kjt_values.extend(extend_uih_kjt_values)
    uih_kjt_lengths.extend(
        [
            uih_seq_len
            for _ in range(
                len(uih_keys) - len(li_config.contextual_feature_to_max_length)
            )
        ]
    )

    dummy_query_time = 0 if timestamps_uih == [] else max(timestamps_uih)
    uih_kjt_values.append(dummy_query_time)
    uih_kjt_lengths.append(1)

    # Prepare candidates data
    item_candidate_category_ids = [
        id // li_config.items_per_category for id in ids_candidates
    ]

    # Build candidates keys list (copy to avoid modifying the original)
    candidates_keys = list(li_config.candidates_keys)

    candidates_kjt_values = (
        ids_candidates
        + ratings_candidates
        + [dummy_query_time] * num_candidates  # item_query_time
        + action_weights_candidates
        + [1] * num_candidates  # item_dummy_watchtime
        + item_candidate_category_ids
    )

    candidates_kjt_lengths = [num_candidates] * len(candidates_keys)

    # OPTIMIZATION: Combine both KJTs into one to reduce mmap regions during multiprocessing
    # This reduces the number of tensors from 4 to 2 (halving mmap pressure)
    combined_keys = uih_keys + ["dummy_query_time"] + candidates_keys
    combined_lengths = uih_kjt_lengths + candidates_kjt_lengths
    combined_values = uih_kjt_values + candidates_kjt_values

    combined_kjt = KeyedJaggedTensor(
        keys=combined_keys,
        lengths=torch.tensor(combined_lengths).long(),
        values=torch.tensor(combined_values).long(),
    )

    # Return combined KJT and the split index (number of uih keys)
    num_uih_keys = len(uih_keys) + 1  # +1 for dummy_query_time
    return combined_kjt, num_uih_keys
