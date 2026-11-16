#!/usr/bin/env python3
"""
Dataset Serializer - Converts CSV-based dataset to preprocessed numpy arrays.

This script loads the dataset from CSV files, creates all KJT tensors,
and serializes them to numpy arrays for fast reloading.

Usage:
    python tools/dataset_dump.py --output-dir /path/to/output [options]

Example:
    # Serialize 10% of the dataset
    python tools/dataset_dump.py \
        --output-dir /data/preprocessed_ds_0.1 \
        --dataset-percentage 0.1

    # Serialize full dataset with multiprocessing
    python tools/dataset_dump.py \
        --output-dir /data/preprocessed_ds_full \
        --dataset-percentage 1.0 \
        --use-multiprocessing
"""

import argparse
import logging
import os
import shutil
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_harness.tools.model_configs import get_hstu_configs
from inference_harness.dataset.mlperf_streaming_qsl import DLRMv3StreamingMLPerfDataset
from inference_harness.dataset.streaming_query_sampler import StreamingQuerySamplerRef

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description='Serialize dataset to numpy arrays for fast loading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save preprocessed numpy arrays'
    )

    # Dataset configuration
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to the raw dataset CSV files'
    )
    parser.add_argument(
        '--dataset-percentage',
        type=float,
        default=1,
        help='Percentage of dataset to load (0.0-1.0)'
    )

    # Loading method
    parser.add_argument(
        '--use-multiprocessing',
        action='store_true',
        help='Use multiprocessing for loading (faster but uses more memory)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of worker processes (only used with --use-multiprocessing)'
    )

    # Advanced options
    parser.add_argument(
        '--train-ts',
        type=int,
        default=90,
        help='Training timestamp'
    )
    parser.add_argument(
        '--total-ts',
        type=int,
        default=100,
        help='Total timestamps in dataset'
    )
    parser.add_argument(
        '--num-users',
        type=int,
        default=50000,
        help='Number of users in dataset'
    )
    return parser.parse_args()


def create_dataset(args) -> StreamingQuerySamplerRef:
    """Create the dataset and sampler."""
    logger.info("Creating HSTU config...")
    hstu_config = get_hstu_configs("production")

    logger.info("Creating dataset...")
    dataset = DLRMv3StreamingMLPerfDataset(
        hstu_config=hstu_config,
        ratings_file_prefix=args.dataset_path,
        is_inference=True,
        train_ts=args.train_ts,
        total_ts=args.total_ts,
        num_files=1,
        num_users=args.num_users,
        num_items=1_000_000_000,
        num_categories=128,
        device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    )

    logger.info("Creating streaming query sampler (using StreamingQuerySamplerRef)...")
    streaming_query_sampler = StreamingQuerySamplerRef(
        ds=dataset,
        dataset_percentage=args.dataset_percentage,
        scenario_name="Offline",
        offline_target_qps=1,
        target_duration=1,
        input_queries=None,
        compute_eval=False,
    )
    return streaming_query_sampler


def load_dataset(streaming_query_sampler, use_multiprocessing: bool) -> None:
    """Load dataset samples into memory."""
    total_samples = streaming_query_sampler.get_item_count()
    logger.info(f"Total samples to load: {total_samples}")

    # Note: StreamingQuerySamplerRef only supports warmup method
    # Multiprocessing is available through the underlying dataset
    logger.info("Loading with warmup method...")
    streaming_query_sampler.load_query_samples_multi_processing(
        range(total_samples)
    )


def serialize_dataset(streaming_query_sampler, output_dir: str) -> None:
    """Serialize the loaded dataset to numpy arrays."""
    logger.info(f"Serializing dataset to: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Copy auxiliary metadata files needed by downstream tools.
    source_dir = streaming_query_sampler.ds.ratings_file_prefix
    aux_files = [
        "offset.csv",
        "requests_per_ts.csv",
        "requests_per_ts_offset.csv",
        "users_cumsum_per_ts.csv",
    ]
    for filename in aux_files:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        if not os.path.exists(src_path):
            logger.warning(f"Aux file missing, skipping copy: {src_path}")
            continue
        shutil.copy2(src_path, dst_path)

    # Serialize
    streaming_query_sampler.ds.serialize_ds(output_dir)


def dataset_stats_calculation(streaming_query_sampler) -> None:
    """Calculate statistics for user interaction history (UIH) lengths in the dataset.

    Collects the length of item_id for all samples across all timestamps,
    calculates statistics (avg, min, max, p90, p95, p99), and plots the distribution.
    """
    logger.info("Calculating UIH statistics...")

    uih_lengths = []

    # Get all timestamp keys (should be 90-99 for 10 timestamps)
    timestamp_keys = sorted(streaming_query_sampler.ds.items_in_memory.keys())
    logger.info(f"Found {len(timestamp_keys)} timestamp keys: {timestamp_keys}")

    # Iterate through all timestamps
    for ts_key in timestamp_keys:
        logger.info(f"Processing timestamp {ts_key}...")
        samples_for_ts = streaming_query_sampler.ds.items_in_memory[ts_key]

        # Iterate through all samples in this timestamp
        for sample_idx in tqdm(range(len(samples_for_ts)), desc=f"TS {ts_key}"):
            try:
                # Get the KJT tuple for this sample
                kjt_tuple = samples_for_ts[sample_idx]

                # Get the first KJT (index 0) which contains item_id
                first_kjt = kjt_tuple[0]

                # Extract item_id field and get its length
                item_id_tensor = first_kjt["item_id"].values()
                uih_length = item_id_tensor.shape[0]
                uih_lengths.append(uih_length)

            except Exception as e:
                logger.warning(f"Error processing timestamp {ts_key}, sample {sample_idx}: {e}")
                continue

    # Convert to numpy array for easier statistics calculation
    uih_lengths = np.array(uih_lengths)

    # Calculate statistics
    avg_length = np.mean(uih_lengths)
    min_length = np.min(uih_lengths)
    max_length = np.max(uih_lengths)
    p90_length = np.percentile(uih_lengths, 90)
    p95_length = np.percentile(uih_lengths, 95)
    p99_length = np.percentile(uih_lengths, 99)
    median_length = np.median(uih_lengths)
    std_length = np.std(uih_lengths)

    # Print statistics
    logger.info("\n" + "=" * 70)
    logger.info("UIH LENGTH STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total samples:     {len(uih_lengths)}")
    logger.info(f"Average:           {avg_length:.2f}")
    logger.info(f"Median:            {median_length:.2f}")
    logger.info(f"Std Dev:           {std_length:.2f}")
    logger.info(f"Min:               {min_length}")
    logger.info(f"Max:               {max_length}")
    logger.info(f"P90:               {p90_length:.2f}")
    logger.info(f"P95:               {p95_length:.2f}")
    logger.info(f"P99:               {p99_length:.2f}")
    logger.info("=" * 70 + "\n")

    # Plot the distribution
    logger.info("Plotting UIH length distribution...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('User Interaction History (UIH) Length Distribution', fontsize=16, fontweight='bold')

    # 1. Histogram with all data
    ax1 = axes[0, 0]
    ax1.hist(uih_lengths, bins=100, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('UIH Length (number of items)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Full Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(avg_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_length:.2f}')
    ax1.axvline(median_length, color='green', linestyle='--', linewidth=2, label=f'Median: {median_length:.2f}')
    ax1.legend()

    # 2. Histogram zoomed to P99
    ax2 = axes[0, 1]
    filtered_data = uih_lengths[uih_lengths <= p99_length]
    ax2.hist(filtered_data, bins=100, edgecolor='black', alpha=0.7, color='lightcoral')
    ax2.set_xlabel('UIH Length (number of items)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution (up to P99)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(p90_length, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90_length:.2f}')
    ax2.axvline(p95_length, color='purple', linestyle='--', linewidth=2, label=f'P95: {p95_length:.2f}')
    ax2.legend()

    # 3. Box plot
    ax3 = axes[1, 0]
    ax3.boxplot(uih_lengths, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax3.set_ylabel('UIH Length (number of items)', fontsize=12)
    ax3.set_title('Box Plot', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_lengths = np.sort(uih_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax4.plot(sorted_lengths, cumulative, linewidth=2, color='navy')
    ax4.set_xlabel('UIH Length (number of items)', fontsize=12)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax4.set_title('Cumulative Distribution Function (CDF)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(90, color='orange', linestyle='--', linewidth=1, label='P90')
    ax4.axhline(95, color='purple', linestyle='--', linewidth=1, label='P95')
    ax4.axhline(99, color='red', linestyle='--', linewidth=1, label='P99')
    ax4.legend()

    plt.tight_layout()

    # Save the plot
    plot_path = 'uih_length_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")

    # Also save statistics to a text file
    stats_path = 'uih_length_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("UIH LENGTH STATISTICS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total samples:     {len(uih_lengths)}\n")
        f.write(f"Average:           {avg_length:.2f}\n")
        f.write(f"Median:            {median_length:.2f}\n")
        f.write(f"Std Dev:           {std_length:.2f}\n")
        f.write(f"Min:               {min_length}\n")
        f.write(f"Max:               {max_length}\n")
        f.write(f"P90:               {p90_length:.2f}\n")
        f.write(f"P95:               {p95_length:.2f}\n")
        f.write(f"P99:               {p99_length:.2f}\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Statistics saved to: {stats_path}")

    return {
        'total_samples': len(uih_lengths),
        'average': avg_length,
        'median': median_length,
        'std': std_length,
        'min': min_length,
        'max': max_length,
        'p90': p90_length,
        'p95': p95_length,
        'p99': p99_length,
        'all_lengths': uih_lengths
    }


def main():
    args = get_args()

    print("=" * 70)
    print("  DATASET SERIALIZER")
    print("=" * 70)
    print(f"  Output directory:     {args.output_dir}")
    print(f"  Dataset path:         {args.dataset_path}")
    print(f"  Dataset percentage:   {args.dataset_percentage * 100:.1f}%")
    print(f"  Use multiprocessing:  {args.use_multiprocessing}")
    print("=" * 70)
    print()

    # Check if output already exists
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        response = input(f"Output directory already contains data. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    start_time = time.time()

    # Step 1: Create dataset
    logger.info("Step 1/3: Creating dataset...")
    step_start = time.time()
    streaming_query_sampler = create_dataset(args)
    logger.info(f"  Dataset created in {time.time() - step_start:.1f}s")

    # Step 2: Load samples
    logger.info("Step 2/4: Loading samples from CSV...")
    step_start = time.time()
    load_dataset(streaming_query_sampler, args.use_multiprocessing)
    load_time = time.time() - step_start
    logger.info(f"  Samples loaded in {load_time:.1f}s")

    # Step 3: Calculate dataset statistics
    logger.info("Step 3/4: Calculating dataset statistics...")
    step_start = time.time()
    dataset_stats_calculation(streaming_query_sampler)
    stats_time = time.time() - step_start
    logger.info(f"  Statistics calculated in {stats_time:.1f}s")

    # Step 4: Serialize
    logger.info("Step 4/4: Serializing to numpy arrays...")
    step_start = time.time()
    serialize_dataset(streaming_query_sampler, args.output_dir)
    serialize_time = time.time() - step_start
    logger.info(f"  Serialization completed in {serialize_time:.1f}s")

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("  SERIALIZATION COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {args.output_dir}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"    - Load time:    {load_time:.1f}s")
    print(f"    - Stats time:   {stats_time:.1f}s")
    print(f"    - Serialize:    {serialize_time:.1f}s")
    print()
    print("  To use the preprocessed dataset:")
    print(f"    streaming_query_sampler.load_query_samples_preprocessed('{args.output_dir}')")
    print("=" * 70)


if __name__ == '__main__':
    main()
