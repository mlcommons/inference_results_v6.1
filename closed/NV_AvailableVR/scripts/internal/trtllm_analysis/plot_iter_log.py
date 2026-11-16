"""
This script parses a .jsonl metrics file to visualize in-flight batching scheduler performance.
It generates a plot with three subplots:
1. Iteration latency over iterations.
2. The number of generation and total tokens in each batch over iterations.
3. The number of max, allocated, and free blocks over iterations.
"""

import json
import argparse
import os
import matplotlib.pyplot as plt

def plot_metrics(file_path, warmup_iters=0):
    """
    Reads a .jsonl metrics file and plots scheduler performance graphs.

    Args:
        file_path (str): The path to the .jsonl metrics file.
        warmup_iters (int): Number of initial iterations to skip.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Data lists for plotting
    iters = []
    iter_latency_ms = []
    gen_tokens = []
    total_tokens = []
    max_num_blocks = []
    used_num_blocks = []
    free_num_blocks = []

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)

                # Ensure essential keys are present
                if "iter" not in data or "iterLatencyMS" not in data:
                    continue

                if data["iter"] <= warmup_iters:
                    continue

                iters.append(data["iter"])
                iter_latency_ms.append(data["iterLatencyMS"])

                # Token data from inflightBatchingStats
                inflight_stats = data.get("inflightBatchingStats", {})
                num_gen_requests = inflight_stats.get("numGenRequests", 0)
                num_ctx_tokens = inflight_stats.get("numCtxTokens", 0)
                gen_tokens.append(num_gen_requests)
                total_tokens.append(num_ctx_tokens + num_gen_requests)

                # Block data from kvCacheStats
                kv_stats = data.get("kvCacheStats", {})
                max_num_blocks.append(kv_stats.get("maxNumBlocks", 0))
                used_num_blocks.append(kv_stats.get("usedNumBlocks", 0))
                free_num_blocks.append(kv_stats.get("freeNumBlocks", 0))

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line in {file_path}")
                continue

    if not iters:
        print("No valid data to plot.")
        return

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # 1. Iteration Latency
    ax1.plot(iters, iter_latency_ms, label="Iter Latency (ms)", color='b')
    ax1.set_title("Iteration Latency")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True)
    ax1.legend()

    # 2. Tokens in Batch
    ax2.plot(iters, gen_tokens, label="Gen Tokens", color='g')
    ax2.plot(iters, total_tokens, label="Total Tokens", color='r')
    ax2.set_title("Tokens in Batch")
    ax2.set_ylabel("Number of Tokens")
    ax2.grid(True)
    ax2.legend()

    # 3. Blocks
    ax3.plot(iters, max_num_blocks, label="Max Blocks", color='purple')
    ax3.plot(iters, used_num_blocks, label="Used Blocks", color='orange')
    ax3.plot(iters, free_num_blocks, label="Free Blocks", color='cyan')
    ax3.set_title("Block Usage")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Number of Blocks")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()

    # Save the plot
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + ".png"
    output_path = os.path.join(os.path.dirname(file_path), output_filename)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # plt.show() # Commented out for non-interactive environments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot in-flight batching scheduler metrics from a .jsonl file.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the metrics_..._.jsonl file."
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of initial iterations to skip for plotting to remove startup noise."
    )
    args = parser.parse_args()

    plot_metrics(args.file_path, args.warmup_iters)
