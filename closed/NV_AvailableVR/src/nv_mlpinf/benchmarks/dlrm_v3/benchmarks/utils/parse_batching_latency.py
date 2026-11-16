#!/usr/bin/env python3
"""
Parse batching latency results CSV and plot latency graph.
Horizontal axis: request ordering (chronological)
Vertical axis: latency (ms)

see dump_latency() in inference_server.py
"""
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_latency_data(csv_path: str):
    """Load latency data from CSV file.

    Supports both formats:
    - Old: request_id, inference_latency_ms
    - New: request_id, send_ts, recv_ts, inference_latency_ms
    """
    request_ids = []
    send_times = []
    recv_times = []
    latencies = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            request_ids.append(int(row['request_id']))
            latencies.append(float(row['inference_latency_ms']))

            # New format has send_ts and recv_ts
            if 'send_ts' in row:
                send_times.append(float(row['send_ts']))
            if 'recv_ts' in row:
                recv_times.append(float(row['recv_ts']))

    return request_ids, send_times, recv_times, latencies


def verify_send_time_order(send_times: list) -> bool:
    """Verify that send times are in increasing order (chronological)."""
    if not send_times:
        print("WARNING: No send_ts data available to verify order")
        return True

    out_of_order_count = 0
    out_of_order_indices = []

    for i in range(1, len(send_times)):
        if send_times[i] < send_times[i - 1]:
            out_of_order_count += 1
            if len(out_of_order_indices) < 10:  # Only store first 10
                out_of_order_indices.append(i)

    print("=" * 50)
    print("SEND TIME ORDER VERIFICATION")
    print("=" * 50)

    if out_of_order_count == 0:
        print("✓ Send times are in INCREASING order (chronological)")
        print(f"  First send_ts: {send_times[0]:.6f}")
        print(f"  Last send_ts:  {send_times[-1]:.6f}")
        print(f"  Duration:      {send_times[-1] - send_times[0]:.3f} seconds")
        return True
    else:
        print(f"✗ Found {out_of_order_count} out-of-order entries!")
        print(f"  First 10 out-of-order indices: {out_of_order_indices}")
        for idx in out_of_order_indices[:5]:
            print(f"    Index {idx}: {send_times[idx - 1]:.6f} -> {send_times[idx]:.6f} (diff: {send_times[idx] - send_times[idx - 1]:.6f})")
        return False


def compute_percentiles(latencies: list):
    """Compute and print percentile statistics."""
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    stats = {
        'count': n,
        'min': min(latencies),
        'max': max(latencies),
        'avg': sum(latencies) / n,
        'p50': sorted_lat[int(n * 0.50)],
        'p90': sorted_lat[int(n * 0.90)],
        'p95': sorted_lat[min(int(n * 0.95), n - 1)],
        'p97': sorted_lat[min(int(n * 0.97), n - 1)],
        'p99': sorted_lat[min(int(n * 0.99), n - 1)],
        'p999': sorted_lat[min(int(n * 0.999), n - 1)],
    }

    print("=" * 50)
    print("LATENCY STATISTICS")
    print("=" * 50)
    print(f"  Total requests: {stats['count']}")
    print(f"  Min latency:    {stats['min']:.3f} ms")
    print(f"  Max latency:    {stats['max']:.3f} ms")
    print(f"  Avg latency:    {stats['avg']:.3f} ms")
    print("-" * 50)
    print(f"  P50 latency:    {stats['p50']:.3f} ms")
    print(f"  P90 latency:    {stats['p90']:.3f} ms")
    print(f"  P95 latency:    {stats['p95']:.3f} ms")
    print(f"  P97 latency:    {stats['p97']:.3f} ms")
    print(f"  P99 latency:    {stats['p99']:.3f} ms")
    print(f"  P99.9 latency:  {stats['p999']:.3f} ms")
    print("=" * 50)

    return stats


def plot_latency(latencies: list, output_path: str = "latency_plot.png", title: str = "Inference Latency", start_offset: int = 0):
    """Plot latency over request ordering.

    Args:
        latencies: List of latency values
        output_path: Output file path for the plot
        title: Plot title
        start_offset: Starting index for x-axis labels (for absolute ordering)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # X-axis: request order with absolute offset
    x = np.arange(start_offset, start_offset + len(latencies))

    # Plot latency as dots
    ax.scatter(x, latencies, s=0.3, alpha=0.5, color='steelblue', label='Latency')

    # Add percentile lines
    stats = compute_percentiles(latencies)
    ax.axhline(y=stats['p50'], color='green', linestyle='--', linewidth=1.5, label=f"P50: {stats['p50']:.1f} ms")
    ax.axhline(y=stats['p90'], color='orange', linestyle='--', linewidth=1.5, label=f"P90: {stats['p90']:.1f} ms")
    ax.axhline(y=stats['p99'], color='red', linestyle='--', linewidth=1.5, label=f"P99: {stats['p99']:.1f} ms")
    ax.axhline(y=stats['p999'], color='purple', linestyle='--', linewidth=1.5, label=f"P99.9: {stats['p999']:.1f} ms")

    ax.set_xlabel('Request Order (chronological)', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    return fig, ax


def parse_range(range_str: str, max_len: int) -> tuple:
    """Parse range string like '80000:150000' into (start, end) indices."""
    if not range_str:
        return 0, max_len

    parts = range_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}. Expected 'start:end'")

    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if parts[1] else max_len

    # Clamp to valid range
    start = max(0, min(start, max_len))
    end = max(0, min(end, max_len))

    if start >= end:
        raise ValueError(f"Invalid range: start ({start}) >= end ({end})")

    return start, end


def main():
    parser = argparse.ArgumentParser(description='Parse and plot latency results')
    parser.add_argument('csv_path', type=str, help='Path to latency CSV file')
    parser.add_argument('--output', '-o', type=str, default='latency_plot.png', help='Output plot path')
    parser.add_argument('--title', '-t', type=str, default='Inference Latency', help='Plot title')
    parser.add_argument('--range', '-r', type=str, default=None,
                        help='Range of requests to plot (e.g., 80000:150000)')
    args = parser.parse_args()

    # Load data
    request_ids, send_times, recv_times, latencies = load_latency_data(args.csv_path)
    print(f"Loaded {len(latencies)} latency samples from {args.csv_path}\n")

    # Verify send time order
    verify_send_time_order(send_times)
    print()

    # Apply range filter if specified
    start_offset = 0
    if args.range:
        start, end = parse_range(args.range, len(latencies))
        print(f"Filtering to range [{start}:{end}] ({end - start} samples)\n")
        request_ids = request_ids[start:end]
        send_times = send_times[start:end] if send_times else []
        recv_times = recv_times[start:end] if recv_times else []
        latencies = latencies[start:end]
        start_offset = start

    # Plot
    plot_latency(latencies, output_path=args.output, title=args.title, start_offset=start_offset)


if __name__ == '__main__':
    main()
