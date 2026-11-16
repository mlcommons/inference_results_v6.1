"""
This script iterates through subdirectories of a given log directory,
finds all 'harness_stats.log' files, and extracts the peak 'tokens/s' metric.
The script then prints the timestamp of the run (extracted from the directory name)
and the corresponding peak performance.
"""

import os
import json
import re
import argparse

def find_peak_performance(log_dir):
    """
    Finds the peak tokens/s from harness_stats.log files in subdirectories.

    Args:
        log_dir (str): The root directory containing the log folders.
    """
    if not os.path.isdir(log_dir):
        print(f"Error: Logs directory not found at {log_dir}")
        return

    print(f"Searching for harness_stats.log in: {log_dir}")

    results = []

    for root, dirs, files in os.walk(log_dir):
        if "harness_stats.log" in files:
            log_path = os.path.join(root, "harness_stats.log")

            # Extract timestamp from the directory path
            path_parts = root.split(os.sep)
            timestamp = "N/A"
            for part in path_parts:
                if re.match(r'\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}', part):
                    timestamp = part
                    break

            try:
                with open(log_path, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        continue
                    log_data = json.loads(content)

                peak_tokens_per_s = 0
                if isinstance(log_data, dict):
                    for key, value in log_data.items():
                        if isinstance(value, dict) and "tokens/s" in value:
                            tokens_per_s = value["tokens/s"]
                            if tokens_per_s > peak_tokens_per_s:
                                peak_tokens_per_s = tokens_per_s

                if peak_tokens_per_s > 0:
                    results.append((timestamp, peak_tokens_per_s))

            except json.JSONDecodeError:
                # Ignore malformed or empty json files silently as requested.
                pass
            except Exception as e:
                print(f"Error processing file {log_path}: {e}")

    # Sort results by timestamp for chronological order
    results.sort()

    for timestamp, peak_perf in results:
        print(f"Time of run: {timestamp}, Peak tokens/s: {peak_perf:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find peak token/s from harness_stats.log files.")
    # Assuming the script is run from the workspace root 'mlperf-inference'
    default_log_dir = os.path.join("build", "logs")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=default_log_dir,
        help=f"Directory to search for logs. Defaults to '{default_log_dir}'"
    )
    args = parser.parse_args()

    find_peak_performance(args.log_dir)
