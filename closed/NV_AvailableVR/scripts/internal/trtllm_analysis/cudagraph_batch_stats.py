
import argparse
import re
from collections import Counter
import ast

def parse_log_file(log_file):
    """
    Parses a log file to extract and analyze request statistics.
    """
    scheduled_requests = []
    gen_requests = []

    # Regex to capture num_scheduled_requests and the states dictionary
    line_regex = re.compile(r"num_scheduled_requests: (\d+), states = (\{.*\})")

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = line_regex.search(line)
            if match:
                try:
                    num_scheduled_requests = int(match.group(1))
                    states_str = match.group(2)
                    states_dict = ast.literal_eval(states_str)

                    num_ctx_requests = states_dict.get('num_ctx_requests', 0)

                    num_gen_requests = num_scheduled_requests - num_ctx_requests

                    scheduled_requests.append(num_scheduled_requests)
                    gen_requests.append(num_gen_requests)
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping malformed line: {line.strip()} - Error: {e}")

    return scheduled_requests, gen_requests

def print_top_stats(title, counts, top_n=10):
    """
    Prints the top N most common statistics.
    """
    print(f"--- Top {top_n} {title} ---")
    counter = Counter(counts)
    total_samples = len(counts)
    if total_samples == 0:
        print("No data to analyze.")
        return

    print(f"{'Value':<10} {'Frequency':<15} {'Percentage':<10}")
    print("-" * 35)
    for value, count in counter.most_common(top_n):
        percentage = (count / total_samples) * 100
        print(f"{value:<10} {count:<15} {percentage:.2f}%")
    print("\n")


def main():
    """
    Main function to parse arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze TRT-LLM log files for request statistics."
    )
    parser.add_argument("log_file", help="Path to the log file to analyze.")
    args = parser.parse_args()

    scheduled_requests, gen_requests = parse_log_file(args.log_file)

    print_top_stats("num_scheduled_requests", scheduled_requests, 30)
    print_top_stats("num_gen_requests", gen_requests, 30)


if __name__ == "__main__":
    main()

