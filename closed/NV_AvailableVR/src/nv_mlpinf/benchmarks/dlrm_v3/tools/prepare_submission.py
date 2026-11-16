#!/usr/bin/env python3
"""
Prepare MLPerf Inference submission directory structure for dlrm-v3.

This script takes a raw results directory and reorganizes it into the
proper MLPerf submission format under results/<system_name>/dlrm-v3/.

Usage:
    python prepare_submission.py \
        --input /path/to/raw/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT \
        --output /path/to/submission/root \
        --user-conf /path/to/user.conf

Example:
    python prepare_submission.py \
        --input <..>/mlperf-inference/closed/NVIDIA/src/nv_mlpinf/benchmarks/dlrm_v3/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT \
        --output <..>/mlperf-inference \
        --user-conf user_gb200x8.conf
"""

import argparse
import json
import os
import shutil


BENCHMARK_NAME = "dlrm-v3"
SUBMITTER = "NVIDIA"
DIVISION = "closed"

# Scenarios to process
SCENARIOS = ["Offline", "Server"]

# Files to copy from accuracy directory
ACCURACY_FILES = [
    "accuracy.txt",
    "mlperf_log_accuracy.json",
    "mlperf_log_detail.txt",
    "mlperf_log_summary.txt",
]

# Files to copy from performance directory
PERFORMANCE_FILES = [
    "mlperf_log_detail.txt",
    "mlperf_log_summary.txt",
]

# Files to copy from audit directory (TEST08 compliance)
AUDIT_FILES = [
    "verify_accuracy.txt",
]

# Default measurements.json content for dlrm-v3
DEFAULT_MEASUREMENTS = {
    "input_data_types": "fp16",
    "retraining": "No",
    "starting_weights_filename": "MLCommons hosted model weights",
    "weight_data_types": "fp16",
    "weight_transformations": "None"
}

# Default README content template
README_TEMPLATE = """# DLRM-v3 {scenario} Submission

## Model
DLRM-v3 recommendation model

## Hardware
{system_name}

## Software
TensorRT, CUDA

## Instructions
See main repository README for build and run instructions.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MLPerf submission directory structure for dlrm-v3"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to raw results directory (e.g., .../GB200-NVL72_GB200-186GB_aarch64x72_TRT)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to submission root directory (should contain 'closed' folder or it will be created)"
    )
    parser.add_argument(
        "--user-conf", "-u",
        required=True,
        help="Path to user.conf file to copy into submission"
    )
    parser.add_argument(
        "--measurements-json", "-m",
        default=None,
        help="Path to measurements.json file (optional, will use default if not provided)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually copying files"
    )
    return parser.parse_args()


def get_system_id_from_path(input_path: str) -> str:
    """Extract system ID from input path (last directory component)."""
    return os.path.basename(os.path.normpath(input_path))


def ensure_dir(path: str, dry_run: bool = False):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        if dry_run:
            print(f"[DRY-RUN] Would create directory: {path}")
        else:
            os.makedirs(path)
            print(f"Created directory: {path}")


def copy_file(src: str, dst: str, dry_run: bool = False):
    """Copy a file from src to dst."""
    if not os.path.exists(src):
        print(f"WARNING: Source file does not exist: {src}")
        return False
    
    if dry_run:
        print(f"[DRY-RUN] Would copy: {src} -> {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    return True


def write_json(path: str, data: dict, dry_run: bool = False):
    """Write JSON data to file."""
    if dry_run:
        print(f"[DRY-RUN] Would write JSON to: {path}")
        print(f"  Content: {json.dumps(data, indent=2)[:200]}...")
    else:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Wrote JSON: {path}")


def write_text(path: str, content: str, dry_run: bool = False):
    """Write text content to file."""
    if dry_run:
        print(f"[DRY-RUN] Would write text to: {path}")
    else:
        with open(path, 'w') as f:
            f.write(content)
        print(f"Wrote: {path}")


def process_scenario(input_dir: str, output_scenario_dir: str, scenario: str,
                     user_conf_path: str, measurements_json_path: str,
                     system_name: str, dry_run: bool = False):
    """Process a single scenario (Offline or Server)."""
    
    input_scenario_dir = os.path.join(input_dir, scenario)
    
    if not os.path.exists(input_scenario_dir):
        print(f"WARNING: Input scenario directory does not exist: {input_scenario_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing scenario: {scenario}")
    print(f"{'='*60}")
    
    # Create output directories
    accuracy_dir = os.path.join(output_scenario_dir, "accuracy")
    performance_dir = os.path.join(output_scenario_dir, "performance", "run_1")
    test08_dir = os.path.join(output_scenario_dir, "TEST08")
    
    ensure_dir(accuracy_dir, dry_run)
    ensure_dir(performance_dir, dry_run)
    ensure_dir(test08_dir, dry_run)
    
    # Copy accuracy files
    print(f"\nCopying accuracy files...")
    input_accuracy_dir = os.path.join(input_scenario_dir, "accuracy")
    for filename in ACCURACY_FILES:
        src = os.path.join(input_accuracy_dir, filename)
        dst = os.path.join(accuracy_dir, filename)
        copy_file(src, dst, dry_run)
    
    # Copy performance files (into run_1 subdirectory)
    print(f"\nCopying performance files...")
    input_performance_dir = os.path.join(input_scenario_dir, "performance")
    for filename in PERFORMANCE_FILES:
        src = os.path.join(input_performance_dir, filename)
        dst = os.path.join(performance_dir, filename)
        copy_file(src, dst, dry_run)
    
    # Copy audit files (TEST08 compliance test results)
    print(f"\nCopying audit files (TEST08 compliance)...")
    input_audit_dir = os.path.join(input_scenario_dir, "audit")
    if os.path.exists(input_audit_dir):
        for filename in AUDIT_FILES:
            src = os.path.join(input_audit_dir, filename)
            dst = os.path.join(test08_dir, filename)
            copy_file(src, dst, dry_run)
    else:
        print(f"WARNING: Audit directory does not exist: {input_audit_dir}")
        print(f"  TEST08 compliance files will need to be added manually.")
    
    # Copy user.conf
    print(f"\nCopying user.conf...")
    user_conf_dst = os.path.join(output_scenario_dir, "user.conf")
    copy_file(user_conf_path, user_conf_dst, dry_run)
    
    # Create or copy measurements.json
    print(f"\nCreating measurements.json...")
    measurements_dst = os.path.join(output_scenario_dir, "measurements.json")
    if measurements_json_path and os.path.exists(measurements_json_path):
        copy_file(measurements_json_path, measurements_dst, dry_run)
    else:
        write_json(measurements_dst, DEFAULT_MEASUREMENTS, dry_run)
    
    # Create README.md
    print(f"\nCreating README.md...")
    readme_dst = os.path.join(output_scenario_dir, "README.md")
    readme_content = README_TEMPLATE.format(scenario=scenario, system_name=system_name)
    write_text(readme_dst, readme_content, dry_run)
    
    return True


def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"ERROR: Input directory does not exist: {args.input}")
        return 1
    
    if not os.path.exists(args.user_conf):
        print(f"ERROR: User conf file does not exist: {args.user_conf}")
        return 1
    
    # Extract system ID from input path
    system_id = get_system_id_from_path(args.input)
    print(f"System ID: {system_id}")
    
    # Build output paths
    results_dir = os.path.join(
        args.output, DIVISION, SUBMITTER, "results", system_id, BENCHMARK_NAME
    )
    
    print(f"\nInput directory: {args.input}")
    print(f"Output results directory: {results_dir}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")
    
    # Create base directories
    ensure_dir(results_dir, args.dry_run)
    
    # Process each scenario
    for scenario in SCENARIOS:
        output_scenario_dir = os.path.join(results_dir, scenario)
        process_scenario(
            args.input,
            output_scenario_dir,
            scenario,
            args.user_conf,
            args.measurements_json,
            system_id,
            args.dry_run
        )
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nSubmission structure created at: {results_dir}")
    
    # Print helpful next steps
    print(f"\nNext steps:")
    print(f"  1. Run truncate_accuracy_log.py to truncate large accuracy logs")
    print(f"  2. Aggregate dlrm-v3 results into general mlperf result folder along with other benchmarks, see submission guide for details")
    print(f"\nExample commands:")
    print(f"  python /opt/inference/tools/submission/truncate_accuracy_log.py --input {args.output} --submitter {SUBMITTER} --backup {args.output}/accuracy_backup")
    
    return 0


if __name__ == "__main__":
    exit(main())
