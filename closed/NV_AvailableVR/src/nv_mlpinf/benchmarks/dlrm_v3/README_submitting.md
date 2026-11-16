# Result Folder Preparation for Submission

This guide covers how to prepare your result folder for MLPerf submission after collecting all benchmark results.

## Prerequisites

Before proceeding, ensure you have completed the result collection process, which includes:
- **Offline scenario**: accuracy, audit, and performance runs
- **Server scenario**: accuracy, audit, and performance runs

## Step 1: Verify Directory Structure

Your result folder should be structured as follows:

```
<YOUR SYSTEM NAME>/
├── Offline/
│   ├── accuracy/
│   │   ├── accuracy.txt
│   │   ├── mlperf_log_accuracy.json
│   │   ├── mlperf_log_detail.txt
│   │   ├── mlperf_log_summary.txt
│   │   └── mlperf_log_trace.json
│   ├── audit/
│   │   ├── mlperf_log_accuracy.json
│   │   ├── mlperf_log_detail.txt
│   │   ├── mlperf_log_summary.txt
│   │   ├── mlperf_log_trace.json
│   │   └── verify_accuracy.txt
│   └── performance/
│       ├── mlperf_log_accuracy.json
│       ├── mlperf_log_detail.txt
│       ├── mlperf_log_summary.txt
│       └── mlperf_log_trace.json
└── Server/
    ├── accuracy/
    │   ├── accuracy.txt
    │   ├── mlperf_log_accuracy.json
    │   ├── mlperf_log_detail.txt
    │   ├── mlperf_log_summary.txt
    │   └── mlperf_log_trace.json
    ├── audit/
    │   ├── mlperf_log_accuracy.json
    │   ├── mlperf_log_detail.txt
    │   ├── mlperf_log_summary.txt
    │   ├── mlperf_log_trace.json
    │   └── verify_accuracy.txt
    └── performance/
        ├── mlperf_log_accuracy.json
        ├── mlperf_log_detail.txt
        ├── mlperf_log_summary.txt
        └── mlperf_log_trace.json
```

## Step 2: Run the Folder Generation Script

Run the submission preparation script to format your results for the submission checker:

```bash
python prepare_submission.py \
    --input <path_to>/GB200-NVL72_GB200-186GB_aarch64x16_TRT \
    --output ./result_collection_folder \
    --user-conf <path_to>/mlperf-inference/closed/NVIDIA/src/nv_mlpinf/benchmarks/dlrm_v3/benchmarks/<corresponding_user.conf>
```

**Parameters:**
- `--input`: Path to your collected results directory
- `--output`: Destination path for the formatted submission folder
- `--user-conf`: Path to the user configuration file used during benchmarking

## Step 3: Truncate Accuracy Logs

Run the accuracy log truncation script from mlcommons repo to reduce the size of accuracy JSON files:

```bash
python /opt/inference/tools/submission/truncate_accuracy_log.py \
    --input ./result_collection_folder/ \
    --submitter NVIDIA \
    --backup ./accuracy_json_backup/
```

**Parameters:**
- `--input`: Path to the result collection folder from Step 2
- `--submitter`: Submitter name (e.g., `NVIDIA`)
- `--backup`: Backup directory for original accuracy JSON files



STEP 4: after this your folder of dlrm-v3 repo should be compliant for submission checker, merge with other benchmark results and submit.