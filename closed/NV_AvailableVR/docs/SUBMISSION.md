# Submission Log Prep

## Required Files For A Benchmark/Scenario

This section is based on this repository's MLCommons submission checker code. It documents the minimal files the checker reads for one benchmark/scenario result in a v6.0/v6.1-style submission layout.

### DSR1 using traditional LoadGen

```text
deepseek-r1/<scenario>/
  README.md
  user.conf
  measurements.json
  performance/run_1/mlperf_log_detail.txt
  accuracy/mlperf_log_detail.txt
  accuracy/mlperf_log_accuracy.json
  accuracy/accuracy.txt
  TEST06/verify_accuracy.txt
```

### GPT-OSS using traditional LoadGen

```text
gpt-oss-120b/<scenario>/
  README.md
  user.conf
  measurements.json
  performance/run_1/mlperf_log_detail.txt
  accuracy/mlperf_log_detail.txt
  accuracy/mlperf_log_accuracy.json
  accuracy/accuracy.txt
  TEST07/verify_accuracy.txt
  TEST09/verify_output_len.txt
```



### Llama2 70B using traditional LoadGen

```text
llama2-70b-99/<scenario>/ or llama2-70b-99.9/<scenario>/
  README.md
  user.conf
  measurements.json
  performance/run_1/mlperf_log_detail.txt
  accuracy/mlperf_log_detail.txt
  accuracy/mlperf_log_accuracy.json
  accuracy/accuracy.txt
  TEST06/verify_accuracy.txt
```



### Shared full-submission files

The checker also needs these files/directories outside the benchmark/scenario folder when run on a full submission root:

```text
<division>/<submitter>/systems/<system>.json
<division>/<submitter>/src/
```

`measurements.json` must be valid JSON and include non-empty values for:

```text
input_data_types
retraining
starting_weights_filename
weight_data_types
weight_transformations
```

Notes from the checker code:

- The required readme file is `README.md`; `readme.json` is not used.
- `user.conf` is required in the same directory as `measurements.json` for traditional LoadGen submissions.
- `mlperf.conf` is not required by this checker version.
- `calibration_process.adoc` is not required by this checker version.
- `mlperf_log_summary.txt` is a normal LoadGen output and may be useful for humans or other tooling, but this non-power checker path does not require it.
- `performance/run_1/mlperf_log_accuracy.json` is optional in the non-power checker path.



### Q3VL results collected with endpoints

```text
qwen3-vl-235b-a22b/<scenario>/
  README.md
  measurements.json
  config.yaml
  performance/result_summary.json
  accuracy/accuracy_results.json
```

For endpoint-style submissions, `user.conf` is not required by this checker path.

### Wan results collected with endpoints

This is still pending 

## Recommended Files For A Benchmark/Scenario

This section is based on the example log folder. It intentionally includes files that are useful for review, reproduction, or other tooling even when the submission checker does not strictly require them.

Use this DeepSeek-R1 Offline example as the recommended shape:

```text
mlperf-inference/closed/NVIDIA/docs/submission_example_log/Test_Dummy_System/deepseek-r1/Offline
```

Current tree:

```text
Offline/
|-- README.md
|-- TEST06/
|   |-- accuracy/
|   |   `-- mlperf_log_accuracy.json
|   `-- verify_accuracy.txt
|-- accuracy/
|   |-- accuracy.txt
|   |-- mlperf_log_accuracy.json
|   |-- mlperf_log_detail.txt
|   `-- mlperf_log_summary.txt
|-- measurements.json
|-- mlperf.conf
|-- performance/
|   `-- run_1/
|       |-- mlperf_log_accuracy.json
|       |-- mlperf_log_detail.txt
|       `-- mlperf_log_summary.txt
`-- user.conf
```



## Obtaining Logs From SFlow Output

Use the raw SFlow output folders as the source of truth for performance, accuracy, and compliance logs. For example:

```text
mlperf-inference/closed/NVIDIA/build/gptoss_result_collection_test/raw_logs
```

Each submitted SFlow job creates one run folder under that raw log root. In the commands below, `sflow-output` means one of those run folders, such as:

```text
mlperf-inference/closed/NVIDIA/build/gptoss_result_collection_test/raw_logs/performance/<job-id>-trtllm_ifb-<timestamp>-<suffix>
mlperf-inference/closed/NVIDIA/build/gptoss_result_collection_test/raw_logs/accuracy/<job-id>-trtllm_ifb-<timestamp>-<suffix>
mlperf-inference/closed/NVIDIA/build/gptoss_result_collection_test/raw_logs/compliance/<TESTXX>/<job-id>-trtllm_ifb-<timestamp>-<suffix>
```

The harness-generated result files live under `sflow-output/mlperf_harness/...`.

For performance logs, copy from:

```text
sflow-output/mlperf_harness/PerformanceOnly/<system>/<benchmark>/<scenario>/
```

to:

```text
<submission-scenario>/performance/run_1/
```

The files to copy are:

```text
mlperf_log_detail.txt
mlperf_log_summary.txt
mlperf_log_accuracy.json
```

For accuracy logs, copy from:

```text
sflow-output/mlperf_harness/AccuracyOnly/<system>/<benchmark>/<scenario>/
```

to:

```text
<submission-scenario>/accuracy/
```

The files to copy are:

```text
accuracy.txt
mlperf_log_accuracy.json
mlperf_log_detail.txt
mlperf_log_summary.txt
```

For compliance logs, copy the generated test directory from:

```text
sflow-output/mlperf_harness/PerformanceOnly/<system>/<benchmark>/<scenario>/<TESTXX>/
```

to:

```text
<submission-scenario>/<TESTXX>/
```

Examples of compliance files are:

```text
TEST07/verify_accuracy.txt
TEST09/verify_output_len.txt
TESTXX/performance/run_1/mlperf_log_detail.txt
TESTXX/performance/run_1/mlperf_log_summary.txt
TESTXX/accuracy/mlperf_log_accuracy.json
```

The per-scenario metadata/config files come from `build/loadgen-configs`, not from the SFlow run folder. Copy from:

```text
mlperf-inference/closed/NVIDIA/build/loadgen-configs/<system>/<benchmark>/<scenario>/
```

to the scenario root:

```text
<submission-scenario>/
```

The generated files to copy are:

```text
README.md
measurements.json
mlperf.conf
user.conf
```

For GPT-OSS x4 Offline in the example raw log folder, the concrete loadgen-config source is:

```text
mlperf-inference/closed/NVIDIA/build/loadgen-configs/GB200-NVL72_GB200-186GB_aarch64x4_TRT/gpt-oss-120b/Offline/
```

Copy skeleton:

```bash
MLPERF_DIR=mlperf-inference/closed/NVIDIA
SYSTEM=GB200-NVL72_GB200-186GB_aarch64x4_TRT
BENCHMARK=gpt-oss-120b
SCENARIO=Offline
REQUIRED_AUDIT_TESTS=(TEST07 TEST09)

PERF_SFLOW_OUTPUT="${MLPERF_DIR}/build/gptoss_result_collection_test/raw_logs/performance/<performance-run-dir>"
ACC_SFLOW_OUTPUT="${MLPERF_DIR}/build/gptoss_result_collection_test/raw_logs/accuracy/<accuracy-run-dir>"
LOADGEN_CONFIG_DIR="${MLPERF_DIR}/build/loadgen-configs/${SYSTEM}/${BENCHMARK}/${SCENARIO}"
DEST="${MLPERF_DIR}/build/submission-log-prep/${SYSTEM}/${BENCHMARK}/${SCENARIO}"

mkdir -p "${DEST}/performance/run_1" "${DEST}/accuracy"

cp "${PERF_SFLOW_OUTPUT}/mlperf_harness/PerformanceOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/mlperf_log_detail.txt" "${DEST}/performance/run_1/"
cp "${PERF_SFLOW_OUTPUT}/mlperf_harness/PerformanceOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/mlperf_log_summary.txt" "${DEST}/performance/run_1/"
cp "${PERF_SFLOW_OUTPUT}/mlperf_harness/PerformanceOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/mlperf_log_accuracy.json" "${DEST}/performance/run_1/"

cp "${ACC_SFLOW_OUTPUT}/mlperf_harness/AccuracyOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/accuracy.txt" "${DEST}/accuracy/"
cp "${ACC_SFLOW_OUTPUT}/mlperf_harness/AccuracyOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/mlperf_log_accuracy.json" "${DEST}/accuracy/"
cp "${ACC_SFLOW_OUTPUT}/mlperf_harness/AccuracyOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/mlperf_log_detail.txt" "${DEST}/accuracy/"
cp "${ACC_SFLOW_OUTPUT}/mlperf_harness/AccuracyOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/mlperf_log_summary.txt" "${DEST}/accuracy/"

for TEST_NAME in "${REQUIRED_AUDIT_TESTS[@]}"; do
  COMP_SFLOW_OUTPUT="${MLPERF_DIR}/build/gptoss_result_collection_test/raw_logs/compliance/${TEST_NAME}/<${TEST_NAME}-run-dir>"
  cp -a "${COMP_SFLOW_OUTPUT}/mlperf_harness/PerformanceOnly/${SYSTEM}/${BENCHMARK}/${SCENARIO}/${TEST_NAME}" "${DEST}/"
done

cp "${LOADGEN_CONFIG_DIR}/README.md" "${DEST}/"
cp "${LOADGEN_CONFIG_DIR}/measurements.json" "${DEST}/"
cp "${LOADGEN_CONFIG_DIR}/mlperf.conf" "${DEST}/"
cp "${LOADGEN_CONFIG_DIR}/user.conf" "${DEST}/"
```



## Truncating Accuracy Logs

Before running the submission checker or pushing a results package, truncate every submitted `mlperf_log_accuracy.json`. The checker rejects large accuracy logs. The truncation tool keeps the first 4 KB and last 4 KB of each log, writes the original full-log SHA256 into `accuracy.txt`, and stores a backup of the original full logs.

Run the tool from the MLPerf Inference checkout root:

```bash
cd mlperf-inference/closed/NVIDIA

SUBMISSION_ROOT=build/artifacts
SUBMITTER=NVIDIA

rm -rf build/full_results
rm -rf "${SUBMISSION_ROOT}/closed/${SUBMITTER}/build/full_results"

python3 3rdparty/mlc-inference/tools/submission/truncate_accuracy_log.py \
  --input "${SUBMISSION_ROOT}" \
  --submitter "${SUBMITTER}" \
  --backup "closed/${SUBMITTER}/build/full_results"

mv "${SUBMISSION_ROOT}/closed/${SUBMITTER}/build/full_results" build/full_results
```

Use `--scenarios-to-skip` if the submission root intentionally does not contain all scenarios. For example:

```bash
python3 3rdparty/mlc-inference/tools/submission/truncate_accuracy_log.py \
  --input "${SUBMISSION_ROOT}" \
  --submitter "${SUBMITTER}" \
  --backup "closed/${SUBMITTER}/build/full_results" \
  --scenarios-to-skip "Server,Interactive"
```

After truncation, each submitted accuracy log should be small and each accuracy directory should have an `accuracy.txt` hash:

```bash
find "${SUBMISSION_ROOT}/closed/${SUBMITTER}/results" \
  -path '*/accuracy/mlperf_log_accuracy.json' \
  -printf '%p %s bytes\n' | sort
```

Example output for GPT-OSS TEST07/TEST09 after truncation:

```text
build/artifacts/closed/NVIDIA/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT/gpt-oss-120b/Interactive/TEST07/accuracy/mlperf_log_accuracy.json 8199 bytes
build/artifacts/closed/NVIDIA/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT/gpt-oss-120b/Interactive/TEST09/accuracy/mlperf_log_accuracy.json 8199 bytes
build/artifacts/closed/NVIDIA/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT/gpt-oss-120b/Offline/TEST07/accuracy/mlperf_log_accuracy.json 8199 bytes
build/artifacts/closed/NVIDIA/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT/gpt-oss-120b/Offline/TEST09/accuracy/mlperf_log_accuracy.json 8199 bytes
build/artifacts/closed/NVIDIA/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT/gpt-oss-120b/Server/TEST07/accuracy/mlperf_log_accuracy.json 8199 bytes
build/artifacts/closed/NVIDIA/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT/gpt-oss-120b/Server/TEST09/accuracy/mlperf_log_accuracy.json 8199 bytes
```

Check the generated hash file:

```bash
cat "${SUBMISSION_ROOT}/closed/${SUBMITTER}/results/GB200-NVL72_GB200-186GB_aarch64x72_TRT/gpt-oss-120b/Server/TEST07/accuracy/accuracy.txt"
```

Example:

```text
hash=c6bb2d89b47cd0b55875e9e2906027093e85aa2ec1e4a4cddeed93494ddc7668
```

The full untruncated accuracy logs are kept under:

```text
mlperf-inference/closed/NVIDIA/build/full_results
```



## Export Submission And Run Submission Checker

Before exporting the submission package, inspect every benchmark/scenario that will be submitted. For each submitted benchmark, verify:

- All required scenarios for that benchmark are present.
- Each scenario has the required performance and accuracy logs.
- Each required compliance test is present and has its verification output.
- Every submitted `accuracy/mlperf_log_accuracy.json` and compliance `TESTXX/accuracy/mlperf_log_accuracy.json` has been truncated.
- Every truncated accuracy directory has an `accuracy.txt` containing the full-log hash.

Stage the result folders under `build/artifacts/closed/NVIDIA/results`, following this shape:

```text
mlperf-inference/closed/NVIDIA/build/artifacts/closed/NVIDIA/results/<system>/<benchmark>/<scenario>/
```

For this manual packaging flow, `build/artifacts` is the source of truth. Do not place manually assembled result folders under `build/submission-staging`; `make export_submission` reads from `build/artifacts`.

Use the dummy example as a reference for the result-folder shape:

```text
mlperf-inference/closed/NVIDIA/build/artifacts/closed/NVIDIA/results/Test_Dummy_System
```

The staged tree should look like:

```text
build/artifacts/closed/NVIDIA/results/
`-- <system>/
    `-- <benchmark>/
        |-- Offline/
        |-- Server/
        `-- Interactive/
```

Not every benchmark requires all three scenarios. Follow the benchmark's required scenario set, and do not leave half-populated scenario folders in the package.

Run these commands from `mlperf-inference/closed/NVIDIA`:

```bash
cd mlperf-inference/closed/NVIDIA

make export_submission
make check_submission_fast
```

`make export_submission` copies staged artifacts from `build/artifacts` into the repository result folders and then creates the final submission package under:

```text
mlperf-inference/build/submission
```

`make check_submission_fast` runs the MLCommons submission checker against that exported package. The checker log is written to:

```text
mlperf-inference/build/submission/closed/NVIDIA/results/submission_checker_log.txt
```

These Makefile targets must be run from the host checkout, not from inside a container. They need access to the project root at `mlperf-inference/`, including paths outside `closed/NVIDIA`.

After `make check_submission_fast` finishes, inspect the checker log and only package the submission if it passed with no errors:

```bash
less mlperf-inference/build/submission/closed/NVIDIA/results/submission_checker_log.txt
```

Then run:

```bash
cd mlperf-inference/closed/NVIDIA
make pack_submission
```

This generates the final submission tarball and SHA1 file under:

```text
mlperf-inference/build/submission/mlperf-inference-NVIDIA-submission.tar.gz
mlperf-inference/build/submission/mlperf-inference-NVIDIA-submission.sha1
```

