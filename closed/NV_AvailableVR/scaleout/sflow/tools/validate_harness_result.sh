#!/bin/bash
# Validate MLPerf harness result artifacts for nv-sflow runs.
#
# This is shared by:
#   - nv-sflow templates, inside the harness container via /work/scaleout/sflow/tools
#   - L0/L2 CI shell scripts, on the login node via scaleout/sflow/tools

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: validate_harness_result.sh --mode MODE --log-dir DIR [--output-file FILE] [--benchmark BENCHMARK] [--audit-test TEST]

Required:
  --mode MODE       PerformanceOnly, AccuracyOnly, Compliance, or Submission
  --log-dir DIR     Directory containing mlperf_log_summary.txt/detail.txt, or a parent

Optional:
  --output-file FILE  Captured harness/sflow output. Required for strict AccuracyOnly validation.
  --benchmark NAME    Benchmark name. Required for Submission compliance validation.
  --audit-test TEST  Compliance test name. Required for Compliance mode.
EOF
}

MODE=""
LOG_DIR=""
OUTPUT_FILE=""
BENCHMARK=""
AUDIT_TEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="${2:-}"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="${2:-}"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="${2:-}"
            shift 2
            ;;
        --audit-test)
            AUDIT_TEST="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" || -z "${LOG_DIR}" ]]; then
    echo "ERROR: --mode and --log-dir are required" >&2
    usage >&2
    exit 2
fi

if [[ "${MODE}" != "PerformanceOnly" && "${MODE}" != "AccuracyOnly" && "${MODE}" != "Compliance" && "${MODE}" != "Submission" ]]; then
    echo "ERROR: unsupported mode ${MODE}; expected PerformanceOnly, AccuracyOnly, Compliance, or Submission" >&2
    exit 2
fi

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "ERROR: log directory not found: ${LOG_DIR}" >&2
    exit 1
fi

validate_single_mode() {
    local mode="$1"
    local log_dir="$2"
    local output_file="${3:-}"
    local harness_ok=true
    local summary_file=""
    local detail_file=""

    echo ""
    echo "========================================="
    echo "=== Validating MLPerf harness (${mode}) ==="
    echo "=== Log dir: ${log_dir}"
    echo "========================================="

    if [[ ! -d "${log_dir}" ]]; then
        echo "ERROR: log directory not found: ${log_dir}" >&2
        return 1
    fi

    summary_file=$(find "${log_dir}" -name "mlperf_log_summary.txt" -print -quit 2>/dev/null || true)
    if [[ -n "${summary_file}" && -s "${summary_file}" ]]; then
        echo "========================================="
        echo "=== MLPerf Log Summary (${mode}) ==="
        echo "========================================="
        cat "${summary_file}"
        echo ""
    else
        echo "ERROR: mlperf_log_summary.txt not found or empty in ${log_dir}"
        echo "This indicates the harness did not complete."
        harness_ok=false
    fi

    if [[ "${harness_ok}" == "true" && "${mode}" == "PerformanceOnly" ]]; then
        if grep -qE '^Result is : VALID' "${summary_file}"; then
            echo "PASSED: mlperf_log_summary.txt reports Result is : VALID"
        else
            echo "FAILED: mlperf_log_summary.txt does not report VALID result:"
            grep -E '^Result is :' "${summary_file}" || echo "  (no 'Result is :' line found)"
            harness_ok=false
        fi
    elif [[ "${harness_ok}" == "true" && "${mode}" == "AccuracyOnly" ]]; then
        if [[ -z "${output_file}" || ! -f "${output_file}" ]]; then
            echo "ERROR: --output-file is required for AccuracyOnly validation"
            harness_ok=false
        elif grep -qE '\|\s+No\s+\|' "${output_file}" 2>/dev/null; then
            echo "FAILED: Accuracy threshold NOT met (All Acc. Pass? = No):"
            grep -nE '\|\s+No\s+\|' "${output_file}" | head -10
            harness_ok=false
        else
            echo "PASSED: No accuracy threshold miss detected in harness output"
        fi
    fi

    detail_file=$(find "${log_dir}" -name "mlperf_log_detail.txt" -print -quit 2>/dev/null || true)
    if [[ -n "${detail_file}" && -f "${detail_file}" ]]; then
        if grep -q '"is_error": true' "${detail_file}" 2>/dev/null; then
            error_count=$(grep -c '"is_error": true' "${detail_file}")
            echo "FAILED: Found ${error_count} error(s) in mlperf_log_detail.txt"
            grep -B2 -A2 '"is_error": true' "${detail_file}"
            harness_ok=false
        else
            echo "PASSED: No errors in mlperf_log_detail.txt"
        fi
    elif [[ "${harness_ok}" == "true" ]]; then
        echo "WARNING: mlperf_log_detail.txt not found"
    fi

    [[ "${harness_ok}" == "true" ]]
}

validate_submission_compliance() {
    local submission_dir="$1"
    local benchmark="$2"
    local compliance_dir="${submission_dir}/compliance"
    local compliance_ok=true

    echo ""
    echo "========================================="
    echo "=== Validating MLPerf submission compliance ==="
    echo "=== Compliance dir: ${compliance_dir}"
    echo "========================================="

    if [[ ! -d "${compliance_dir}" ]]; then
        echo "ERROR: compliance directory not found: ${compliance_dir}" >&2
        return 1
    fi

    case "${benchmark}" in
        gpt-oss-120b)
            # TEST07 is intentionally skipped for GPT OSS because it is flaky when LoadGen seed changes.
            echo "SKIPPED: GPT OSS TEST07 validation"
            if grep -Rqs "Overall: TEST PASS" "${compliance_dir}/test09"* 2>/dev/null; then
                echo "PASSED: TEST09 Overall: TEST PASS detected"
            else
                echo "FAILED: TEST09 Overall: TEST PASS not detected"
                compliance_ok=false
            fi
            ;;
        deepseek-r1|llama2-70b)
            if grep -RqsE "First token check pass: (True|Skipped)" "${compliance_dir}/test06"* 2>/dev/null; then
                echo "PASSED: TEST06 First token check pass: True or Skipped"
            else
                echo "FAILED: TEST06 First token check pass: True/Skipped not detected"
                compliance_ok=false
            fi

            for pattern in \
                "EOS check pass: True" \
                "Sample length check pass: True"; do
                if grep -Rqs "${pattern}" "${compliance_dir}/test06"* 2>/dev/null; then
                    echo "PASSED: TEST06 ${pattern}"
                else
                    echo "FAILED: TEST06 ${pattern} not detected"
                    compliance_ok=false
                fi
            done
            ;;
        *)
            echo "ERROR: unsupported benchmark for submission compliance validation: ${benchmark}" >&2
            compliance_ok=false
            ;;
    esac

    [[ "${compliance_ok}" == "true" ]]
}

validate_compliance_test() {
    local audit_test="$1"
    local log_dir="$2"
    local output_file="${3:-}"
    local compliance_ok=true
    local search_paths=()

    echo ""
    echo "========================================="
    echo "=== Validating MLPerf compliance ${audit_test} ==="
    echo "=== Log dir: ${log_dir}"
    echo "========================================="

    if [[ -d "${log_dir}" ]]; then
        search_paths+=("${log_dir}")
    fi
    if [[ -n "${output_file}" && -f "${output_file}" ]]; then
        search_paths+=("${output_file}")
    fi
    if [[ "${#search_paths[@]}" -eq 0 ]]; then
        echo "ERROR: no compliance log directory or output file found" >&2
        return 1
    fi

    case "${audit_test}" in
        TEST07)
            if grep -Rqs "Accuracy check pass: True" "${search_paths[@]}" 2>/dev/null; then
                echo "PASSED: TEST07 accuracy check pass detected"
            else
                echo "FAILED: TEST07 accuracy check pass not detected"
                compliance_ok=false
            fi
            ;;
        TEST09)
            if grep -Rqs "Overall: TEST PASS" "${search_paths[@]}" 2>/dev/null; then
                echo "PASSED: TEST09 Overall: TEST PASS detected"
            else
                echo "FAILED: TEST09 Overall: TEST PASS not detected"
                compliance_ok=false
            fi
            ;;
        TEST06)
            if grep -RqsE "First token check pass: (True|Skipped)" "${search_paths[@]}" 2>/dev/null; then
                echo "PASSED: TEST06 First token check pass: True or Skipped"
            else
                echo "FAILED: TEST06 First token check pass: True/Skipped not detected"
                compliance_ok=false
            fi

            for pattern in \
                "EOS check pass: True" \
                "Sample length check pass: True"; do
                if grep -Rqs "${pattern}" "${search_paths[@]}" 2>/dev/null; then
                    echo "PASSED: TEST06 ${pattern}"
                else
                    echo "FAILED: TEST06 ${pattern} not detected"
                    compliance_ok=false
                fi
            done
            ;;
        *)
            echo "ERROR: unsupported audit test for compliance validation: ${audit_test}" >&2
            compliance_ok=false
            ;;
    esac

    [[ "${compliance_ok}" == "true" ]]
}

if [[ "${MODE}" == "Compliance" ]]; then
    if [[ -z "${AUDIT_TEST}" ]]; then
        echo "ERROR: --audit-test is required for Compliance validation" >&2
        exit 2
    fi
    validate_compliance_test "${AUDIT_TEST}" "${LOG_DIR}" "${OUTPUT_FILE}"
    exit $?
fi

if [[ "${MODE}" == "Submission" ]]; then
    if [[ -z "${BENCHMARK}" ]]; then
        echo "ERROR: --benchmark is required for Submission validation" >&2
        exit 2
    fi

    VALIDATION_RC=0
    validate_single_mode "PerformanceOnly" "${LOG_DIR}/performance" "${LOG_DIR}/performance.log" || VALIDATION_RC=1
    validate_single_mode "AccuracyOnly" "${LOG_DIR}/accuracy" "${LOG_DIR}/accuracy.log" || VALIDATION_RC=1
    validate_submission_compliance "${LOG_DIR}" "${BENCHMARK}" || VALIDATION_RC=1
    exit "${VALIDATION_RC}"
fi

validate_single_mode "${MODE}" "${LOG_DIR}" "${OUTPUT_FILE}"
