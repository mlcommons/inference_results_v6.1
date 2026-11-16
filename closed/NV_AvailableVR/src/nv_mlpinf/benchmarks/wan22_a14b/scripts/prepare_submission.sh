#!/bin/bash
# Assemble a wan22 (wan-2.2-t2v-a14b) endpoint submission folder for ONE
# system+scenario, from an accuracy run and a TEST04 audit run, matching the
# MLCommons reference layout (results/<system>/wan-2.2-t2v-a14b/<scenario>/).
#
# It copies the run output, stages the 10 human-audit videos under their official
# MLCommons sample IDs, prunes run-only artifacts that the reference omits
# (.ready markers, per-sample events.jsonl, sample_idx_map.json), and writes
# measurements.json + README.md. Run once per scenario (Offline, SingleStream).
#
# Usage:
#   prepare_submission.sh --system <SYSTEM> --scenario Offline|SingleStream \
#                         --acc-run   <accuracy run>/endpoint_benchmark \
#                         --audit-run <TEST04 audit run>/endpoint_benchmark \
#                         --out <submission root> \
#                         [--videos-dir results/wan22-a14b/videos/<job-id>] \
#                         [--samples-map <path to samples_filename_ids.txt>] \
#                         [--measurements <measurements.json to copy>] \
#                         [--run-checker]
#
# --videos-dir : dir holding the raw video files, matched by the vbench_videos
#                symlink target's basename. Needed on the host, where those
#                symlinks point at the in-container /work/... path. Omit only
#                when they resolve directly (e.g. run in-container).
#                Each run writes to results/wan22-a14b/videos/$SLURM_JOB_ID, so
#                pass that run's job-id dir, not the parent. Read the id off the
#                run with:
#                  readlink <acc-run>/vbench_videos/*-0.mp4 | head -1
# --measurements : copy this file instead of writing the default 5-field JSON.
# --run-checker  : run the v6.1 submission_checker on <out> afterwards (also
#                  needs closed/NVIDIA/systems/<SYSTEM>.json and src/).
#
# Output: <out>/closed/NVIDIA/results/<SYSTEM>/wan-2.2-t2v-a14b/<scenario>/
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# scripts -> wan22_a14b -> benchmarks -> nv_mlpinf -> src -> closed/NVIDIA
CLOSED_NVIDIA=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
DEFAULT_MAP="$CLOSED_NVIDIA/3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/samples_filename_ids.txt"

SYSTEM=""; SCENARIO=""; ACC_RUN=""; AUDIT_RUN=""; OUT=""
VIDEOS_DIR=""; SAMPLES_MAP="$DEFAULT_MAP"; MEASUREMENTS=""; RUN_CHECKER=0

usage() { awk 'NR>1{ if(/^#/){sub(/^# ?/,"");print} else exit }' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --system)       SYSTEM="$2"; shift 2 ;;
        --scenario)     SCENARIO="$2"; shift 2 ;;
        --acc-run)      ACC_RUN="$2"; shift 2 ;;
        --audit-run)    AUDIT_RUN="$2"; shift 2 ;;
        --out)          OUT="$2"; shift 2 ;;
        --videos-dir)   VIDEOS_DIR="$2"; shift 2 ;;
        --samples-map)  SAMPLES_MAP="$2"; shift 2 ;;
        --measurements) MEASUREMENTS="$2"; shift 2 ;;
        --run-checker)  RUN_CHECKER=1; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "ERROR: unknown arg '$1'" >&2; usage 1 ;;
    esac
done

[ -n "$SYSTEM" ]    || { echo "ERROR: --system is required" >&2; usage 1; }
[ -n "$SCENARIO" ]  || { echo "ERROR: --scenario is required" >&2; usage 1; }
[ -n "$ACC_RUN" ]   || { echo "ERROR: --acc-run is required" >&2; usage 1; }
[ -n "$AUDIT_RUN" ] || { echo "ERROR: --audit-run is required" >&2; usage 1; }
[ -n "$OUT" ]       || { echo "ERROR: --out is required" >&2; usage 1; }
case "$SCENARIO" in Offline|SingleStream) ;; *) echo "ERROR: --scenario must be Offline or SingleStream" >&2; exit 1 ;; esac
[ -d "$ACC_RUN" ]   || { echo "ERROR: acc-run dir not found: $ACC_RUN" >&2; exit 1; }
[ -d "$AUDIT_RUN/audit" ] || { echo "ERROR: no audit/ under audit-run: $AUDIT_RUN" >&2; exit 1; }
[ -z "$MEASUREMENTS" ] || [ -f "$MEASUREMENTS" ] || { echo "ERROR: --measurements not found: $MEASUREMENTS" >&2; exit 1; }
# Guard --videos-dir up front: a wrong value (e.g. a stale container path on the
# host) would otherwise mark every audit sample MISSING and blame vbench_videos.
[ -z "$VIDEOS_DIR" ] || [ -d "$VIDEOS_DIR" ] || { echo "ERROR: videos-dir not found: $VIDEOS_DIR" >&2; exit 1; }
[ -f "$SAMPLES_MAP" ] || { echo "ERROR: samples map not found: $SAMPLES_MAP" >&2
    echo "       (init the mlc-inference submodule, or pass --samples-map)" >&2; exit 1; }

DST="$OUT/closed/NVIDIA/results/$SYSTEM/wan-2.2-t2v-a14b/$SCENARIO"
mkdir -p "$DST/accuracy"

# Full accuracy-run output (config, run report, perf, metrics, vbench, accuracy).
cp "$ACC_RUN/config.yaml" "$ACC_RUN/report.txt"                       "$DST/"
cp -r "$ACC_RUN/performance" "$ACC_RUN/metrics" "$ACC_RUN/vbench_results" "$DST/"
cp "$ACC_RUN/accuracy/accuracy_results.json" "$DST/accuracy/accuracy_results.json"

# Human-audit sample videos: the checker requires exactly the 10 <id>.mp4 named in
# the upstream samples_filename_ids.txt, plus captions.txt. Each map line is
# '<prompt>-0.mp4, <official_id>.mp4' and the prompt itself can contain ', ', so
# split on the LAST ', '. vbench_videos/<prompt>-0.mp4 symlinks the generated clip.
# Rebuilt from scratch: the checker requires EXACTLY these 10, so a leftover clip
# from an earlier staging into the same --out would fail it.
VIDEOS_OUT="$DST/accuracy/videos"
CAPTIONS="$VIDEOS_OUT/captions.txt"
rm -rf "$VIDEOS_OUT"
mkdir -p "$VIDEOS_OUT"
: > "$CAPTIONS"

missing=0
copied=0
while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] || continue
    src="${line%, *}"
    id="${line##*, }"
    link="$ACC_RUN/vbench_videos/$src"

    if [ -L "$link" ] && [ -n "$VIDEOS_DIR" ]; then
        raw="$VIDEOS_DIR/$(basename "$(readlink "$link")")"
    else
        raw="$link"            # symlink resolves directly, or it is a real file
    fi

    if [ ! -e "$raw" ]; then
        echo "MISSING: $src -> $raw" >&2
        missing=$((missing + 1))
        continue
    fi
    # The server image bakes in ffmpeg and always emits MP4; an AVI means the run
    # used an older image and cannot be submitted for human audit.
    case "$raw" in
        *.mp4|*.MP4) ;;
        *) echo "ERROR: non-mp4 source $raw" >&2
           echo "       re-run the accuracy run on the v6.1-jul24+ server image" >&2; exit 1 ;;
    esac

    cp -- "$raw" "$VIDEOS_OUT/$id"
    printf '%s\t%s\n' "$id" "${src%-0.mp4}" >> "$CAPTIONS"
    echo "  $id  <-  $src"
    copied=$((copied + 1))
done < "$SAMPLES_MAP"

echo "Staged $copied audit video(s) + captions.txt to $VIDEOS_OUT"
if [ "$missing" -gt 0 ]; then
    echo "ERROR: $missing required audit sample(s) missing from $ACC_RUN/vbench_videos" >&2
    exit 1
fi

# Full TEST04 audit output (audit json + verify txt + output_caching/ + reference/).
# rm first: cp -r onto an existing same-named dir nests it ($DST/audit/audit) and
# leaves the previous run's results in place, so a re-stage would ship the old audit.
rm -rf "$DST/audit"
cp -r "$AUDIT_RUN/audit" "$DST/audit"

# Prune run-only artifacts the reference submission omits.
find "$DST" \( -name '.ready' -o -name 'events.jsonl' -o -name 'sample_idx_map.json' \) -delete

# measurements.json: copy if provided, else the default 5 required fields.
if [ -n "$MEASUREMENTS" ]; then
    cp "$MEASUREMENTS" "$DST/measurements.json"
else
    cat > "$DST/measurements.json" <<'JSON'
{
    "input_data_types": "bf16",
    "retraining": "No",
    "starting_weights_filename": "Original Huggingface model weights",
    "weight_data_types": "fp8",
    "weight_transformations": "quantization"
}
JSON
fi

# README.md: short pointer to the setup + benchmark guide (matches the reference).
cat > "$DST/README.md" <<EOF
To run this benchmark, first follow the setup steps in closed/NVIDIA/README.md and the Wan2.2-T2V-A14B benchmark guide in closed/NVIDIA/src/nv_mlpinf/benchmarks/wan22_a14b/README.md. Then launch the Wan2.2 $SYSTEM $SCENARIO videogen endpoint workflow from closed/NVIDIA.
EOF

echo "Wrote submission tree to $DST"
echo "NOTE: also provide closed/NVIDIA/systems/$SYSTEM.json and a closed/NVIDIA/src/ dir at the submission root."

if [ "$RUN_CHECKER" -eq 1 ]; then
    echo "== running submission_checker (v6.1) =="
    ( cd "$CLOSED_NVIDIA/3rdparty/mlc-inference/tools/submission" \
      && python3 -m submission_checker.main --input "$(realpath "$OUT")" \
           --version v6.1 --submitter NVIDIA --skip-power-check )
fi
