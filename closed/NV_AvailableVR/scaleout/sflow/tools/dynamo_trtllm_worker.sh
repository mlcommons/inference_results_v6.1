#!/bin/bash
# One dynamo.trtllm worker rank (prefill or decode) of the sflow server tasks, inside the server image.
# Env: DYN_ROLE=prefill|decode, ENGINE_YAML (TensorRT-LLM LLM-API options), WORKER_ENV_YAML (KEY: value),
# MODEL_PATH, SERVED_MODEL_NAME, WORK_DIR, CLIENT_NODE_IP, ETCD_PORT, NATS_PORT, DYN_NAMESPACE,
# optional SITE_ENV_YAML, DYN_WORKER_EXTRA_ARGS.
set -euo pipefail
trap 'rc=$?; echo "WORKER_PREEXEC_FAILED rc=$rc line $LINENO: $BASH_COMMAND" >&2' ERR
: "${DYN_ROLE:?}" "${ENGINE_YAML:?}" "${WORKER_ENV_YAML:?}" "${MODEL_PATH:?}" "${SERVED_MODEL_NAME:?}" "${WORK_DIR:?}"
TOOLS="${WORK_DIR}/scaleout/sflow/tools"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp
R=${SLURM_PROCID:-0}
B="/tmp/dsr1-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}-${SLURM_STEP_ID:-0}-${R}"
export HF_MODULES_CACHE="$B/hf-modules" XDG_CACHE_HOME="$B/cache" HOME="$B/home"
mkdir -p "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" "$HOME"
ENV_EXPORTS=$(python3 "${TOOLS}/run_with_env.py" "${WORKER_ENV_YAML}")
eval "${ENV_EXPORTS}"
if [ -n "${SITE_ENV_YAML:-}" ]; then ENV_EXPORTS=$(python3 "${TOOLS}/run_with_env.py" "${SITE_ENV_YAML}"); eval "${ENV_EXPORTS}"; fi
export DYN_NAMESPACE="${DYN_NAMESPACE:-dynamo}" DYN_DISCOVERY_BACKEND=etcd \
  ETCD_ENDPOINTS="http://${CLIENT_NODE_IP}:${ETCD_PORT:-2379}" NATS_SERVER="nats://${CLIENT_NODE_IP}:${NATS_PORT:-4222}" \
  DYN_REQUEST_PLANE=tcp
ENG="$HOME/engine-${DYN_ROLE}.yaml"
python3 - "$ENGINE_YAML" "$ENG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
dropped = {k: cfg.pop(k) for k in ("num_serve_frontends",) if k in cfg}
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
print(f"[worker] engine yaml {sys.argv[1]} -> {sys.argv[2]} (dropped trtllm-serve-only keys: {dropped})", flush=True)
PY
KVB=$(python3 -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])) or {}; print(int((c.get("kv_cache_config") or {}).get("tokens_per_block", 32)))' "$ENG")
echo "[worker] role=${DYN_ROLE} replica=${SFLOW_REPLICA_INDEX:-0} rank=${R} host=$(hostname -s) cvd=${CUDA_VISIBLE_DEVICES:-} ns=${DYN_NAMESPACE} kv_block=${KVB} ucx_tls=${UCX_TLS:-} ucx_dev=${UCX_NET_DEVICES:-}"
LAUNCH_ENV=(env)
while IFS= read -r item; do LAUNCH_ENV+=("${item}"); done < <(env | grep '^UCX_' || true)
trap - ERR
set -x
# shellcheck disable=SC2086
exec trtllm-llmapi-launch "${LAUNCH_ENV[@]}" \
  python3 -m dynamo.trtllm --model-path "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" \
  --extra-engine-args "$ENG" --disaggregation-mode "${DYN_ROLE}" --kv-block-size "$KVB" ${DYN_WORKER_EXTRA_ARGS:-}
