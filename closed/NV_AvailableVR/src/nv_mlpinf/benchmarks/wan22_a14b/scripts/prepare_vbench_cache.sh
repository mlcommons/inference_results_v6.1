#!/bin/bash
# Pre-stage ALL VBench scorer model weights for wan22 accuracy runs (offline).
#
# The mlcommons endpoints `-vbench` image ships the VBench venv but NONE of the 6
# MLPerf wan22 dimension weights (verified by inspecting the published image; its
# /opt/vbench_cache is an empty dir). VBench would otherwise download them at
# evaluation time, which FAILS on compute nodes that run offline (HF_HUB_OFFLINE=1,
# no wget/unzip in the client image). Run this once on a login node (needs
# curl/wget + unzip + git + network) so every dimension scores offline.
#
# This host CACHE_DIR (default build/vbench_cache) sits under WORK_DIR, which is
# mounted at /work in-container -- so it is reachable at /work/build/vbench_cache
# with no extra bind-mount (perf-only / audit runs that never stage it are
# unaffected). The accuracy template (wan22_videogen_endpoints_submission.yaml)
# exports:
#   VBENCH_CACHE_DIR=/work/build/vbench_cache          (amt/raft/caption)
#   TORCH_HOME=/work/build/vbench_cache/torch_home     (DINO torch.hub cache)
# and endpoint_env_vbench.yaml exports:
#   HOME=/work/build/vbench_cache/home                 (CLIP -> $HOME/.cache/clip)
# This script populates all three subtrees so those paths resolve offline.
#
# The scene dim ALSO needs bert-base-uncased (Tag2Text's init_tokenizer() loads a
# BertTokenizer AND BertModel weights). VBench reads that from the HF hub cache that
# endpoint_env_vbench.yaml points at (HF_HUB_CACHE=<scratch>/.hf_cache/hub) -- a
# DIFFERENT location than build/vbench_cache -- so it is staged there too (below).
# Without it, offline runs fail: first "Can't load tokenizer for 'bert-base-uncased'",
# then "does not appear to have a file named pytorch_model.bin".
#
# Usage: prepare_vbench_cache.sh [CACHE_DIR] [HF_CACHE_DIR]
#   CACHE_DIR    default: build/vbench_cache
#   HF_CACHE_DIR default: ${MLPERF_SCRATCH_PATH}/models/wan22-a14b/.hf_cache/hub
set -euo pipefail

CACHE_DIR=${1:-build/vbench_cache}
HF_CACHE_DIR=${2:-${MLPERF_SCRATCH_PATH:-/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone}/models/wan22-a14b/.hf_cache/hub}
mkdir -p "$CACHE_DIR"/{amt_model,raft_model,caption_model,torch_home/hub/checkpoints,home/.cache/clip}
cd "$CACHE_DIR"

dl() {  # dl <url> <dest-file>
    [ -f "$2" ] && { echo "  have $2"; return; }
    if command -v curl >/dev/null; then curl -fsSL -o "$2" "$1"
    else wget -q -O "$2" "$1"; fi
}

echo "== 3 wget-based dims (motion_smoothness, dynamic_degree, scene) =="
# motion_smoothness — AMT-S checkpoint
dl https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth amt_model/amt-s.pth
# dynamic_degree — RAFT models (VBench expects raft_model/models/raft-things.pth)
if [ ! -f raft_model/models/raft-things.pth ]; then
    dl https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip raft_model/models.zip
    unzip -q -o -d raft_model/ raft_model/models.zip && rm -f raft_model/models.zip
fi
# scene — Tag2Text checkpoint
dl https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth \
   caption_model/tag2text_swin_14m.pth
# scene (cont.) — bert-base-uncased tokenizer + BertModel weights, staged into the HF
# hub-cache layout that VBench reads offline (HF_HUB_CACHE, on the scratch mount --
# NOT under CACHE_DIR). SHA-pinned resolve URLs keep the snapshot dir name and the
# downloaded file provenance in sync (plain `curl -fsSL` follows the LFS redirect).
BERT_SHA=86b5e0934494bd15c9632b12f734a8a67f723594  # bert-base-uncased pinned commit
BERT_SNAP="$HF_CACHE_DIR/models--bert-base-uncased/snapshots/$BERT_SHA"
mkdir -p "$BERT_SNAP" "$HF_CACHE_DIR/models--bert-base-uncased/refs"
printf '%s' "$BERT_SHA" > "$HF_CACHE_DIR/models--bert-base-uncased/refs/main"
for _bf in config.json vocab.txt tokenizer_config.json tokenizer.json pytorch_model.bin; do
    dl "https://huggingface.co/bert-base-uncased/resolve/$BERT_SHA/$_bf" "$BERT_SNAP/$_bf"
done

echo "== subject_consistency — DINO (torch.hub github checkout + weights) =="
# VBench (local=False) calls torch.hub.load('facebookresearch/dino:main',
# 'dino_vitb16', source='github'); with the repo pre-cloned under
# $TORCH_HOME/hub and validation bypassed by vbench_runner, it loads offline.
if [ ! -d torch_home/hub/facebookresearch_dino_main ]; then
    git clone --depth 1 https://github.com/facebookresearch/dino \
        torch_home/hub/facebookresearch_dino_main
fi
dl https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth \
   torch_home/hub/checkpoints/dino_vitbase16_pretrain.pth

echo "== background_consistency + appearance_style — CLIP ViT-B/32 =="
# VBench (local=False) calls clip.load('ViT-B/32'), which resolves to
# $HOME/.cache/clip/ViT-B-32.pt. sha256 is verified by the clip loader.
dl "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
   home/.cache/clip/ViT-B-32.pt

echo "VBench cache ready at $(pwd):"
du -sh amt_model raft_model caption_model torch_home home 2>/dev/null || true
echo "bert-base-uncased (scene) staged at $HF_CACHE_DIR/models--bert-base-uncased:"
du -sh "$HF_CACHE_DIR/models--bert-base-uncased" 2>/dev/null || true
