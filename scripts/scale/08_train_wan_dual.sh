#!/bin/bash
# E8: Wan-1.3B dual-stream latent diffusion on the 910B cluster.
#
# Same harness as E7 (train_dual_diffusion.py, online encoding, absolute
# target + z0 clean-past, 10k oft subset) with ONE variable changed: the
# generator is the pretrained Wan2.1 backbone via WanCompactAdapter instead
# of the from-scratch DiT. Compare its curves directly against
# dual_diffusion_absolute (E7-a):
#
#   Wan clearly below E7-a vmse by ~2k steps -> prior connected, main road.
#   Wan ~= E7-a                              -> prior unused, inspect adapter.
#   Wan worse/unstable                       -> prior mismatch, add alignment.
#
# Wan specifics: 1.3B backbone loaded fresh from WAN_CKPT_DIR every run
# (checkpoints store trainable params only); frozen weights bf16; full-QKV
# unfreeze (TRAIN_QKV_LAST_N=0) — affordable on 1.3B, unlike the 14B last-4
# compromise; text conditioning off per project decision.
#
#   bash scripts/scale/08_train_wan_dual.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Capture an explicit user override BEFORE the config's candidate list runs —
# that list prefers the leftover 14B checkpoint in the ref dir, which with
# full-QKV unfreeze is exactly the 3.2B-trainable / 12.6GiB-reducer OOM.
USER_WAN_CKPT_DIR="${WAN_CKPT_DIR:-}"
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# --- Wan checkpoint: 1.3B ONLY for this stage ---------------------------------
if [[ -n "${USER_WAN_CKPT_DIR}" ]]; then
  WAN_CKPT_DIR="${USER_WAN_CKPT_DIR}"
else
  WAN_CKPT_DIR=""
  for candidate in \
    "${VGGAE_REF_ROOT}/Wan2.1/checkpoints/Wan2.1-T2V-1.3B" \
    "${VGGAE_REF_ROOT}/Wan2.1-T2V-1.3B" \
    "${VGGAE_REF_ROOT}/checkpoints/Wan2.1-T2V-1.3B" \
    "${PROJECT}/Wan2.1/checkpoints/Wan2.1-T2V-1.3B"; do
    if [[ -d "${candidate}" ]]; then
      WAN_CKPT_DIR="${candidate}"
      break
    fi
  done
fi
if [[ "${WAN_CKPT_DIR}" == *14B* && "${ALLOW_WAN_14B:-0}" != "1" ]]; then
  echo "Refusing 14B checkpoint '${WAN_CKPT_DIR}' for this stage:" \
       "full-QKV 14B OOMs the 910B reducer (12.6GiB buckets). Point" \
       "WAN_CKPT_DIR at Wan2.1-T2V-1.3B, or set ALLOW_WAN_14B=1 plus" \
       "TRAIN_QKV_LAST_N<=4 if you really mean 14B." >&2
  exit 1
fi

# --- dataset override: identical to 07 (the sparse -oft 10k mirror) ---------
SPATIALVID_OFT_ROOT="obs://yw-ads-training-gy1/data/external/personal/g00833899/y50046448/spatial-vid-hq-oft"
SPATIALVID_METADATA_URL="${SPATIALVID_OFT_ROOT}/data/train/SpatialVID_HQ_metadata.csv"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_OFT_ROOT}/videos/SpatialVid/HQ/videos"
SPATIALVID_METADATA="${LOCAL_CACHE_ROOT}/metadata/SpatialVID_HQ_metadata_oft.csv"
SPATIALVID_SPLIT_DIR="${RUN_ROOT}/metadata/spatialvid_oft_seed${SPLIT_SEED}"
SPATIALVID_TRAIN_10K_CSV="${SPATIALVID_SPLIT_DIR}/train_10k.csv"
SPATIALVID_EVAL_CSV="${SPATIALVID_SPLIT_DIR}/eval.csv"
SPATIALVID_OVERFIT_CSV="${SPATIALVID_SPLIT_DIR}/overfit.csv"

# ----------------------------- editable settings -----------------------------
TARGET_MODE="${TARGET_MODE:-absolute}"   # the E7-validated contract
# Run tag separates incompatible trainable-param sets (v1 = frozen-FFN,
# no pseudo-context; v2 = pseudo-context + two-speed LR + adapter warmup).
WAN_RUN_TAG="${WAN_RUN_TAG:-v2}"
OUTPUT_DIR="${SCALE_ROOT}/dual_diffusion_wan_${TARGET_MODE}_${WAN_RUN_TAG}"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/dual_diffusion_wan_${TARGET_MODE}_${WAN_RUN_TAG}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"

DUAL_AE_CKPT="${DUAL_AE_CKPT:-${VGGAE_REF_ROOT}/checkpoints/e5_dual_stream_r5.pt}"
DUAL_AE_CKPT_URL="${DUAL_AE_CKPT_URL:-}"

MAX_STEPS="${MAX_STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"          # 48 x 1 x 2 = global 96, matches E7-a
ADAPTER_LR="${ADAPTER_LR:-1e-4}"         # fresh translation layers
WAN_LR="${WAN_LR:-1e-5}"                 # pretrained prior
WAN_FREEZE_STEPS="${WAN_FREEZE_STEPS:-500}"
PSEUDO_TEXT="${PSEUDO_TEXT:-8}"          # learned null prompt tokens
WARMUP_STEPS="${WARMUP_STEPS:-300}"
TRAIN_QKV_LAST_N="${TRAIN_QKV_LAST_N:-0}"  # 0 = all blocks
# FFN unfreeze is the FOLLOW-UP arm, not the default: test the principled
# fixes (pseudo-context / two-speed LR / warmup) in isolation first.
TRAIN_FFN_LAST_N="${TRAIN_FFN_LAST_N:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-50}"
SAMPLE_STEPS="${SAMPLE_STEPS:-30}"
EVAL_CLIPS="${EVAL_CLIPS:-16}"
MASTER_PORT=29650
# -----------------------------------------------------------------------------

configure_modelarts_distributed
require_scale_cluster
require_output_url
ensure_spatialvid_subset_splits

if [[ -z "${WAN_CKPT_DIR}" || ! -d "${WAN_CKPT_DIR}" ]]; then
  echo "Wan2.1-T2V-1.3B checkpoint not found (searched the 1.3B candidates" \
       "under VGGAE_REF_ROOT=${VGGAE_REF_ROOT} and the repo). Export" \
       "WAN_CKPT_DIR=<path-to-Wan2.1-T2V-1.3B> and relaunch." >&2
  exit 1
fi

if [[ ! -s "${DUAL_AE_CKPT}" && -n "${DUAL_AE_CKPT_URL}" ]]; then
  mkdir -p "$(dirname "${DUAL_AE_CKPT}")"
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${DUAL_AE_CKPT_URL}" "${DUAL_AE_CKPT}"
fi
require_file "${DUAL_AE_CKPT}" "R5 dual-stream AE checkpoint"

echo "E8 launch: wan=${WAN_CKPT_DIR} arm=${TARGET_MODE}" \
     "NNODES=${NNODES} WORLD_SIZE=${WORLD_SIZE}"

EXTRA_ARGS=()
if [[ "${AUTO_RESUME}" -eq 1 || -n "${RESUME}" ]]; then
  RESUME=$(resolve_resume_checkpoint \
    "${RESUME}" \
    "${OUTPUT_DIR}/checkpoint_latest.pt" \
    "${REMOTE_OUTPUT_DIR}/checkpoint_latest.pt" \
    "${LOCAL_CACHE_ROOT}/resume/dual_diffusion_wan_${TARGET_MODE}_${WAN_RUN_TAG}.pt" \
    "${SCALE_MIRROR_ROOT}/dual_diffusion_wan_${TARGET_MODE}_${WAN_RUN_TAG}/checkpoint_latest.pt")
fi
if [[ -n "${RESUME}" ]]; then
  echo "Resuming from ${RESUME}"
  EXTRA_ARGS+=(--resume "${RESUME}")
  # Container restarts begin with an empty local metrics.jsonl which the
  # directory sync would then push over the remote history (observed on the
  # first E8 run). Restore the remote copy first so appends continue it.
  if [[ ! -s "${OUTPUT_DIR}/metrics.jsonl" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${REMOTE_OUTPUT_DIR}/metrics.jsonl" \
      "${OUTPUT_DIR}/metrics.jsonl" 2>/dev/null \
    || "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
      "${SCALE_MIRROR_ROOT}/dual_diffusion_wan_${TARGET_MODE}_${WAN_RUN_TAG}/metrics.jsonl" \
      "${OUTPUT_DIR}/metrics.jsonl" 2>/dev/null || true
  fi
fi

start_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"
trap 'stop_output_sync "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}"' EXIT

run_torchrun "${PROJECT}/train_dual_diffusion.py" \
  --generator wan \
  --wan_ckpt_dir "${WAN_CKPT_DIR}" \
  --train_qkv_last_n "${TRAIN_QKV_LAST_N}" \
  --train_ffn_last_n "${TRAIN_FFN_LAST_N}" \
  --target_mode "${TARGET_MODE}" \
  --csv "${SPATIALVID_TRAIN_10K_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --eval_csv "${SPATIALVID_EVAL_CSV}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --dual_ae_ckpt "${DUAL_AE_CKPT}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --accum_steps "${ACCUM_STEPS}" \
  --adapter_lr "${ADAPTER_LR}" \
  --wan_lr "${WAN_LR}" \
  --wan_freeze_steps "${WAN_FREEZE_STEPS}" \
  --pseudo_text_tokens "${PSEUDO_TEXT}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_clips "${EVAL_CLIPS}" \
  --sample_steps "${SAMPLE_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --seq_len 8 --target_size 518 --clip_duration_seconds 1.0 \
  --dtype bf16 \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
