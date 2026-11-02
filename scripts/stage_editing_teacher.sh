#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/spatialvid_config.sh"
source "${SCRIPT_DIR}/lib/modelarts.sh"
ensure_local_checkpoint "${STREAMVGGT_CKPT}" "${STREAMVGGT_URL:-}" \
  'existing StreamVGGT' "${STREAMVGGT_MIRROR_URL:-}"
ensure_local_checkpoint "${EDIT_RAE}" \
  "${EDIT_RAE_URL:-${SCALE_REMOTE_ROOT}/scene_rae_c256_15day_v1/ae/best_joint.pt}" \
  'production Scene C256 AE' \
  "${EDIT_RAE_MIRROR_URL:-${SCALE_MIRROR_ROOT}/scene_rae_c256_15day_v1/ae/best_joint.pt}"
