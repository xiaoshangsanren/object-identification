#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
annotation_root="$(cd "${script_dir}/../.." && pwd)"
gpu_index="${1:-4}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

cd "${annotation_root}"
CUDA_VISIBLE_DEVICES="${gpu_index}" conda run -n cc_PAD_Lite \
  python -m PAD_Lite.detector_zero_shot_eval \
  --config PAD_Lite/configs/detector_zero_shot_yolo_detr.json \
  --mode full \
  --device cuda:0 \
  "$@"
