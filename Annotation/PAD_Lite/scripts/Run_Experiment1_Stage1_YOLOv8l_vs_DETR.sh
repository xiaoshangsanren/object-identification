#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
annotation_root="$(cd "${script_dir}/../.." && pwd)"
gpu_index="${1:-4}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

free_mb="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || [[ "${free_mb}" -lt 10000 ]]; then
  echo "GPU ${gpu_index} does not have the required 10000 MiB free memory (current: ${free_mb:-unknown})." >&2
  exit 2
fi

cd "${annotation_root}"
CUDA_VISIBLE_DEVICES="${gpu_index}" conda run -n cc_PAD_Lite \
  python -m PAD_Lite.detector_zero_shot_eval \
  --config PAD_Lite/configs/detector_zero_shot_yolov8l_detr.json \
  --mode full \
  --device cuda:0 \
  "$@"
