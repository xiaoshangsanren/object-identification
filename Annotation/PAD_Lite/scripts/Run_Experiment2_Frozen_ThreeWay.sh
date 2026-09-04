#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pad_root="$(cd "${script_dir}/.." && pwd)"
annotation_root="$(cd "${pad_root}/.." && pwd)"
dino_gpu="${1:-4}"
convnext_gpu="${2:-6}"
run_id="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${pad_root}/outputs/04_gallery_and_recognizer_evaluation/frozen_dino_convnext_p2b_comparison/${run_id}"
config="${pad_root}/configs/frozen_dino_convnext_p2b_comparison.json"

for gpu in "${dino_gpu}" "${convnext_gpu}"; do
  free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || [[ "${free_mb}" -lt 6000 ]]; then
    echo "GPU ${gpu} requires at least 6000 MiB free memory (current: ${free_mb:-unknown})." >&2
    exit 2
  fi
done
if [[ "${dino_gpu}" == "${convnext_gpu}" ]]; then
  echo "DINO and ConvNeXt must use different GPUs for the parallel run." >&2
  exit 2
fi

mkdir -p "${run_root}"
cd "${annotation_root}"
CUDA_VISIBLE_DEVICES="${dino_gpu}" conda run -n cc_PAD_Lite \
  python -m PAD_Lite.frozen_backbone_comparison run \
  --config "${config}" --backbone dino_raw --folds all --device cuda:0 \
  --run-root "${run_root}" >"${run_root}/dino_raw.log" 2>&1 &
dino_pid=$!
CUDA_VISIBLE_DEVICES="${convnext_gpu}" conda run -n cc_PAD_Lite \
  python -m PAD_Lite.frozen_backbone_comparison run \
  --config "${config}" --backbone convnext_raw --folds all --device cuda:0 \
  --run-root "${run_root}" >"${run_root}/convnext_raw.log" 2>&1 &
convnext_pid=$!

status=0
wait "${dino_pid}" || status=$?
wait "${convnext_pid}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "Frozen backbone comparison failed; inspect logs in ${run_root}." >&2
  exit "${status}"
fi
conda run -n cc_PAD_Lite python -m PAD_Lite.frozen_backbone_comparison summarize \
  --config "${config}" --run-root "${run_root}"
echo "Experiment 2 completed: ${run_root}"
