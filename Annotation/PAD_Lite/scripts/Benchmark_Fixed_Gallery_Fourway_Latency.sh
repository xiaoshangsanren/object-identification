#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pad_root="$(cd "${script_dir}/.." && pwd)"
annotation_root="$(cd "${pad_root}/.." && pwd)"
gpu_index="${1:-4}"
run_root="${2:-${pad_root}/outputs/04_gallery_and_recognizer_evaluation/fixed_gallery_fourway_v1_00/fixed_gallery_v1_00_20260828}"
config="${pad_root}/configs/fixed_gallery_fourway_v1_00.json"
python_bin="/home/NCUT/25/cc/.conda/envs/cc_PAD_Lite/bin/python"

if [[ "${run_root}" != /* ]]; then
  run_root="$(realpath -m "${run_root}")"
fi

free_mb="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || [[ "${free_mb}" -lt 10000 ]]; then
  echo "GPU ${gpu_index} requires at least 10000 MiB free (current: ${free_mb:-unknown})." >&2
  exit 2
fi

cd "${annotation_root}"
CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" -m PAD_Lite.src.fixed_gallery_latency \
  --config "${config}" \
  --run-root "${run_root}" \
  --device cuda:0 \
  --batch-sizes 1 32 \
  --repeats 3 \
  --warmup-batches 10

echo "Latency benchmark completed: ${run_root}/LATENCY_RESULTS.md"
