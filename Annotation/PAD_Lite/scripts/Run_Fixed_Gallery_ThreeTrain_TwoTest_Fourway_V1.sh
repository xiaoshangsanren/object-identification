#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pad_root="$(cd "${script_dir}/.." && pwd)"
annotation_root="$(cd "${pad_root}/.." && pwd)"
dino_gpu="${1:-4}"
convnext_gpu="${2:-6}"
run_id="${3:-$(date +%Y%m%dT%H%M%S)}"
run_root="${pad_root}/outputs/04_gallery_and_recognizer_evaluation/fixed_gallery_three_train_two_test_fourway_v1_00/${run_id}"
config="${pad_root}/configs/fixed_gallery_three_train_two_test_fourway_v1_00.json"
python_bin="/home/NCUT/25/cc/.conda/envs/cc_PAD_Lite/bin/python"

if [[ "${dino_gpu}" == "${convnext_gpu}" ]]; then
  echo "DINO and ConvNeXt chains require different GPUs." >&2
  exit 2
fi
for gpu in "${dino_gpu}" "${convnext_gpu}"; do
  free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || [[ "${free_mb}" -lt 10000 ]]; then
    echo "GPU ${gpu} requires at least 10000 MiB free (current: ${free_mb:-unknown})." >&2
    exit 2
  fi
done

mkdir -p "${run_root}"
cd "${annotation_root}"
"${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway validate \
  --config "${config}" --run-root "${run_root}"

(
  CUDA_VISIBLE_DEVICES="${dino_gpu}" "${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway raw \
    --config "${config}" --run-root "${run_root}" --backbone dino_raw --device cuda:0
  CUDA_VISIBLE_DEVICES="${dino_gpu}" "${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway dino-p0 \
    --config "${config}" --run-root "${run_root}" --device cuda:0
  CUDA_VISIBLE_DEVICES="${dino_gpu}" "${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway dino-p2a \
    --config "${config}" --run-root "${run_root}" --device cuda:0
  CUDA_VISIBLE_DEVICES="${dino_gpu}" "${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway dino-p2b \
    --config "${config}" --run-root "${run_root}" --device cuda:0
) >"${run_root}/dino_chain.log" 2>&1 &
dino_pid=$!

(
  CUDA_VISIBLE_DEVICES="${convnext_gpu}" "${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway raw \
    --config "${config}" --run-root "${run_root}" --backbone convnext_raw --device cuda:0
  CUDA_VISIBLE_DEVICES="${convnext_gpu}" "${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway convnext-p2b \
    --config "${config}" --run-root "${run_root}" --device cuda:0
) >"${run_root}/convnext_chain.log" 2>&1 &
convnext_pid=$!

status=0
wait "${dino_pid}" || status=$?
wait "${convnext_pid}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "3-train/2-test four-way experiment failed; inspect logs in ${run_root}." >&2
  exit "${status}"
fi

"${python_bin}" -m PAD_Lite.src.fixed_gallery_three_two_fourway summarize \
  --config "${config}" --run-root "${run_root}"
echo "Experiment completed: ${run_root}"
