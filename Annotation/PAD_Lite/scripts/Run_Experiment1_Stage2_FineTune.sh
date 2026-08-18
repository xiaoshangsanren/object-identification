#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
annotation_root="$(cd "${script_dir}/../.." && pwd)"
yolo_gpu="${1:-4}"
detr_gpu="${2:-6}"
run_id="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${annotation_root}/PAD_Lite/outputs/detector_finetune/${run_id}"

if [[ "${yolo_gpu}" == "${detr_gpu}" ]]; then
  echo "Stage 2 runs YOLO and DETR concurrently; choose two different GPU indices." >&2
  exit 2
fi
for gpu_index in "${yolo_gpu}" "${detr_gpu}"; do
  free_mb="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || [[ "${free_mb}" -lt 18000 ]]; then
    echo "GPU ${gpu_index} does not have the required 18000 MiB free memory (current: ${free_mb:-unknown})." >&2
    exit 2
  fi
done

mkdir -p "${run_root}"
cd "${annotation_root}"

echo "Stage 2 Run ID: ${run_id}"
echo "YOLO GPU: ${yolo_gpu}; DETR GPU: ${detr_gpu}"
echo "Output: ${run_root}"

conda run -n cc_PAD_Lite python -m PAD_Lite.detector_finetune_eval prepare \
  --config PAD_Lite/configs/detector_finetune_yolo_detr.json

set +e
(
  set -o pipefail
  CUDA_VISIBLE_DEVICES="${yolo_gpu}" conda run -n cc_PAD_Lite \
    python -m PAD_Lite.detector_finetune_eval run \
    --config PAD_Lite/configs/detector_finetune_yolo_detr.json \
    --detector yolo \
    --fold all \
    --device cuda:0 \
    --run-root "${run_root}" \
    --resume 2>&1 | tee "${run_root}/yolo.log"
) &
yolo_pid=$!
(
  set -o pipefail
  CUDA_VISIBLE_DEVICES="${detr_gpu}" conda run -n cc_PAD_Lite \
    python -m PAD_Lite.detector_finetune_eval run \
    --config PAD_Lite/configs/detector_finetune_yolo_detr.json \
    --detector detr \
    --fold all \
    --device cuda:0 \
    --run-root "${run_root}" \
    --resume 2>&1 | tee "${run_root}/detr.log"
) &
detr_pid=$!

status=0
wait "${yolo_pid}" || status=1
wait "${detr_pid}" || status=1
set -e
if [[ "${status}" -ne 0 ]]; then
  echo "At least one detector run failed. Reuse run ID ${run_id} after fixing the issue." >&2
  exit "${status}"
fi

conda run -n cc_PAD_Lite python -m PAD_Lite.detector_finetune_eval summarize \
  --config PAD_Lite/configs/detector_finetune_yolo_detr.json \
  --run-root "${run_root}"

echo "Experiment 1 stage 2 completed: ${run_root}"
