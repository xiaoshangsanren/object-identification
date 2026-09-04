#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANNOTATION_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${ANNOTATION_ROOT}/.." && pwd)"

POINTCLOUD_GPU="${POINTCLOUD_GPU:-6}"
POINTCLOUD_BATCH_SIZE="${POINTCLOUD_BATCH_SIZE:-16}"
POINTCLOUD_COUNT="${POINTCLOUD_COUNT:-4096}"
POINTCLOUD_TARGET="${POINTCLOUD_TARGET:-/mnt/sata_ssd/cc-2025/annotation_datasets/processed/russian_pad_pointcloud_v1}"
POINTCLOUD_LINK="${PROJECT_ROOT}/datasets/processed/russian_pad_pointcloud_v1"

if [[ -e "${POINTCLOUD_TARGET}" ]]; then
  echo "Refusing to overwrite existing output: ${POINTCLOUD_TARGET}" >&2
  exit 2
fi

cd "${ANNOTATION_ROOT}"
CUDA_VISIBLE_DEVICES="${POINTCLOUD_GPU}" conda run -n cc_PAD_Lite --no-capture-output \
  python -m PAD_Lite.src.build_monocular_pointcloud_dataset \
  --output "${POINTCLOUD_TARGET}" \
  --device cuda:0 \
  --batch-size "${POINTCLOUD_BATCH_SIZE}" \
  --point-count "${POINTCLOUD_COUNT}"

if [[ -L "${POINTCLOUD_LINK}" ]]; then
  if [[ "$(readlink -f "${POINTCLOUD_LINK}")" != "$(readlink -f "${POINTCLOUD_TARGET}")" ]]; then
    echo "Existing project link points elsewhere: ${POINTCLOUD_LINK}" >&2
    exit 3
  fi
elif [[ -e "${POINTCLOUD_LINK}" ]]; then
  echo "Project path exists and is not a symlink: ${POINTCLOUD_LINK}" >&2
  exit 4
else
  ln -s "${POINTCLOUD_TARGET}" "${POINTCLOUD_LINK}"
fi

echo "Point-cloud dataset ready: ${POINTCLOUD_LINK}"
