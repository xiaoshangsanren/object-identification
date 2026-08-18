#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu_index="${1:-4}"
free_mb="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_mb}" =~ ^[0-9]+$ ]] || [[ "${free_mb}" -lt 8000 ]]; then
  echo "GPU ${gpu_index} does not have the required 8000 MiB free memory (current: ${free_mb:-unknown})." >&2
  exit 2
fi
exec bash "${script_dir}/Run_Detector_ZeroShot_Full.sh" "$@"
