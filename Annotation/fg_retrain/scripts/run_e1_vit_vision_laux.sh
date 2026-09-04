#!/usr/bin/env bash
set -euo pipefail

STAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${STAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

conda run --no-capture-output \
  -n cc_object_identification \
  python -m fg_retrain.train_laux \
  --config "${STAGE_ROOT}/configs/e1_vit_vision_laux.json" \
  "$@"

