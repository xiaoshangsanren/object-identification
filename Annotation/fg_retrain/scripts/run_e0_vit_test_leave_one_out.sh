#!/usr/bin/env bash
set -euo pipefail

STAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${STAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

conda run --no-capture-output \
  -n cc_object_identification \
  python -m fg_retrain.evaluate_vit_leave_one_out \
  --config "${STAGE_ROOT}/configs/e0_vit_test_leave_one_out.json" \
  "$@"

