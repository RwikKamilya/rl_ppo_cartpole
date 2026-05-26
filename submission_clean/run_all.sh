#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

if [[ -x ../venv/bin/python ]]; then
  PY=../venv/bin/python
else
  PY=python
fi

"$PY" scripts/fix_results_layout.py --study-root results/final_study

"$PY" scripts/run_final_experiments.py \
  --stage all \
  --seeds 0 1 2 3 4 \
  --ablation-seeds 0 1 2 \
  --total-env-steps 1000000 \
  --ablation-total-env-steps 500000 \
  --results-root results \
  --study-name final_study \
  --skip-existing \
  --python-bin "$PY"

"$PY" scripts/check_results_complete.py --study-root results/final_study

"$PY" scripts/compute_metrics.py \
  --study-root results/final_study \
  --output-dir results/final_study/metrics

"$PY" scripts/plot_results.py \
  --study-root results/final_study \
  --output-dir results/final_study/plots
