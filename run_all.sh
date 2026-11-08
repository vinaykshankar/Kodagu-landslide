#!/usr/bin/env bash
# Reproduce every result: bash run_all.sh   (data/ must contain the inputs listed in README.md)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs figures
python3 scripts/00_prepare_dem.py data
python3 scripts/01_factors.py data
python3 scripts/02_samples.py data outputs
python3 scripts/03_models.py outputs
python3 scripts/04_map_shap.py data outputs
python3 scripts/06_figures.py data outputs figures
python3 scripts/07_sensitivity_overlay.py data outputs
echo "All outputs written to outputs/ and figures/"
