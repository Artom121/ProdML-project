#!/usr/bin/env bash
# параметры из config.yaml можно переопределить: ./run.sh model=logreg depth=6
set -euo pipefail
cd "$(dirname "$0")"

export MLFLOW_DISABLE_AGENT_HINT=1
uv sync --frozen
uv run --frozen python prepare_data.py
uv run --frozen python train.py "$@"
