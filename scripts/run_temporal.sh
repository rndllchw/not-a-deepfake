#!/usr/bin/env bash
# Runs pooling, the LSTM and the Transformer on one frame run, then evaluates them,
# chooses the decision rule and prints the comparison table.
#
# 1. Export the features once:  bash scripts/export_features.sh <frame run folder>
# 2. From the repo root, in tmux:
#        bash scripts/run_temporal.sh ~/data/runs/frames/<mask or no-mask>/<run name>
#    Extra options go to the two trainings, e.g. a quick test: --epochs 1 --samples-per-epoch 256
#
# Safe to rerun: finished trainings are skipped and unfinished ones resume.
set -euo pipefail

RUN="${1:?usage: bash scripts/run_temporal.sh <frame run folder> [train options]}"
shift
RUN="${RUN/#\~/$HOME}"
# frame run = <data root>/runs/frames/<setting>/<run name>
DATA_ROOT="$(cd "$RUN/../../../.." && pwd)"
TEMPORAL="$DATA_ROOT/runs/temporal/$(basename "$(dirname "$RUN")")/$(basename "$RUN")"

echo "=== pooling ==="
python -m src.temporal.pooling --run "$RUN" --how mean
python -m src.temporal.pooling --run "$RUN" --how topk --k 5

for model in lstm transformer; do
    echo "=== $model ==="
    python -m src.temporal.train --frame-run "$RUN" --model "$model" --resume "$@"
done

for aggregator in mean topk-5 lstm transformer; do
    echo "=== evaluate and decide: $aggregator ==="
    python -m src.temporal.evaluate --dir "$TEMPORAL/$aggregator"
    python -m src.temporal.decision --dir "$TEMPORAL/$aggregator"
done

python -m src.temporal.compare --data-root "$DATA_ROOT"
