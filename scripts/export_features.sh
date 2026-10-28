#!/usr/bin/env bash
# Exports the frame model's features for every split, clean and call quality (6 exports),
# for the temporal models. Do this once, for the frame run you chose.
#
# On the GPU machine, from the repo root, in tmux (train is the slow one):
#     bash scripts/export_features.sh ~/data/runs/frames/<mask or no-mask>/<run name>
# Options after the run folder go to every export, e.g. --workers 8 --batch-size 128.
#
# Safe to rerun: finished exports are skipped. To redo one, delete its <name>_features.npy.
# This rewrites test.csv and test_call.csv with the same scores as before.
set -euo pipefail

RUN="${1:?usage: bash scripts/export_features.sh <frame run folder> [predict options]}"
shift
RUN="${RUN/#\~/$HOME}"

# val and test first, train (the largest) last
for split in val test train; do
    for call in "" "--call-quality"; do
        name="$split${call:+_call}"
        features="$RUN/predictions/${name}_features.npy"
        scores="$RUN/predictions/${name}.csv"
        # predict writes the csv after the features, so a csv newer than the .npy means it finished
        if [[ -f "$features" && -f "$scores" && "$scores" -nt "$features" ]]; then
            echo "=== $name: already exported ==="
            continue
        fi
        echo "=== $name ==="
        python -m src.frames.predict --run "$RUN" --split "$split" --features $call "$@"
    done
done
echo "features exported to $RUN/predictions"
