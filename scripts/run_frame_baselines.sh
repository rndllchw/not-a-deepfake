#!/usr/bin/env bash
# Trains and tests the four baseline frame models, each without and with the
# face mask (8 runs), then prints the comparison table.
#
# Run on the GPU machine from the repo root, inside tmux:
#     bash scripts/run_frame_baselines.sh
# Extra options are passed to every run, e.g. a quick test:
#     bash scripts/run_frame_baselines.sh --data-root ~/test-data --epochs 1 --samples-per-epoch 64
#
# Safe to rerun after the machine stops: finished runs are skipped, and an
# unfinished one continues from its last epoch (--resume).
set -euo pipefail

BASELINES=(
    "xception full"
    "efficientnet_b4 full"
    "vit_b16 full"
    "clip_vit_b16 norm"
)

for mask in "" "--mask"; do  # "" = no-mask setting, --mask = mask setting
    for baseline in "${BASELINES[@]}"; do
        read -r backbone mode <<< "$baseline"
        echo "=== $backbone ($mode) ${mask:-no-mask} ==="
        python -m src.frames.train --backbone "$backbone" --train-mode "$mode" $mask --resume "$@"
    done
done

# compare.py only needs --data-root; pick it out of the extra options if given
data_root=()
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--data-root" ]]; then data_root=(--data-root "${args[$((i + 1))]}"); fi
done
python -m src.frames.compare "${data_root[@]}"
