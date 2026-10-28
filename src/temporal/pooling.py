"""
Pooling baseline (Step 4): a window's score is the mean, or the mean of the k highest, of its frame probabilities.
    python -m src.temporal.pooling --run ~/data/runs/frames/<mask or no-mask>/<run name> [--how topk --k 5]

No features needed. Reads {run}/predictions/{val,val_call,test,test_call}.csv (from src.frames.predict)
and writes window-score tables (src.temporal.windows) under runs/temporal/.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.temporal.windows import (COLUMNS, DEFAULT_MIN_FRAMES, split_name, temporal_dir, windows_path,
                                  write_windows)

KEYS = ["video_id", "window"]


def aggregator_name(how: str, k: int) -> str:
    return "mean" if how == "mean" else f"topk-{k}"


def pool_windows(frames: pd.DataFrame, how: str = "mean", k: int = 5,
                 min_frames: int = DEFAULT_MIN_FRAMES) -> pd.DataFrame:
    """Window-score table from per-frame rows (video_id, window, dataset, method, label, prob).
    Windows with fewer than min_frames frames are left out, as for the temporal models."""
    frames = frames[frames.groupby(KEYS)["prob"].transform("size") >= min_frames]
    if how == "mean":
        used = frames
    elif how == "topk":
        # keep the k highest probabilities of each window; a window with fewer keeps all of them
        ranked = frames.sort_values("prob", ascending=False)
        used = ranked[ranked.groupby(KEYS).cumcount() < k]
    else:
        raise ValueError("how must be 'mean' or 'topk'")
    table = used.groupby(KEYS).agg(dataset=("dataset", "first"), method=("method", "first"),
                                   label=("label", "first"), score=("prob", "mean")).reset_index()
    return table[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run", type=Path, required=True, help="frame run folder written by src.frames.train")
    parser.add_argument("--how", default="mean", choices=["mean", "topk"])
    parser.add_argument("--k", type=int, default=5, help="frames averaged per window with --how topk")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES,
                        help="windows with fewer frames are left out (same default as the temporal models)")
    args = parser.parse_args()

    run = args.run.expanduser()
    agg_dir = temporal_dir(run, aggregator_name(args.how, args.k))
    found = False
    for split in ("val", "test"):
        for call_quality in (False, True):
            name = split_name(split, call_quality)
            source = run / "predictions" / f"{name}.csv"
            if not source.exists():
                continue
            found = True
            frames = pd.read_csv(source, dtype={"video_id": str})
            table = pool_windows(frames, args.how, args.k, args.min_frames)
            write_windows(table, windows_path(agg_dir, name))
            print(f"{name}: {len(frames)} frames -> {len(table)} windows of {table['video_id'].nunique()} videos")
    if not found:
        raise SystemExit(f"no scored splits in {run / 'predictions'}; run src.frames.predict first")
    print(f"wrote {agg_dir / 'predictions'}")


if __name__ == "__main__":
    main()
