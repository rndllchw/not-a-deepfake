"""
The window-score table: the one format every aggregator writes and evaluate and decision read.
One row per (video_id, window): video_id, window, dataset, method, label, score (fake probability, 0-1).

Temporal outputs sit next to the frame run they are built on:
    {data_root}/runs/temporal/<mask or no-mask>/<run name>/<aggregator>/predictions/<name>_windows.csv
where <name> is the split (val, test), plus _call for call quality.
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["video_id", "window", "dataset", "method", "label", "score"]
# A window needs this many usable face crops (of 32) to be scored. The quality filters in
# src.frames can drop frames, and a window with only a few left says little about the video.
DEFAULT_MIN_FRAMES = 8


def split_name(split: str, call_quality: bool) -> str:
    """File name stem used by src.frames.predict: 'val', 'test_call', ..."""
    return split + ("_call" if call_quality else "")


def temporal_dir(frame_run: Path, aggregator: str) -> Path:
    """Output folder of an aggregator built on a frame run."""
    # frame_run = runs/frames/<mask setting>/<run>
    return frame_run.parents[2] / "temporal" / frame_run.parent.name / frame_run.name / aggregator


def data_root_of(agg_dir: Path) -> Path:
    """The data root an aggregator folder sits in."""
    # agg_dir = {data_root}/runs/temporal/<mask setting>/<run>/<aggregator>
    return agg_dir.parents[4]


def frame_run_of(agg_dir: Path) -> Path:
    """The frame run an aggregator folder was built on (the inverse of temporal_dir)."""
    # agg_dir = runs/temporal/<mask setting>/<run>/<aggregator>
    return agg_dir.parents[3] / "frames" / agg_dir.parents[1].name / agg_dir.parents[0].name


def windows_path(agg_dir: Path, name: str) -> Path:
    return agg_dir / "predictions" / f"{name}_windows.csv"


def write_windows(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table[COLUMNS].sort_values(["video_id", "window"]).to_csv(path, index=False)


def read_windows(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"video_id": str})
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path} is not a window-score table, missing {missing}")
    return table.sort_values(["video_id", "window"]).reset_index(drop=True)


def video_table(windows: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """One row per video: its label, dataset, method, and `score` = the mean (or
    max) of its window scores."""
    if how not in ("mean", "max"):
        raise ValueError("how must be 'mean' or 'max'")
    return windows.groupby("video_id").agg(label=("label", "first"), dataset=("dataset", "first"),
                                           method=("method", "first"), score=("score", how))


def available_names(agg_dir: Path) -> list[str]:
    """Names (val, val_call, test, test_call) that have a window-score table, in that order."""
    names = [split_name(s, c) for s in ("val", "test") for c in (False, True)]
    return [n for n in names if windows_path(agg_dir, n).exists()]
