"""
Scores windows with a trained temporal model (Step 4). train does this when it finishes; use this to rescore.
    python -m src.temporal.predict --dir ~/data/runs/temporal/<mask or no-mask>/<run name>/<aggregator>

Scores val and test, clean and call quality, wherever features were exported. Writes
predictions/<name>_windows.csv and timing.json (ms per window, temporal model only).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.temporal.models import build_model
from src.temporal.sequences import WindowSequences, feature_files_exist
from src.temporal.windows import COLUMNS, frame_run_of, split_name, windows_path, write_windows


def load_model(checkpoint: Path, device: str) -> tuple[torch.nn.Module, dict]:
    """The trained temporal model and the args it was trained with."""
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(ckpt["name"], ckpt["d_in"], **ckpt["options"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["args"]


@torch.no_grad()
def score_windows(model: torch.nn.Module, dataset: WindowSequences, device: str, batch_size: int = 256,
                  workers: int = 0) -> np.ndarray:
    """Fake probability of each window in dataset.meta, in the same order."""
    model.eval()
    scores = np.zeros(len(dataset), np.float32)
    for x, mask, _, idx in DataLoader(dataset, batch_size, num_workers=workers, pin_memory=device == "cuda"):
        logits = model(x.to(device, non_blocking=True), mask.to(device, non_blocking=True))
        scores[idx.numpy()] = torch.sigmoid(logits).cpu().numpy()
    return scores


def window_table(dataset: WindowSequences, scores: np.ndarray) -> pd.DataFrame:
    """The window-score table for a scored dataset."""
    return dataset.meta.assign(score=scores)[COLUMNS]


@torch.no_grad()
def window_ms(model: torch.nn.Module, d_in: int, device: str, max_len: int = 32, repeats: int = 100) -> float:
    """Time for the temporal model to score one full window (batch size 1, as in a live call)."""
    x = torch.zeros(1, max_len, d_in, device=device)
    mask = torch.ones(1, max_len, dtype=torch.bool, device=device)
    for i in range(repeats + 10):
        if i == 10:  # the first runs are slow while the model warms up
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        model(x, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def score_all(agg_dir: Path, splits=("val", "test"), checkpoint: str = "best.pt", batch_size: int = 256,
              workers: int = 0) -> list[str]:
    """Scores every available split and version, writes the tables and timing.json, returns the names written."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame_run = frame_run_of(agg_dir)
    model, train_args = load_model(agg_dir / checkpoint, device)
    written = []
    for split in splits:
        for call_quality in (False, True):
            version = "call" if call_quality else "clean"
            name = split_name(split, call_quality)
            if not feature_files_exist(frame_run, split, version):
                print(f"{name}: no exported features in {frame_run / 'predictions'}, skipped")
                continue
            dataset = WindowSequences(frame_run, split, (version,), train_args["min_frames"])
            table = window_table(dataset, score_windows(model, dataset, device, batch_size, workers))
            write_windows(table, windows_path(agg_dir, name))
            print(f"{name}: {len(table)} windows of {table['video_id'].nunique()} videos")
            written.append(name)
    timing = dict(ms_head_per_window=window_ms(model, train_args["d_in"], device))
    (agg_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"{timing['ms_head_per_window']:.2f} ms per window (temporal model only)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dir", type=Path, required=True, help="aggregator folder written by src.temporal.train")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best.pt", help="file in the aggregator folder")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if not score_all(args.dir.expanduser(), args.splits, args.checkpoint, args.batch_size, args.workers):
        raise SystemExit("nothing was scored; export the frame model's features first (src.frames.predict --features)")


if __name__ == "__main__":
    main()
