"""
Feature sequences for the temporal models (Step 4): one window of frame features per item.

Reads what src.frames.predict --features saved (scripts/export_features.sh): {run}/predictions/<name>.csv and
<name>_features.npy, same row order. Rows are grouped by (video_id, window), sorted by frame, padded to 32:
    x (32, D) features, mask (32,) True for real frames, label (1 = fake), i (row in .meta)
Windows with fewer than min_frames frames (the frame filters can drop some) are left out.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.temporal.windows import DEFAULT_MIN_FRAMES

KEYS = ["video_id", "window"]


def version_name(split: str, version: str) -> str:
    """'clean' -> 'val', 'call' -> 'val_call'."""
    if version not in ("clean", "call"):
        raise ValueError("version must be 'clean' or 'call'")
    return split + ("_call" if version == "call" else "")


def feature_files_exist(frame_run: Path, split: str, version: str) -> bool:
    name = version_name(split, version)
    pred = frame_run / "predictions"
    return (pred / f"{name}.csv").exists() and (pred / f"{name}_features.npy").exists()


def window_index(frames: pd.DataFrame, min_frames: int) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per window, and the csv row numbers of all windows' frames laid out one
    window after another in time order. meta["start"] and meta["n_frames"] say which
    part of that array belongs to a window."""
    if not frames.index.equals(pd.RangeIndex(len(frames))):
        raise ValueError("frames must have the default index: row i is feature row i")
    ordered = frames.sort_values(KEYS + ["frame"], kind="stable")
    meta = ordered.groupby(KEYS, sort=False).agg(dataset=("dataset", "first"), method=("method", "first"),
                                                 label=("label", "first"), n_frames=("frame", "size")).reset_index()
    meta["start"] = meta["n_frames"].cumsum() - meta["n_frames"]
    return meta[meta["n_frames"] >= min_frames].reset_index(drop=True), ordered.index.to_numpy()


def sampling_weights(meta: pd.DataFrame, real_share: float = 0.5) -> np.ndarray:
    """Weight per window so real windows make up `real_share` of draws, and within real
    and fake, every video is drawn equally often (as src.frames.train does for crops)."""
    windows_in_video = meta.groupby("video_id")["video_id"].transform("size")
    videos_in_class = meta.groupby("label")["video_id"].transform("nunique")
    class_share = np.where(meta["label"] == 0, real_share, 1 - real_share)
    return class_share / (windows_in_video * videos_in_class).to_numpy()


class WindowSequences(Dataset):
    """One window of frame features per item: (x, mask, label, i).

    versions:  which exports to read, "clean" and/or "call". With both, each item
               picks one at random: the whole window is clean or the whole window
               is call quality, as a real call degrades every frame. Evaluation
               uses one version per dataset.
    """

    def __init__(self, frame_run: Path, split: str, versions=("clean",), min_frames: int = DEFAULT_MIN_FRAMES,
                 max_len: int = 32):
        self.max_len = max_len
        self.paths = {}
        reference = None
        for v in versions:
            name = version_name(split, v)
            csv, npy = frame_run / "predictions" / f"{name}.csv", frame_run / "predictions" / f"{name}_features.npy"
            for p in (csv, npy):
                if not p.exists():
                    raise FileNotFoundError(f"{p} is missing; export it with src.frames.predict --features"
                                            + (" --call-quality" if v == "call" else ""))
            frames = pd.read_csv(csv, dtype={"video_id": str})
            self.paths[v] = npy
            features = np.load(npy, mmap_mode="r")
            if len(features) != len(frames):
                raise ValueError(f"{npy.name} has {len(features)} rows but {csv.name} has {len(frames)}")
            self.dim = features.shape[1]
            keys = frames[KEYS + ["frame"]]
            if reference is None:
                reference = keys
                self.meta, self.rows = window_index(frames, min_frames)
            elif not keys.equals(reference):
                raise ValueError(f"{csv.name} does not have the same frames as the other versions")
        self.versions = list(versions)
        self._features = {}  # opened on first use, so DataLoader workers do not copy the arrays

    def __len__(self) -> int:
        return len(self.meta)

    def _array(self, version: str) -> np.ndarray:
        if version not in self._features:
            self._features[version] = np.load(self.paths[version], mmap_mode="r")
        return self._features[version]

    def __getitem__(self, i: int):
        version = self.versions[int(torch.randint(len(self.versions), (1,)))] if len(self.versions) > 1 \
            else self.versions[0]
        start, n = int(self.meta["start"].iloc[i]), min(int(self.meta["n_frames"].iloc[i]), self.max_len)
        x = np.zeros((self.max_len, self.dim), np.float32)
        x[:n] = self._array(version)[self.rows[start:start + n]]
        mask = np.zeros(self.max_len, bool)
        mask[:n] = True
        return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(float(self.meta["label"].iloc[i])), i
