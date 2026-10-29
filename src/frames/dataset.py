"""
Face crops for the frame models (Step 3).

Reads {data_root}/processed/faces.csv, written by extract_faces, and serves one
face crop at a time, resized and normalised for the chosen backbone.
"""

import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.frames.degrade import apply_call, call_settings, degrade
from src.frames.mask import apply_mask
from src.frames.sbi import self_blend


def load_frames(data_root: Path, split: str, datasets: list[str] | None = None,
                exclude_methods: list[str] = (), min_score: float = 0.0, min_side: int = 0) -> pd.DataFrame:
    """Rows of faces.csv that have a crop and pass the quality filters: RetinaFace
    score, and crop size in the original frame (small faces are mostly upscaled blur).
    datasets=None keeps every dataset."""
    faces = pd.read_csv(data_root / "processed" / "faces.csv", dtype={"crop": str})
    # Row number within its video, counted before any row is dropped: landmarks68.npy is indexed by it.
    faces["lm_row"] = faces.groupby("video_id").cumcount()
    keep = (faces["crop"].notna() & (faces["split"] == split) & ~faces["method"].isin(exclude_methods)
            & (faces["score"] >= min_score) & (faces["crop_side"] >= min_side))
    if datasets is not None:
        keep &= faces["dataset"].isin(datasets)
    return faces[keep].reset_index(drop=True)


def evenly_spaced(frames: pd.DataFrame, per_video: int) -> pd.DataFrame:
    """At most `per_video` frames from each video, spread over all its frames.
    Used for validation, where every frame would be slow and add little."""
    pos = frames.groupby("video_id").cumcount()
    n = frames.groupby("video_id")["video_id"].transform("size")
    # keep the first frame of each of `per_video` equal parts of the video
    keep = np.floor(pos * per_video / n) != np.floor((pos - 1) * per_video / n)
    return frames[keep].reset_index(drop=True)


class FaceFrames(Dataset):
    """One face crop per item: (image tensor, label, row number in `frames`).

    mask:          "mask" input setting: black out everything but the face (see mask.py);
                   False is the "no-mask" setting, the plain crop
    train:         random horizontal flip, and call-quality degradation with probability degrade_prob
    sbi_prob:      chance that a real crop is replaced by a Self-Blended fake (label 1)
    call_quality:  for evaluation; every crop is degraded, the same way each time, with settings
                   drawn once per window so they hold steady over its frames (see degrade.py)
    """

    def __init__(self, frames: pd.DataFrame, faces_dir: Path, data_config: dict, mask: bool = False,
                 train: bool = False, degrade_prob: float = 0.0, sbi_prob: float = 0.0,
                 call_quality: bool = False):
        # plain lists and arrays: DataFrame lookups per item are slow
        self.crops = frames["crop"].tolist()
        self.labels = frames["label"].to_numpy()
        self.video_ids = frames["video_id"].tolist()
        self.windows = frames["window"].to_numpy()
        self.lm_rows = frames["lm_row"].to_numpy()
        self.faces_dir = faces_dir
        _, h, w = data_config["input_size"]
        self.size = (w, h)
        self.mean = np.array(data_config["mean"], np.float32) * 255
        self.std = np.array(data_config["std"], np.float32) * 255
        self.mask = mask
        self.train, self.degrade_prob, self.sbi_prob, self.call_quality = train, degrade_prob, sbi_prob, call_quality

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.faces_dir / self.crops[i]))
        if img is None:
            raise FileNotFoundError(self.faces_dir / self.crops[i])
        label = int(self.labels[i])
        # Call-quality test sets use fixed generators, so every model sees the same images:
        # one per crop (its sensor noise) and one per window (the call's settings).
        rng = np.random.default_rng(zlib.crc32(self.crops[i].encode()) if self.call_quality else None)
        make_sbi = label == 0 and self.sbi_prob and rng.random() < self.sbi_prob
        points = None
        if self.mask or make_sbi:
            # 68 landmarks in crop pixels; every row with a crop has them
            points = np.load(self.faces_dir / self.video_ids[i] / "landmarks68.npy")[self.lm_rows[i]]

        # 1. Self-Blended fake, on the whole crop (training only)
        if make_sbi:
            img, label = self_blend(img, points, rng), 1
        # 2. Call quality: the whole crop is degraded, as the call would degrade the whole frame
        if self.train and rng.random() < self.degrade_prob:
            img = degrade(img, rng)
        elif not self.train and self.call_quality:
            window_rng = np.random.default_rng(zlib.crc32(f"{self.video_ids[i]}/{self.windows[i]}".encode()))
            img = apply_call(img, call_settings(window_rng), rng)
        # 3. MASK setting only: black out everything but the face
        if self.mask:
            img = apply_mask(img, points)
        # 4. Flip last, so the landmarks above never need flipping
        if self.train and rng.random() < 0.5:
            img = cv2.flip(img, 1)

        interp = cv2.INTER_AREA if img.shape[1] > self.size[0] else cv2.INTER_CUBIC
        img = cv2.resize(img, self.size, interpolation=interp)
        x = (img[..., ::-1].astype(np.float32) - self.mean) / self.std  # BGR to RGB
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.tensor(label, dtype=torch.float32), i
