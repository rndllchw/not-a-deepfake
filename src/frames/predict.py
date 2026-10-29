"""
Scores face crops with a trained frame model and reports the metrics.

train already scores the test split (clean and call quality) when it finishes;
use this to score again, other splits, or new datasets (e.g. DFDC). From the
repo root on the GPU machine:
    python -m src.frames.predict --run ~/data/runs/frames/no-mask/<run name>
    python -m src.frames.predict --run ~/data/runs/frames/mask/<run name> --call-quality
Feature sequences for the temporal models (Step 4), for each split:
    python -m src.frames.predict --run ~/data/runs/frames/<mask setting>/<run name> --split train --features

Uses the training run's quality filters and mask setting, so a model is scored
the way it was trained. Methods left out of training (--exclude-methods) are
scored too, as unseen methods.

Writes to {run}/predictions/, where <name> is the split, plus _call with --call-quality:
    <name>.csv           one row per frame: video_id, dataset, method, label, split, window, frame, prob
    <name>_features.npy  (--features) float16, one feature vector per csv row, same order
    <name>_metrics.json  frame and video metrics per dataset, video AUC per fake method,
                         and inference time
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.frames.backbones import FrameClassifier
from src.frames.dataset import FaceFrames, load_frames
from src.metrics import binary_metrics, frame_and_video_metrics, video_scores


def load_model(checkpoint: Path, device: str) -> tuple[FrameClassifier, dict]:
    """A FrameClassifier with trained weights, and the args it was trained with."""
    ckpt = torch.load(checkpoint, map_location=device)
    model = FrameClassifier(ckpt["args"]["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["args"]


@torch.no_grad()
def predict_frames(model: FrameClassifier, loader: DataLoader, device: str,
                   features_out: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Fake probability per dataset row, and the model's time in ms per frame
    (data loading excluded). Feature vectors are written into features_out if given."""
    model.eval()
    probs = np.zeros(len(loader.dataset), np.float32)
    model_seconds = 0.0
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        start = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits, features = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        model_seconds += time.perf_counter() - start
        probs[idx.numpy()] = torch.sigmoid(logits.float()).cpu().numpy()
        if features_out is not None:
            features_out[idx.numpy()] = features.cpu().numpy().astype(np.float16)
    return probs, 1000 * model_seconds / max(1, len(probs))


@torch.no_grad()
def single_frame_ms(model: FrameClassifier, device: str, repeats: int = 100) -> float:
    """Time for one frame at a time (batch size 1), as in a live call."""
    x = torch.zeros(1, *model.data_config["input_size"], device=device)
    for i in range(repeats + 10):
        if i == 10:  # the first runs are slow while the GPU warms up
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def method_aucs(videos: pd.DataFrame) -> dict:
    """Video-level AUC of each fake method against all real videos of the same dataset."""
    real = videos[videos["label"] == 0]
    out = {}
    for method, fakes in videos[videos["label"] == 1].groupby("method"):
        both = pd.concat([real, fakes])
        out[method] = binary_metrics(both["label"], both["prob"]).get("auc")
    return out


def score_run(run: Path, data_root: Path, split: str = "test", datasets: list[str] | None = None,
              call_quality: bool = False, features: bool = False, checkpoint: str = "best.pt",
              batch_size: int = 64, workers: int = 4) -> dict:
    """Scores one split with a run's model, writes the files listed at the top, and returns the metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_args = load_model(run / checkpoint, device)
    mask = train_args.get("mask", False)

    frames = load_frames(data_root, split, datasets,
                         min_score=train_args["min_score"], min_side=train_args["min_side"])
    dataset = FaceFrames(frames, data_root / "faces", model.data_config, mask=mask, call_quality=call_quality)
    loader = DataLoader(dataset, batch_size, num_workers=workers, pin_memory=device == "cuda")
    name = split + ("_call" if call_quality else "")
    out_dir = run / "predictions"
    out_dir.mkdir(exist_ok=True)

    feature_file = None
    if features:
        # written straight to disk: the train split's features do not fit in memory
        feature_file = np.lib.format.open_memmap(out_dir / f"{name}_features.npy", mode="w+", dtype=np.float16,
                                                 shape=(len(frames), model.backbone.num_features))
    print(f"scoring {len(frames)} frames ({name}, {'mask' if mask else 'no-mask'}) with {run.name} on {device}")
    frames["prob"], batched_ms = predict_frames(model, loader, device, feature_file)
    if feature_file is not None:
        feature_file.flush()

    columns = ["video_id", "dataset", "method", "label", "split", "window", "frame", "prob"]
    frames[columns].to_csv(out_dir / f"{name}.csv", index=False)

    results = dict(run=run.name, backbone=train_args["backbone"], mask=mask, split=split, call_quality=call_quality)
    for dataset_name, part in frames.groupby("dataset"):
        results[dataset_name] = frame_and_video_metrics(part)
        results[dataset_name]["video_auc_per_method"] = method_aucs(video_scores(part))
    results["ms_per_frame_batched"] = batched_ms
    results["ms_per_frame_single"] = single_frame_ms(model, device)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(results, indent=2))

    for dataset_name, m in results.items():
        if isinstance(m, dict):
            print(f"{dataset_name}: frame AUC {m.get('frame_auc', float('nan')):.3f} | video AUC "
                  f"{m.get('video_auc', float('nan')):.3f}, EER {m.get('video_eer', float('nan')):.3f}, "
                  f"acc {m.get('video_acc', float('nan')):.3f}, "
                  f"TPR@1%FPR {m.get('video_tpr_at_fpr', float('nan')):.3f} | per method "
                  + ", ".join(f"{k} {v:.3f}" for k, v in m["video_auc_per_method"].items() if v is not None))
    print(f"{batched_ms:.2f} ms/frame batched, {results['ms_per_frame_single']:.2f} ms/frame one at a time")
    print(f"wrote {out_dir / name}*")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run", type=Path, required=True, help="run folder written by src.frames.train")
    parser.add_argument("--checkpoint", default="best.pt", help="file in the run folder")
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--datasets", nargs="+", default=None, help="default: every dataset in faces.csv")
    parser.add_argument("--call-quality", action="store_true", help="degrade every crop like a video call")
    parser.add_argument("--features", action="store_true", help="also save feature vectors (for Step 4)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    score_run(args.run.expanduser(), args.data_root.expanduser(), args.split, args.datasets, args.call_quality,
              args.features, args.checkpoint, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
