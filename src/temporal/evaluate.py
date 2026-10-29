"""
Evaluates an aggregator (Step 5): window and video metrics from its window-score tables.
    python -m src.temporal.evaluate --dir ~/data/runs/temporal/<mask or no-mask>/<run name>/<aggregator>

A video's score is the mean of its window scores (--video-how max: the highest). Writes {name}_metrics.json
next to each table: AUC, accuracy, F1 and TPR at low FPR per dataset, video AUC per fake method, ms per window.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.metrics import binary_metrics
from src.temporal.windows import available_names, read_windows, video_table, windows_path


def method_aucs(videos: pd.DataFrame) -> dict:
    """Video-level AUC of each fake method against all real videos of the same dataset."""
    real = videos[videos["label"] == 0]
    out = {}
    for method, fakes in videos[videos["label"] == 1].groupby("method"):
        both = pd.concat([real, fakes])
        out[method] = binary_metrics(both["label"], both["score"]).get("auc")
    return out


def evaluate(windows: pd.DataFrame, video_how: str = "mean") -> dict:
    """Metrics per dataset: window_* over windows, video_* over videos, keys as in src.frames.predict."""
    out = {}
    for dataset, part in windows.groupby("dataset"):
        videos = video_table(part, video_how)
        m = {f"window_{k}": v for k, v in binary_metrics(part["label"], part["score"]).items()}
        m.update({f"video_{k}": v for k, v in binary_metrics(videos["label"], videos["score"]).items()})
        m["video_auc_per_method"] = method_aucs(videos)
        out[dataset] = m
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dir", type=Path, required=True, help="aggregator folder under runs/temporal")
    parser.add_argument("--video-how", default="mean", choices=["mean", "max"],
                        help="how window scores make a video score")
    args = parser.parse_args()

    agg_dir = args.dir.expanduser()
    names = available_names(agg_dir)
    if not names:
        raise SystemExit(f"no window-score tables in {agg_dir / 'predictions'}")
    timing_file = agg_dir / "timing.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}

    for name in names:
        windows = read_windows(windows_path(agg_dir, name))
        results = dict(aggregator=agg_dir.name, run=agg_dir.parent.name, split=name.split("_")[0],
                       call_quality=name.endswith("_call"), video_how=args.video_how, **timing)
        results.update(evaluate(windows, args.video_how))
        (agg_dir / "predictions" / f"{name}_metrics.json").write_text(json.dumps(results, indent=2))
        for dataset, m in results.items():
            if isinstance(m, dict):
                print(f"{name} {dataset}: window AUC {m.get('window_auc', float('nan')):.3f} | video AUC "
                      f"{m.get('video_auc', float('nan')):.3f}, acc {m.get('video_acc', float('nan')):.3f}, "
                      f"TPR@1%FPR {m.get('video_tpr_at_fpr', float('nan')):.3f}")


if __name__ == "__main__":
    main()
