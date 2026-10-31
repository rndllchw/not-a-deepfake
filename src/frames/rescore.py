"""
Recomputes the frame runs' metrics files from their saved scores: no model, no GPU.
Use after the list of excluded videos (src.exclusions) changes. From the repo root:
    python -m src.frames.rescore                      every run under {data_root}/runs/frames
    python -m src.frames.rescore --run ~/data/runs/frames/no-mask/<run name>

Rewrites each predictions/<name>_metrics.json that has a <name>.csv next to it, keeping
its run details and timings. Then rerun src.frames.compare for the comparison table.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.exclusions import excluded_videos, metric_parts
from src.metrics import binary_metrics, frame_and_video_metrics, video_scores


def method_aucs(videos: pd.DataFrame) -> dict:
    """Video-level AUC of each fake method against all real videos of the same dataset."""
    real = videos[videos["label"] == 0]
    out = {}
    for method, fakes in videos[videos["label"] == 1].groupby("method"):
        both = pd.concat([real, fakes])
        out[method] = binary_metrics(both["label"], both["prob"]).get("auc")
    return out


def dataset_metrics(frames: pd.DataFrame, excluded: set[str]) -> dict:
    """Frame and video metrics and video AUC per fake method, per dataset, without the
    excluded videos (plus <dataset>_unfiltered with them, see src.exclusions)."""
    out = {}
    for dataset_name, part in metric_parts(frames, excluded):
        out[dataset_name] = frame_and_video_metrics(part)
        out[dataset_name]["video_auc_per_method"] = method_aucs(video_scores(part))
    return out


def print_metrics(results: dict) -> None:
    for dataset_name, m in results.items():
        if isinstance(m, dict):
            print(f"{dataset_name}: frame AUC {m.get('frame_auc', float('nan')):.3f} | video AUC "
                  f"{m.get('video_auc', float('nan')):.3f}, EER {m.get('video_eer', float('nan')):.3f}, "
                  f"acc {m.get('video_acc', float('nan')):.3f}, "
                  f"TPR@1%FPR {m.get('video_tpr_at_fpr', float('nan')):.3f} | per method "
                  + ", ".join(f"{k} {v:.3f}" for k, v in m["video_auc_per_method"].items() if v is not None))


def rescore_run(run: Path, excluded: set[str]) -> int:
    """Rewrites a run's metrics files from its prediction CSVs; returns how many it rewrote."""
    out_dir = run / "predictions"
    done = 0
    for metrics_file in sorted(out_dir.glob("*_metrics.json")):
        scores = out_dir / (metrics_file.name.removesuffix("_metrics.json") + ".csv")
        if not scores.exists():
            continue
        old = json.loads(metrics_file.read_text())
        results = {k: v for k, v in old.items() if not isinstance(v, dict)}  # run details and timings
        results.update(dataset_metrics(pd.read_csv(scores, dtype={"video_id": str}), excluded))
        metrics_file.write_text(json.dumps(results, indent=2))
        print(f"== {run.parent.name}/{run.name} {scores.stem}")
        print_metrics(results)
        done += 1
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--run", type=Path, default=None, help="one run folder; default: every run")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    runs_dir = data_root / "runs" / "frames"
    runs = [args.run.expanduser()] if args.run else sorted(p.parent for p in runs_dir.glob("*/*/predictions"))
    excluded = excluded_videos(data_root)
    done = sum(rescore_run(run, excluded) for run in runs)
    if not done:
        raise SystemExit(f"no prediction files with metrics in {args.run or runs_dir}")
    print(f"\nrewrote {done} metrics files, {len(excluded)} videos excluded")


if __name__ == "__main__":
    main()
