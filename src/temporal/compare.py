"""
Collects every aggregator's test results into one table (Step 5).
    python -m src.temporal.compare

Run after src.temporal.evaluate. Writes {data_root}/runs/temporal/comparison.csv (one row per aggregator, test set
and dataset) and prints video AUC side by side. ms_per_window_total = 32 frames through the frame model + the temporal model.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.temporal.windows import frame_run_of

WINDOW_FRAMES = 32


def collect(temporal_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(temporal_dir.glob("*/*/*/predictions/test*_metrics.json")):
        m = json.loads(path.read_text())
        agg_dir = path.parents[1]
        # per-frame time of the frame model, from the frame run's own test metrics
        frame_metrics = frame_run_of(agg_dir) / "predictions" / "test_metrics.json"
        frame_ms = json.loads(frame_metrics.read_text()).get("ms_per_frame_single") if frame_metrics.exists() else None
        head_ms = m.get("ms_head_per_window", 0.0)  # pooling has no model of its own
        decision = agg_dir / "decision.json"
        rule = json.loads(decision.read_text()) if decision.exists() else {}
        for dataset, d in m.items():
            if not isinstance(d, dict):
                continue
            rows.append(dict(
                input=agg_dir.parents[1].name, frame_run=agg_dir.parent.name, aggregator=agg_dir.name,
                test_set="call quality" if m["call_quality"] else "clean", dataset=dataset,
                video_auc=d.get("video_auc"), video_acc=d.get("video_acc"), video_f1=d.get("video_f1"),
                video_tpr_at_1pct_fpr=d.get("video_tpr_at_fpr"), window_auc=d.get("window_auc"),
                ms_head_per_window=head_ms,
                ms_per_window_total=None if frame_ms is None else WINDOW_FRAMES * frame_ms + head_ms,
                rule_n=rule.get("n"), rule_threshold=rule.get("threshold"),
                **{f"auc_{k}": v for k, v in d["video_auc_per_method"].items()}))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    args = parser.parse_args()

    temporal_dir = args.data_root.expanduser() / "runs" / "temporal"
    table = collect(temporal_dir)
    if table.empty:
        raise SystemExit(f"no evaluated aggregators in {temporal_dir}; run src.temporal.evaluate first")
    table.to_csv(temporal_dir / "comparison.csv", index=False)

    pd.set_option("display.width", 200)
    print("video AUC by aggregator")
    print(table.pivot_table(index=["input", "frame_run", "dataset", "test_set"], columns="aggregator",
                            values="video_auc").round(3).to_string())
    print(f"\nfull table ({len(table)} rows): {temporal_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
