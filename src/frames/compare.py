"""
Collects the test results of every frame-model run into one table, to compare
backbones and the mask / no-mask input settings.

Run from the repo root after training:
    python -m src.frames.compare

Writes {data_root}/runs/frames/comparison.csv, one row per run, test set
(clean or call quality) and dataset, and prints video AUC with mask and
no-mask side by side for each backbone.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_dir.glob("*/*/predictions/test*_metrics.json")):
        m = json.loads(path.read_text())
        mask_setting = path.parents[2].name  # runs/frames/<mask or no-mask>/<run>/predictions/...
        for dataset, d in m.items():
            if not isinstance(d, dict):
                continue
            run = path.parents[1].name
            rows.append(dict(
                # the run without its mask setting, so the mask and no-mask versions line up
                model=run.replace(f"_{mask_setting}_", "_"), input=mask_setting, run=run, backbone=m["backbone"],
                test_set="call quality" if m["call_quality"] else "clean", dataset=dataset,
                video_auc=d.get("video_auc"), video_eer=d.get("video_eer"),
                video_acc=d.get("video_acc"), video_f1=d.get("video_f1"),
                video_tpr_at_1pct_fpr=d.get("video_tpr_at_fpr"), frame_auc=d.get("frame_auc"),
                ms_per_frame=m["ms_per_frame_single"],
                **{f"auc_{k}": v for k, v in d["video_auc_per_method"].items()}))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    args = parser.parse_args()

    runs_dir = args.data_root.expanduser() / "runs" / "frames"
    table = collect(runs_dir)
    if table.empty:
        raise SystemExit(f"no scored runs in {runs_dir}")
    table.to_csv(runs_dir / "comparison.csv", index=False)

    pd.set_option("display.width", 200)
    print("video AUC: mask vs no-mask")
    print(table.pivot_table(index=["model", "dataset", "test_set"], columns="input",
                            values="video_auc").round(3).to_string())
    print(f"\nfull table ({len(table)} rows): {runs_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
