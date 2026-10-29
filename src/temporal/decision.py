"""
Decision rule (Step 6): flag a call when the window score stays at or above a threshold for N consecutive windows.
    python -m src.temporal.decision --dir ~/data/runs/temporal/<mask or no-mask>/<run name>/<aggregator>

Chooses the threshold and N on val (clean and call quality together) to catch the most fakes with FPR <= --max-fpr
(1%) on each and on both pooled, then scores test once. Writes decision.json and decision_test.csv (FPR, TPR,
detection delay in windows). "Consecutive" means consecutive window numbers, not adjacent frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.temporal.windows import read_windows, split_name, windows_path

ABOVE_ALL_SCORES = 1.0 + 1e-6  # a threshold that nothing reaches: scores are probabilities, at most 1


def run_score(scores: np.ndarray, n: int) -> float:
    """The highest threshold at which a video is flagged: the best minimum over
    any n consecutive window scores. -inf if the video has fewer than n windows."""
    if len(scores) < n:
        return -np.inf
    return float(sliding_window_view(scores, n).min(axis=1).max())


def video_run_scores(windows: pd.DataFrame, n: int) -> pd.DataFrame:
    """One row per video: label, dataset and run_score(n). The video is flagged
    at threshold t exactly when its run_score is at least t."""
    rows = [dict(video_id=video_id, label=g["label"].iloc[0], dataset=g["dataset"].iloc[0],
                 run_score=run_score(g["score"].to_numpy(), n))
            for video_id, g in windows.sort_values(["video_id", "window"]).groupby("video_id")]
    return pd.DataFrame(rows)


def rate_at(run_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Share of videos flagged at each threshold."""
    if len(run_scores) == 0:
        return np.full(len(thresholds), np.nan)
    ordered = np.sort(run_scores)
    return (len(ordered) - np.searchsorted(ordered, thresholds, side="left")) / len(ordered)


def choose_rule(val: dict[str, pd.DataFrame], ns=(1, 2, 3, 4), max_fpr: float = 0.01) -> dict:
    """Threshold and N from validation window-score tables, one per condition
    (e.g. {"clean": ..., "call": ...}). Per N, the lowest threshold whose FPR is at
    most max_fpr on every condition and on all of them pooled; then the N with the
    highest pooled TPR (the smaller N if tied)."""
    best = None
    for n in ns:
        per_condition = {cond: video_run_scores(table, n) for cond, table in val.items()}
        groups = dict(per_condition, pooled=pd.concat(per_condition.values()))
        finite = np.concatenate([g["run_score"].to_numpy() for g in groups.values()])
        candidates = np.unique(np.append(finite[np.isfinite(finite)], ABOVE_ALL_SCORES))
        fpr = {name: rate_at(g.loc[g["label"] == 0, "run_score"].to_numpy(), candidates)
               for name, g in groups.items()}
        # FPR only falls as the threshold rises, so the lowest passing threshold has the highest TPR
        passes = np.all([np.nan_to_num(f) <= max_fpr for f in fpr.values()], axis=0)
        t = int(np.argmax(passes))  # candidates end with ABOVE_ALL_SCORES, which always passes
        tpr_pooled = rate_at(groups["pooled"].loc[groups["pooled"]["label"] == 1, "run_score"].to_numpy(),
                             candidates[t:t + 1])[0]
        if best is None or tpr_pooled > best["val_tpr_pooled"]:
            best = dict(n=n, threshold=float(candidates[t]), max_fpr=max_fpr, val_tpr_pooled=float(tpr_pooled),
                        val_fpr={name: float(f[t]) for name, f in fpr.items()},
                        val_tpr={name: float(rate_at(g.loc[g["label"] == 1, "run_score"].to_numpy(),
                                                     candidates[t:t + 1])[0]) for name, g in groups.items()})
    return best


def flag_delay(scores: np.ndarray, threshold: float, n: int) -> int | None:
    """Windows seen when the video is first flagged (the end of the first run of n
    windows at or above the threshold), or None if it never is."""
    above = scores >= threshold
    if len(above) < n:
        return None
    ok = sliding_window_view(above, n).all(axis=1)
    return int(np.argmax(ok)) + n if ok.any() else None


def apply_rule(windows: pd.DataFrame, threshold: float, n: int) -> pd.DataFrame:
    """FPR, TPR and median detection delay of a rule, per dataset and over all of them."""
    videos = []
    for video_id, g in windows.sort_values(["video_id", "window"]).groupby("video_id"):
        delay = flag_delay(g["score"].to_numpy(), threshold, n)
        videos.append(dict(label=g["label"].iloc[0], dataset=g["dataset"].iloc[0],
                           flagged=delay is not None, delay=delay if delay is not None else np.nan))
    videos = pd.DataFrame(videos)
    rows = []
    for dataset, part in [("all", videos)] + list(videos.groupby("dataset")):
        real, fake = part[part["label"] == 0], part[part["label"] == 1]
        rows.append(dict(dataset=dataset, n_real=len(real), n_fake=len(fake),
                         fpr=real["flagged"].mean() if len(real) else np.nan,
                         tpr=fake["flagged"].mean() if len(fake) else np.nan,
                         median_delay_windows=fake.loc[fake["flagged"], "delay"].median()))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dir", type=Path, required=True, help="aggregator folder under runs/temporal")
    parser.add_argument("--max-fpr", type=float, default=0.01, help="false-positive rate allowed on val")
    parser.add_argument("--max-n", type=int, default=4, help="largest number of consecutive windows tried")
    args = parser.parse_args()

    agg_dir = args.dir.expanduser()
    val = {cond: read_windows(windows_path(agg_dir, split_name("val", call)))
           for cond, call in (("clean", False), ("call", True))
           if windows_path(agg_dir, split_name("val", call)).exists()}
    if not val:
        raise SystemExit(f"no val window-score tables in {agg_dir / 'predictions'}")
    if len(val) < 2:
        print(f"warning: only the {', '.join(val)} val set exists, the rule is chosen on it alone")
    max_windows = min(g.groupby("video_id").size().max() for g in val.values())
    ns = range(1, min(args.max_n, max_windows) + 1)

    rule = choose_rule(val, ns, args.max_fpr)
    (agg_dir / "decision.json").write_text(json.dumps(rule, indent=2))
    print(f"rule: flag after {rule['n']} consecutive windows >= {rule['threshold']:.4f} | val FPR "
          + ", ".join(f"{k} {v:.3f}" for k, v in rule["val_fpr"].items())
          + f" | val TPR pooled {rule['val_tpr_pooled']:.3f}")

    tables = []
    for call_quality in (False, True):
        name = split_name("test", call_quality)
        if windows_path(agg_dir, name).exists():
            result = apply_rule(read_windows(windows_path(agg_dir, name)), rule["threshold"], rule["n"])
            tables.append(result.assign(test_set="call quality" if call_quality else "clean"))
    if tables:
        test = pd.concat(tables)[["test_set", "dataset", "n_real", "n_fake", "fpr", "tpr", "median_delay_windows"]]
        test.to_csv(agg_dir / "decision_test.csv", index=False)
        print(test.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
