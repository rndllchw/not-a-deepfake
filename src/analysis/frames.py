"""
Training curves and error analysis for the frame models. No GPU needed.

Reads the frame runs' metrics.csv and predictions/ (test, test_call, val) from
{data_root}/runs/frames/ and writes figures and tables to --out. From the repo root:
    python -m src.analysis.frames
On a laptop, after copying runs/frames/ (without best.pt, last.pt, *.npy) into <folder>/runs/frames/:
    python -m src.analysis.frames --data-root <folder> --out <folder>/analysis

Writes to {out} (default {data_root}/analysis/frames):
    1_training_curves.png                   train loss and val video AUC per epoch, kept checkpoint marked
    2_vit_score_distributions.png           ViT video scores, real vs fake, per test dataset
    3_dfdc_score_distributions_all_models   the same on DFDC, every model
    4_auc_per_method.png, auc_per_method.csv     video AUC per fake method (clean test)
    5_clean_vs_call_quality.png, clean_vs_call.csv    video AUC clean vs call quality
    thresholds.csv       accuracy, TPR and FPR at 0.5, at a val-chosen threshold, and at each test set's best threshold
    dfdc_hard_videos.csv every DFDC video's score per model and how many models get it wrong at 0.5
A video's score is the mean of its frame probabilities, as in src.metrics.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; no display on the GPU machine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

# run folder under runs/frames -> display name; the order fixes each model's colour in every figure
RUNS = {
    "no-mask/vit_b16_full_no-mask_ffpp+celebdf_sbi-none": "ViT-B/16",
    "no-mask/clip_vit_b16_norm_no-mask_ffpp+celebdf_sbi-none": "CLIP ViT-B/16",
    "no-mask/efficientnet_b4_full_no-mask_ffpp+celebdf_sbi-none": "EfficientNet-B4",
    "no-mask/xception_full_no-mask_ffpp+celebdf_sbi-none": "Xception",
    "mask/xception_full_mask_ffpp+celebdf_sbi-none": "Xception + mask",
}
DATASETS = {"ffpp": "FF++", "celebdf": "Celeb-DF", "dfdc": "DFDC (unseen)"}

# categorical palette (light mode), slots in a fixed order that stays distinguishable with colour blindness
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
COLOR = dict(zip(RUNS.values(), SERIES))
REAL, FAKE = SERIES[0], SERIES[1]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK, "font.size": 9.5,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def video_scores(path: Path) -> pd.DataFrame:
    """One row per video: dataset, method, label and the mean of its frame probabilities."""
    frames = pd.read_csv(path, dtype={"video_id": str})
    return frames.groupby("video_id").agg(dataset=("dataset", "first"), method=("method", "first"),
                                          label=("label", "first"), prob=("prob", "mean"))


def load_all(runs_dir: Path, name: str) -> dict:
    """{display name: video scores} for one predictions file, for every run that has it."""
    out = {}
    for run, label in RUNS.items():
        path = runs_dir / run / "predictions" / f"{name}.csv"
        if path.exists():
            out[label] = video_scores(path)
    return out


def rates(videos: pd.DataFrame, threshold: float) -> dict:
    real, fake = videos[videos.label == 0], videos[videos.label == 1]
    tpr, fpr = (fake.prob >= threshold).mean(), (real.prob >= threshold).mean()
    acc = ((videos.prob >= threshold) == videos.label.astype(bool)).mean()
    return dict(acc=acc, balanced_acc=(tpr + 1 - fpr) / 2, tpr=tpr, fpr=fpr)


def best_threshold(videos: pd.DataFrame) -> float:
    """Threshold with the highest balanced accuracy (ties: the lowest)."""
    candidates = np.unique(videos.prob)
    scores = [rates(videos, t)["balanced_acc"] for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}")


def training_curves(runs_dir: Path, out: Path) -> pd.DataFrame:
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 3.8))
    rows = []
    for run, label in RUNS.items():
        path = runs_dir / run / "metrics.csv"
        if not path.exists():
            continue
        m = pd.read_csv(path)
        best = m.loc[m.video_auc.idxmax()]
        rows.append(dict(model=label, epochs_run=len(m), best_epoch=int(best.epoch),
                         best_val_video_auc=best.video_auc, final_train_loss=m.train_loss.iloc[-1]))
        for ax, col in ((ax_loss, "train_loss"), (ax_auc, "video_auc")):
            ax.plot(m.epoch, m[col], color=COLOR[label], label=label)
        # the checkpoint kept as best.pt
        ax_auc.plot(best.epoch, best.video_auc, "o", ms=8, color=COLOR[label], mec=SURFACE, mew=2, zorder=3)
    ax_loss.set(title="Training loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    ax_auc.set(title="Validation video AUC (dot = checkpoint kept)", xlabel="Epoch", ylabel="AUC")
    ax_loss.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "1_training_curves.png")
    return pd.DataFrame(rows)


def hist_pair(ax, videos: pd.DataFrame, title: str) -> None:
    bins = np.linspace(0, 1, 26)
    for label, color, name in ((0, REAL, "Real"), (1, FAKE, "Fake")):
        p = videos.loc[videos.label == label, "prob"]
        # share of that class, so the 4:1 fake:real imbalance does not hide the reals
        ax.hist(p, bins, weights=np.full(len(p), 1 / max(1, len(p))), color=color, alpha=0.55,
                label=f"{name} (n={len(p)})", edgecolor=SURFACE, linewidth=1)
    ax.axvline(0.5, color=INK_2, lw=1)
    ax.set(title=title, xlim=(0, 1), xlabel="Video score (mean fake probability)")
    ax.legend(loc="upper center")


def score_distributions(test: dict, out: Path) -> None:
    if "ViT-B/16" in test:
        vit = test["ViT-B/16"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for ax, (key, name) in zip(axes, DATASETS.items()):
            hist_pair(ax, vit[vit.dataset == key], name)
        axes[0].set_ylabel("Share of videos in class")
        fig.suptitle("ViT-B/16 test video scores (line = 0.5 threshold)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        save(fig, out, "2_vit_score_distributions.png")

    fig, axes = plt.subplots(1, len(test), figsize=(3.1 * len(test), 3.2), sharey=True, squeeze=False)
    for ax, (label, videos) in zip(axes[0], test.items()):
        hist_pair(ax, videos[videos.dataset == "dfdc"], label)
    axes[0, 0].set_ylabel("Share of videos in class")
    fig.suptitle("DFDC (unseen) test video scores, every model", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "3_dfdc_score_distributions_all_models.png")


def threshold_table(runs_dir: Path, test: dict) -> pd.DataFrame:
    """Accuracy at 0.5, at a threshold chosen on val (runs with val predictions), and
    at the best threshold for that test set itself (an oracle, to show how far off 0.5 is)."""
    val = load_all(runs_dir, "val")
    rows = []
    for label, videos in test.items():
        val_t = best_threshold(val[label]) if label in val else None
        for key, name in DATASETS.items():
            part = videos[videos.dataset == key]
            row = dict(model=label, dataset=name, n_real=int((part.label == 0).sum()),
                       n_fake=int((part.label == 1).sum()))
            row.update({f"{k}@0.5": v for k, v in rates(part, 0.5).items()})
            if val_t is not None:
                row["val_threshold"] = val_t
                row.update({f"{k}@val": v for k, v in rates(part, val_t).items()})
            oracle = best_threshold(part)
            row["oracle_threshold"] = oracle
            row["balanced_acc@oracle"] = rates(part, oracle)["balanced_acc"]
            rows.append(row)
    return pd.DataFrame(rows)


def method_auc(test: dict, out: Path) -> pd.DataFrame:
    """Video AUC of each fake method against all real videos of the same dataset."""
    table = {}
    for label, videos in test.items():
        aucs = {}
        for key, name in DATASETS.items():
            part = videos[videos.dataset == key]
            real = part[part.label == 0]
            for method, fakes in part[part.label == 1].groupby("method"):
                both = pd.concat([real, fakes])
                aucs[name if key == "dfdc" else f"{name}: {method}"] = roc_auc_score(both.label, both.prob)
        table[label] = aucs
    table = pd.DataFrame(table).T

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.55 * len(table)))
    ax.imshow(table.values, cmap=BLUES, vmin=0.75, vmax=1.0, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=25, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.9 else INK)
    ax.set_title("Clean test video AUC per manipulation method (fakes of that method vs all reals of the dataset)",
                 loc="left")
    fig.tight_layout()
    save(fig, out, "4_auc_per_method.png")
    return table


def call_quality_drop(test: dict, test_call: dict, out: Path) -> pd.DataFrame:
    rows = []
    for label in [m for m in test if m in test_call]:
        for key, name in DATASETS.items():
            auc = {}
            for cond, source in (("clean", test), ("call", test_call)):
                part = source[label][source[label].dataset == key]
                auc[cond] = roc_auc_score(part.label, part.prob)
            rows.append(dict(model=label, dataset=name, clean=auc["clean"], call=auc["call"],
                             drop=auc["clean"] - auc["call"]))
    table = pd.DataFrame(rows)

    labels = [m for m in RUNS.values() if m in set(table.model)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.6 + 0.65 * len(labels)), sharey=True)
    for ax, name in zip(axes, DATASETS.values()):
        part = table[table.dataset == name].set_index("model").loc[labels]
        y = np.arange(len(labels))[::-1]
        ax.hlines(y, part.call, part.clean, color=GRID, lw=2)
        ax.plot(part.clean, y, "o", ms=8, color=REAL, mec=SURFACE, mew=2, label="Clean")
        ax.plot(part.call, y, "o", ms=8, color=FAKE, mec=SURFACE, mew=2, label="Call quality")
        ax.set(title=name, xlabel="Video AUC")
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(min(part.call.min(), part.clean.min()) - 0.02, 1.0)
    axes[0].legend(loc="lower left")
    fig.suptitle("Robustness: test video AUC clean vs call quality (note the different x-axis ranges)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "5_clean_vs_call_quality.png")
    return table


def hard_videos(test: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DFDC videos with each model's score and how many models get each one wrong at 0.5,
    and how alike the models' DFDC scores are (Spearman)."""
    scores = pd.DataFrame({label: v.loc[v.dataset == "dfdc", "prob"] for label, v in test.items()})
    models = list(test)
    scores.insert(0, "label", test[models[0]].loc[scores.index, "label"])
    fake = scores.label.to_numpy()[:, None] == 1
    wrong = np.where(fake, scores[models] < 0.5, scores[models] >= 0.5)
    scores["models_wrong"] = wrong.sum(1)
    scores["mean_score"] = scores[models].mean(1)
    scores = scores.sort_values(["models_wrong", "mean_score"], ascending=[False, True])
    return scores, scores[models].corr(method="spearman")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--out", type=Path, default=None, help="default: {data_root}/analysis/frames")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    runs_dir = data_root / "runs" / "frames"
    out = (args.out or data_root / "analysis" / "frames").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    test, test_call = load_all(runs_dir, "test"), load_all(runs_dir, "test_call")
    if not test:
        raise SystemExit(f"no test predictions in {runs_dir}; run src.frames.train or src.frames.predict first")
    pd.set_option("display.width", 200)

    print("\n== Training curves ==")
    print(training_curves(runs_dir, out).round(4).to_string(index=False))

    print("\n== Score distributions (video score mean / median by label) ==")
    score_distributions(test, out)
    summary = pd.concat(test, names=["model"]).reset_index()
    print(summary.groupby(["model", "dataset", "label"]).prob.agg(["size", "mean", "median"])
          .unstack("label").round(3).to_string())

    print("\n== Thresholds ==")
    thresholds = threshold_table(runs_dir, test)
    thresholds.to_csv(out / "thresholds.csv", index=False)
    cols = ["model", "dataset", "n_real", "n_fake", "acc@0.5", "tpr@0.5", "fpr@0.5", "val_threshold",
            "balanced_acc@val", "oracle_threshold", "balanced_acc@oracle"]
    print(thresholds[[c for c in cols if c in thresholds]].round(3).to_string(index=False))

    print("\n== AUC per method ==")
    per_method = method_auc(test, out)
    per_method.to_csv(out / "auc_per_method.csv")
    print(per_method.round(3).to_string())

    if test_call:
        print("\n== Clean vs call quality ==")
        drop = call_quality_drop(test, test_call, out)
        drop.to_csv(out / "clean_vs_call.csv", index=False)
        print(drop.round(3).to_string(index=False))

    if any((v.dataset == "dfdc").any() for v in test.values()):
        print("\n== Hard DFDC videos ==")
        hard, corr = hard_videos(test)
        hard.to_csv(out / "dfdc_hard_videos.csv")
        n = len(test)
        for label, name in ((1, "fakes"), (0, "reals")):
            part = hard[hard.label == label]
            print(f"DFDC {name}: {len(part)}; wrong in all {n} models: {(part.models_wrong == n).sum()}, "
                  f"in none: {(part.models_wrong == 0).sum()}")
        print("Spearman correlation of DFDC video scores between models:")
        print(corr.round(2).to_string())
        print("\nHardest DFDC videos:")
        print(hard.head(15).round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
