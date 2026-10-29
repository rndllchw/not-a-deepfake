"""
Grad-CAM heatmaps for the frame models: which parts of a face crop drove each model's decision.

Needs the runs' best.pt and the face crops, so run it on the GPU machine from the repo root:
    python -m src.analysis.gradcam
Quick test: python -m src.analysis.gradcam --per-dataset 4

For each example crop, each model's map shows the evidence for the decision it made (fake if its
fake probability is >= 0.5, otherwise real), computed on the last layer that still has a spatial
layout: act4 (Xception), bn2 (EfficientNet-B4), or the patch tokens entering the last block (ViT,
CLIP). Crops are preprocessed with src.frames.dataset.FaceFrames, so each model sees them exactly
as in training, masked for mask runs. Each map is scaled to its own maximum.

Examples are chosen from the test predictions (src.frames.predict), one frame per video: the frame
whose ViT score is closest to the video's ViT score.
    in-domain:    per FF++ method and Celeb-DF, a fake and a real every model gets right (median score)
    dfdc_missed:  the DFDC fakes every model misses, lowest mean score first
    dfdc_correct: DFDC fakes and reals every model gets right (median score)

Writes to {out} (default {data_root}/analysis/gradcam):
    examples.csv                      the chosen crops, why, and every model's video and frame score
    1_in_domain.png, 2_dfdc_missed.png, 3_dfdc_correct.png    crop, then each model's map
    4_sanity_random_weights.png       the in-domain crops with trained vs randomly initialised models;
                                      maps that look alike would mean Grad-CAM shows edges, not learning
    face_share.csv                    share of each map inside the face outline (68 landmarks, as in
                                      src.frames.mask), on --per-dataset random test crops per dataset,
                                      next to the share of the crop the face covers
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.analysis.frames import DATASETS, INK, INK_2, RUNS, SURFACE, load_all, plt, save
from src.frames.backbones import FrameClassifier
from src.frames.dataset import FaceFrames, load_frames
from src.frames.mask import apply_mask, face_region
from src.frames.predict import load_model

HEAT = np.array([0xd9, 0x48, 0x0f]) / 255  # one warm hue; opacity carries the strength


def target_layer(model: FrameClassifier) -> tuple[torch.nn.Module, int]:
    """The layer Grad-CAM reads, and how many leading non-patch tokens it has (0 for CNNs)."""
    b = model.backbone
    if hasattr(b, "blocks") and hasattr(b, "num_prefix_tokens"):  # ViT, CLIP: the head reads only the
        return b.blocks[-1].norm1, b.num_prefix_tokens               # class token, so use the last block's input
    if hasattr(b, "act4"):  # Xception
        return b.act4, 0
    if hasattr(b, "bn2"):  # EfficientNet: conv head + BatchNorm + activation
        return b.bn2, 0
    raise ValueError(f"no Grad-CAM layer known for {type(b).__name__}")


class GradCAM:
    def __init__(self, model: FrameClassifier):
        self.model = model
        layer, self.prefix = target_layer(model)
        self.handle = layer.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.act = output.detach()
        output.register_hook(lambda grad: setattr(self, "grad", grad.detach()))

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(batch, h, w) maps scaled to 0-1, and fake probabilities. Each map explains the model's own
        decision: the gradient of the fake logit if it says fake, of its negative if it says real."""
        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(x)
        sign = torch.where(logits >= 0, 1.0, -1.0)  # logit >= 0 is probability >= 0.5
        (sign * logits).sum().backward()
        act, grad = self.act.float(), self.grad.float()
        if act.dim() == 3:  # tokens (batch, tokens, channels) -> (batch, channels, side, side)
            act, grad = act[:, self.prefix:], grad[:, self.prefix:]
            side = int(round(act.shape[1] ** 0.5))
            act, grad = (t.reshape(t.shape[0], side, side, -1).permute(0, 3, 1, 2) for t in (act, grad))
        cam = torch.relu((grad.mean((2, 3), keepdim=True) * act).sum(1))
        cam = cam / cam.flatten(1).amax(1).clamp(min=1e-8)[:, None, None]
        return cam.cpu().numpy(), torch.sigmoid(logits.detach().float()).cpu().numpy()

    def remove(self) -> None:
        self.handle.remove()


def crop_rows(data_root: Path, picks: pd.DataFrame) -> pd.DataFrame:
    """faces.csv rows (crop path, landmark row) for picks with video_id, window, frame, in picks' order."""
    faces = load_frames(data_root, "test")  # no quality filters: the picks already passed them
    cols = ["video_id", "window", "frame", "crop", "lm_row", "label", "dataset", "method"]
    rows = picks[["video_id", "window", "frame"]].merge(faces[cols], on=["video_id", "window", "frame"], how="left")
    if rows["crop"].isna().any():
        raise ValueError(f"no crop for {rows.loc[rows['crop'].isna(), 'video_id'].tolist()}")
    return rows


def typical(videos: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """The k videos around the median mean score."""
    videos = videos.sort_values("mean_score")
    start = max(0, len(videos) // 2 - k // 2)
    return videos.iloc[start:start + k]


def choose_examples(runs_dir: Path) -> pd.DataFrame:
    test = load_all(runs_dir, "test")
    models = list(test)
    videos = next(iter(test.values()))[["dataset", "method", "label"]].copy()
    for model, scores in test.items():
        videos[model] = scores["prob"]
    says_fake = videos[models] >= 0.5
    videos["models_right"] = says_fake.eq(videos["label"].astype(bool), axis=0).sum(1)
    videos["mean_score"] = videos[models].mean(1)
    right = videos[videos["models_right"] == len(models)]

    picks = []
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "real"):
        picks.append(typical(right[(right["dataset"] == "ffpp") & (right["method"] == method)]).assign(group="in_domain"))
    for label in (1, 0):
        picks.append(typical(right[(right["dataset"] == "celebdf") & (right["label"] == label)]).assign(group="in_domain"))
    dfdc = videos[videos["dataset"] == "dfdc"]
    missed = dfdc[(dfdc["label"] == 1) & (dfdc["models_right"] == 0)]
    picks.append(missed.nsmallest(5, "mean_score").assign(group="dfdc_missed"))
    dfdc_right = right[right["dataset"] == "dfdc"]
    picks.append(typical(dfdc_right[dfdc_right["label"] == 1], 3).assign(group="dfdc_correct"))
    picks.append(typical(dfdc_right[dfdc_right["label"] == 0], 2).assign(group="dfdc_correct"))
    picks = pd.concat(picks).reset_index(names="video_id")

    # one frame per video: the one whose ViT score is closest to the video's ViT score
    vit = pd.read_csv(runs_dir / next(r for r, m in RUNS.items() if m == "ViT-B/16") / "predictions" / "test.csv",
                      dtype={"video_id": str})
    vit = vit[vit["video_id"].isin(picks["video_id"])]
    vit["gap"] = (vit["prob"] - vit["video_id"].map(picks.set_index("video_id")["ViT-B/16"])).abs()
    frames = vit.loc[vit.groupby("video_id")["gap"].idxmin(), ["video_id", "window", "frame"]]
    return picks.merge(frames, on="video_id")


def random_frames(runs_dir: Path, per_dataset: int, seed: int = 0) -> pd.DataFrame:
    """per_dataset test videos per dataset (half real, half fake), one random kept frame each."""
    vit = pd.read_csv(runs_dir / next(r for r, m in RUNS.items() if m == "ViT-B/16") / "predictions" / "test.csv",
                      dtype={"video_id": str})
    rng = np.random.default_rng(seed)
    out = []
    for (dataset, label), part in vit.groupby(["dataset", "label"]):
        ids = part["video_id"].unique()
        chosen = rng.choice(ids, min(len(ids), per_dataset // 2), replace=False)
        for video_id in chosen:
            rows = part[part["video_id"] == video_id]
            out.append(rows.iloc[int(rng.integers(len(rows)))][["video_id", "window", "frame"]])
    return pd.DataFrame(out).reset_index(drop=True)


def run_cams(model: FrameClassifier, rows: pd.DataFrame, faces_dir: Path, mask: bool, device: str,
             batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM maps and fake probabilities for every row, in order."""
    cam = GradCAM(model)
    dataset = FaceFrames(rows, faces_dir, model.data_config, mask=mask)
    maps, probs = [], []
    for x, _, _ in DataLoader(dataset, batch_size):
        m, p = cam(x.to(device))
        maps.append(m)
        probs.append(p)
    cam.remove()
    return np.concatenate(maps), np.concatenate(probs)


def cams_for_runs(runs_dir: Path, rows: pd.DataFrame, faces_dir: Path, device: str) -> dict:
    """{model name: (maps, probs, mask setting)} for every run in RUNS that has a best.pt."""
    out = {}
    for run, name in RUNS.items():
        path = runs_dir / run / "best.pt"
        if not path.exists():
            continue
        model, args = load_model(path, device)
        out[name] = (*run_cams(model, rows, faces_dir, args.get("mask", False), device), args.get("mask", False))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def load_crop(faces_dir: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The crop (BGR) and its 68 landmarks in crop pixels."""
    img = cv2.imread(str(faces_dir / row["crop"]))
    points = np.load(faces_dir / row["video_id"] / "landmarks68.npy")[row["lm_row"]]
    return img, points


def overlay(ax, img: np.ndarray, cam: np.ndarray | None) -> None:
    """The crop in grey, with the map on top as one warm hue whose opacity is the map's strength."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    ax.imshow(0.15 + 0.7 * grey, cmap="gray", vmin=0, vmax=1)  # lifted and flattened so the heat reads
    if cam is not None:
        heat = np.clip(cv2.resize(cam, img.shape[1::-1], interpolation=cv2.INTER_CUBIC), 0, 1)
        rgba = np.dstack([np.broadcast_to(HEAT, heat.shape + (3,)), 0.8 * heat])
        ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def grid(rows: pd.DataFrame, cams: dict, faces_dir: Path, title: str, out: Path, name: str) -> None:
    models = list(cams)
    fig, axes = plt.subplots(len(rows), 1 + len(models), figsize=(1.75 * (1 + len(models)), 1.95 * len(rows) + 0.6),
                             squeeze=False)
    for i, (_, row) in enumerate(rows.iterrows()):
        img, points = load_crop(faces_dir, row)
        truth = "fake" if row["label"] == 1 else "real"
        method = DATASETS[row["dataset"]] if row["method"] in ("fake", "real") and row["dataset"] == "dfdc" \
            else f"{DATASETS[row['dataset']]}: {row['method']}"
        overlay(axes[i, 0], img, None)
        axes[i, 0].set_ylabel(f"{method}\n{truth} · {row['video_id'].rsplit('/', 1)[-1]}", fontsize=8, color=INK,
                              rotation=0, ha="right", va="center", labelpad=6)
        for j, model in enumerate(models, 1):
            maps, probs, mask = cams[model]
            shown = apply_mask(img, points) if mask else img
            overlay(axes[i, j], shown, maps[i])
            p = float(probs[i])
            right = (p >= 0.5) == (row["label"] == 1)
            axes[i, j].set_xlabel(f"fake {p:.2f}" + ("" if right else "  ✗"), fontsize=8,
                                  color=INK if right else INK_2, labelpad=2)
    for j, header in enumerate(["Crop"] + models):
        axes[0, j].set_title(header, fontsize=9, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5)
    fig.text(0.01, 0.005, "Orange = regions that drove the model's decision (each map scaled to its own maximum). "
             "Number = this frame's fake probability; ✗ = frame wrong at 0.5 (groups are chosen on video scores).", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    save(fig, out, name)


def face_share(rows: pd.DataFrame, cams: dict, faces_dir: Path) -> pd.DataFrame:
    """Per crop and model: share of the map inside the face outline, and the share of the crop the face covers."""
    records = []
    for i, (_, row) in enumerate(rows.iterrows()):
        img, points = load_crop(faces_dir, row)
        face = face_region(img.shape, points).astype(np.float32)
        for model, (maps, probs, _) in cams.items():
            heat = np.clip(cv2.resize(maps[i], img.shape[1::-1], interpolation=cv2.INTER_CUBIC), 0, None)
            records.append(dict(model=model, dataset=DATASETS[row["dataset"]], label=int(row["label"]),
                                heat_in_face=float((heat * face).sum() / max(heat.sum(), 1e-8)),
                                face_area=float(face.mean())))
    return pd.DataFrame(records)


def sanity(runs_dir: Path, rows: pd.DataFrame, cams: dict, faces_dir: Path, device: str, out: Path) -> None:
    """Trained vs randomly initialised ViT and Xception on the same crops."""
    torch.manual_seed(0)
    compare = {}
    for name in ("ViT-B/16", "Xception"):
        if name not in cams:
            continue
        run = next(r for r, m in RUNS.items() if m == name)
        backbone = json.loads((runs_dir / run / "args.json").read_text())["backbone"]
        random_model = FrameClassifier(backbone, pretrained=False).to(device).eval()
        compare[name] = cams[name]
        compare[f"{name}\nrandom weights"] = (*run_cams(random_model, rows, faces_dir, False, device), False)
        del random_model
    grid(rows, compare, faces_dir, "Sanity check: trained vs randomly initialised models (the random maps should "
         "look unrelated to the trained ones)", out, "4_sanity_random_weights.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--out", type=Path, default=None, help="default: {data_root}/analysis/gradcam")
    parser.add_argument("--per-dataset", type=int, default=60,
                        help="random test crops per dataset (half real, half fake) for face_share.csv")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    runs_dir, faces_dir = data_root / "runs" / "frames", data_root / "faces"
    out = (args.out or data_root / "analysis" / "gradcam").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pd.set_option("display.width", 200)

    picks = choose_examples(runs_dir)
    rows = crop_rows(data_root, picks)
    print(f"{len(rows)} examples on {device}")
    cams = cams_for_runs(runs_dir, rows, faces_dir, device)
    examples = picks.copy()
    for model, (_, probs, _) in cams.items():
        examples[f"{model} frame"] = probs
    examples.to_csv(out / "examples.csv", index=False)

    titles = {"in_domain": "In-domain test videos every model gets right (FF++ per method, Celeb-DF), one frame each",
              "dfdc_missed": "DFDC (unseen) fake videos every model misses, lowest mean score first, one frame each",
              "dfdc_correct": "DFDC (unseen) fake and real videos every model gets right, one frame each"}
    for k, (group, title) in enumerate(titles.items(), 1):
        index = np.flatnonzero(picks["group"] == group)
        grid(rows.iloc[index], {m: (c[0][index], c[1][index], c[2]) for m, c in cams.items()},
             faces_dir, title, out, f"{k}_{group}.png")
    index = np.flatnonzero(picks["group"] == "in_domain")
    sanity(runs_dir, rows.iloc[index], {m: (c[0][index], c[1][index], c[2]) for m, c in cams.items()},
           faces_dir, device, out)

    sample = crop_rows(data_root, random_frames(runs_dir, args.per_dataset))
    print(f"face share on {len(sample)} random test crops")
    shares = face_share(sample, cams_for_runs(runs_dir, sample, faces_dir, device), faces_dir)
    table = shares.groupby(["model", "dataset"], sort=False)[["heat_in_face", "face_area"]].mean()
    table.to_csv(out / "face_share.csv")
    print(table.round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
