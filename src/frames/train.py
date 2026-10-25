"""
Fine-tunes one pretrained backbone to tell real from fake face crops.

Each epoch draws --samples-per-epoch crops with balanced sampling: real and fake
are drawn equally often, and within each, every video equally often however
many frames it has, so the larger fake class and longer videos do not dominate.
After each epoch the model is scored on up to --val-frames crops per val video,
and the checkpoint with the best video-level val AUC is kept. At the end, the
test split is scored clean and at call quality (src.frames.predict).

Input setting: without --mask the model sees the plain face crop ("no-mask");
with --mask everything but the face is blacked out ("mask", see mask.py).
The two settings are kept in separate folders so their runs never mix.

Run on the GPU machine from the repo root, after extract_faces, inside tmux:
    python -m src.frames.train --backbone efficientnet_b4
    python -m src.frames.train --backbone efficientnet_b4 --mask
    python -m src.frames.train --backbone efficientnet_b4 --sbi only
    python -m src.frames.train --backbone clip_vit_b16 --train-mode norm
    python -m src.frames.train --backbone xception --datasets ffpp --exclude-methods NeuralTextures
Quick test that everything runs:
    python -m src.frames.train --backbone efficientnet_b4 --epochs 1 --samples-per-epoch 256 --val-frames 1
All baselines, with and without the mask: scripts/run_frame_baselines.sh

Writes {data_root}/runs/frames/<mask or no-mask>/<run name>/:
    args.json    the settings used
    metrics.csv  one row per epoch: train loss, val frame and video metrics
    best.pt      weights with the best val video AUC (used by src.frames.predict)
    last.pt      everything needed to continue; if the machine stops (e.g. SmartSchedule),
                 rerun the same command with --resume
    predictions/ test scores and metrics, clean (test_*) and at call quality (test_call_*)
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.frames.backbones import BACKBONES, CNNS, FrameClassifier, set_trainable
from src.frames.dataset import FaceFrames, evenly_spaced, load_frames
from src.frames.predict import predict_frames, score_run
from src.metrics import frame_and_video_metrics

# --sbi: (share of draws that are real crops, chance a drawn real crop becomes a Self-Blended fake)
SBI_MODES = {
    "none": (0.5, 0.0),
    "only": (1.0, 0.5),    # as in the SBI paper: no dataset fakes, half the real crops are blended
    "mix": (0.75, 1 / 3),  # 1/2 real, 1/4 Self-Blended, 1/4 dataset fakes
}


def default_lr(backbone: str, mode: str) -> float:
    """Full fine-tuning of a Transformer needs a smaller step than a CNN;
    training only a few layers can take a much larger one."""
    if mode == "linear":
        return 1e-3
    if mode == "norm":
        return 1e-4
    return 1e-4 if backbone in CNNS else 2e-5


def sampling_weights(frames: pd.DataFrame, real_share: float) -> np.ndarray:
    """Weight per frame so that real crops make up `real_share` of draws, and
    within real and fake, each video is drawn equally often."""
    frames_in_video = frames.groupby("video_id")["video_id"].transform("size")
    videos_in_class = frames.groupby("label")["video_id"].transform("nunique")
    class_share = np.where(frames["label"] == 0, real_share, 1 - real_share)
    return class_share / (frames_in_video * videos_in_class).to_numpy()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device: str, train_mode: str) -> float:
    model.train()
    if train_mode == "linear":
        model.backbone.eval()  # a frozen backbone also keeps its BatchNorm statistics fixed
    total, start = 0.0, time.time()
    for step, (x, y, _) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits, _ = model(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        if step % 100 == 0:
            print(f"  step {step}/{len(loader)} | loss {total / step:.4f} | "
                  f"{(time.time() - start) / step:.2f} s/step", flush=True)
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--backbone", required=True, choices=list(BACKBONES))
    parser.add_argument("--mask", action="store_true",
                        help="black out everything but the face (the 'mask' input setting)")
    parser.add_argument("--train-mode", default="full", choices=["full", "linear", "norm"],
                        help="which weights to train (see backbones.set_trainable)")
    parser.add_argument("--sbi", default="none", choices=list(SBI_MODES),
                        help="Self-Blended Images: none, only (no dataset fakes), or mix")
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--datasets", nargs="+", default=["ffpp", "celebdf"])
    parser.add_argument("--exclude-methods", nargs="*", default=[],
                        help="fake methods left out of train and val, to test as unseen methods")
    parser.add_argument("--min-score", type=float, default=0.9, help="minimum RetinaFace score")
    parser.add_argument("--min-side", type=int, default=64,
                        help="minimum crop side in the original frame, in pixels")
    parser.add_argument("--degrade-prob", type=float, default=0.5,
                        help="share of training crops degraded to call quality")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="default depends on backbone and --train-mode")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5, help="stop after this many epochs without improvement")
    parser.add_argument("--val-frames", type=int, default=8, help="crops scored per val video")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the run's last.pt (starts a new run if there is none)")
    parser.add_argument("--skip-test", action="store_true", help="do not score the test split at the end")
    args = parser.parse_args()

    args.lr = args.lr or default_lr(args.backbone, args.train_mode)
    mask_setting = "mask" if args.mask else "no-mask"
    args.run_name = args.run_name or "_".join(
        [args.backbone, args.train_mode, mask_setting, "+".join(args.datasets), f"sbi-{args.sbi}"]
        + [f"no-{m}" for m in args.exclude_methods])
    data_root = args.data_root.expanduser()
    run_dir = data_root / "runs" / "frames" / mask_setting / args.run_name
    if (run_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{run_dir} already has a run; pass --resume to continue it, or a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FrameClassifier(args.backbone).to(device)
    set_trainable(model, args.train_mode)

    real_share, sbi_prob = SBI_MODES[args.sbi]
    filters = dict(datasets=args.datasets, exclude_methods=args.exclude_methods,
                   min_score=args.min_score, min_side=args.min_side)
    train = load_frames(data_root, "train", **filters)
    if args.sbi == "only":
        train = train[train["label"] == 0].reset_index(drop=True)
    val = evenly_spaced(load_frames(data_root, "val", **filters), args.val_frames)
    for name, part in (("train", train), ("val", val)):
        print(f"{name}: {len(part)} crops from {part['video_id'].nunique()} videos | "
              + ", ".join(f"{m} {n}" for m, n in part.groupby("method")["video_id"].nunique().items()))

    faces_dir = data_root / "faces"
    sampler = WeightedRandomSampler(sampling_weights(train, real_share), args.samples_per_epoch,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(
        FaceFrames(train, faces_dir, model.data_config, mask=args.mask, train=True,
                   degrade_prob=args.degrade_prob, sbi_prob=sbi_prob),
        args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=device == "cuda",
        drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(FaceFrames(val, faces_dir, model.data_config, mask=args.mask), args.batch_size * 2,
                            num_workers=args.workers, pin_memory=device == "cuda")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup = max(1, total_steps // 20)
    # linear warm-up for the first 5% of steps, then cosine decay to 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup)
                                                  * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    # plain types only, so checkpoints load with torch.load's safe default
    saved_args = json.loads(json.dumps(vars(args), default=str))
    start_epoch, best_auc, stale = 0, -1.0, 0
    if args.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_auc, stale = state["epoch"], state["best_auc"], state["stale"]
        print(f"resuming after epoch {start_epoch} (best val video AUC {best_auc:.4f})")
    else:
        (run_dir / "args.json").write_text(json.dumps(saved_args, indent=2))

    print(f"{args.run_name}: {args.epochs} epochs x {len(train_loader)} steps on {device}, lr {args.lr}")
    for epoch in range(start_epoch, args.epochs):
        if stale >= args.patience:  # checked here too, so resuming a stopped run does not train again
            print(f"no improvement for {args.patience} epochs, stopping")
            break
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.train_mode)
        val["prob"], _ = predict_frames(model, val_loader, device)
        metrics = frame_and_video_metrics(val)
        row = dict(epoch=epoch + 1, train_loss=loss, lr=scheduler.get_last_lr()[0], **metrics,
                   minutes=(time.time() - start) / 60)
        log = run_dir / "metrics.csv"
        pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"epoch {epoch + 1}: loss {loss:.4f} | val frame AUC {metrics.get('frame_auc', float('nan')):.4f}, "
              f"video AUC {metrics.get('video_auc', float('nan')):.4f} | {row['minutes']:.1f} min", flush=True)

        auc = metrics.get("video_auc", -1.0)
        if auc > best_auc:
            best_auc, stale = auc, 0
            torch.save(dict(model=model.state_dict(), args=saved_args, epoch=epoch + 1), run_dir / "best.pt")
        else:
            stale += 1
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch + 1, best_auc=best_auc, stale=stale), run_dir / "last.pt")
    print(f"best val video AUC {best_auc:.4f}; weights in {run_dir / 'best.pt'}")

    if not args.skip_test:
        for call_quality in (False, True):
            name = "test_call" if call_quality else "test"
            if (run_dir / "predictions" / f"{name}_metrics.json").exists():
                print(f"{name} already scored")
                continue
            score_run(run_dir, data_root, "test", call_quality=call_quality,
                      batch_size=args.batch_size * 2, workers=args.workers)


if __name__ == "__main__":
    main()
