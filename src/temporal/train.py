"""
Trains a temporal model (LSTM or Transformer) on the frame model's feature sequences (Step 4); the frame model stays fixed.
Export the features first (scripts/export_features.sh), then from the repo root:
    python -m src.temporal.train --frame-run ~/data/runs/frames/<mask or no-mask>/<run name> --model lstm
    python -m src.temporal.train --frame-run <frame run> --model transformer
Quick test: add --epochs 1 --samples-per-epoch 256

Draws real and fake windows equally often, each window clean or call quality at random (--train-versions).
Keeps the epoch with the best mean val video AUC. If the machine stops, rerun with --resume.

Writes runs/temporal/<mask or no-mask>/<run name>/<model>/: args.json, metrics.csv, best.pt, last.pt, and
predictions/ (window-score tables for val and test). Next: src.temporal.evaluate and src.temporal.decision.
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.metrics import binary_metrics
from src.temporal.models import build_model
from src.temporal.predict import score_all, score_windows, window_table
from src.temporal.sequences import WindowSequences, feature_files_exist, sampling_weights
from src.temporal.windows import DEFAULT_MIN_FRAMES, temporal_dir, video_table


def model_options(args: argparse.Namespace) -> dict:
    """Constructor arguments of the chosen model, from the command line."""
    if args.model == "lstm":
        return dict(hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    return dict(d_model=args.hidden, layers=args.layers, heads=args.heads, dropout=args.dropout)


def train_one_epoch(model, loader, optimizer, scheduler, device: str) -> float:
    model.train()
    total, start = 0.0, time.time()
    for step, (x, mask, y, _) in enumerate(loader, 1):
        x, mask, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device)
        loss = F.binary_cross_entropy_with_logits(model(x, mask), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if step % 100 == 0:
            print(f"  step {step}/{len(loader)} | loss {total / step:.4f} | "
                  f"{(time.time() - start) / step:.3f} s/step", flush=True)
    return total / max(1, len(loader))


def val_video_aucs(model, val_sets: dict, device: str, batch_size: int) -> dict:
    """Video AUC on each val version, from the mean of each video's window scores."""
    aucs = {}
    for version, dataset in val_sets.items():
        table = window_table(dataset, score_windows(model, dataset, device, batch_size))
        videos = video_table(table)
        aucs[version] = binary_metrics(videos["label"], videos["score"]).get("auc", float("nan"))
    return aucs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--frame-run", type=Path, required=True,
                        help="frame run folder written by src.frames.train, whose features are used")
    parser.add_argument("--model", required=True, choices=["lstm", "transformer"])
    parser.add_argument("--train-versions", nargs="+", default=["clean", "call"], choices=["clean", "call"],
                        help="train on clean features, call-quality features, or both (mixed per window)")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES,
                        help="windows with fewer frames are left out")
    parser.add_argument("--hidden", type=int, default=256, help="LSTM hidden size, or Transformer width")
    parser.add_argument("--layers", type=int, default=None, help="default: 1 for the LSTM, 2 for the Transformer")
    parser.add_argument("--heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--dropout", type=float, default=None, help="default: 0.2 for the LSTM, 0.1 for the Transformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=6, help="stop after this many epochs without improvement")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None, help="default: the model name")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the run's last.pt (starts a new run if there is none)")
    parser.add_argument("--skip-test", action="store_true", help="do not score val and test at the end")
    args = parser.parse_args()

    args.layers = args.layers or (1 if args.model == "lstm" else 2)
    args.dropout = args.dropout if args.dropout is not None else (0.2 if args.model == "lstm" else 0.1)
    args.run_name = args.run_name or args.model
    frame_run = args.frame_run.expanduser()
    run_dir = temporal_dir(frame_run, args.run_name)
    if (run_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{run_dir} already has a run; pass --resume to continue it, or a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = WindowSequences(frame_run, "train", tuple(args.train_versions), args.min_frames)
    val_sets = {v: WindowSequences(frame_run, "val", (v,), args.min_frames)
                for v in ("clean", "call") if feature_files_exist(frame_run, "val", v)}
    if not val_sets:
        raise SystemExit("no val features; export them with src.frames.predict --split val --features")
    args.d_in = train.dim
    for name, part in (("train", train.meta), *((f"val {v}", d.meta) for v, d in val_sets.items())):
        print(f"{name}: {len(part)} windows from {part['video_id'].nunique()} videos | "
              + ", ".join(f"{m} {n}" for m, n in part.groupby("method")["video_id"].nunique().items()))

    sampler = WeightedRandomSampler(sampling_weights(train.meta), args.samples_per_epoch,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=device == "cuda", drop_last=True, persistent_workers=args.workers > 0)

    options = model_options(args)
    model = build_model(args.model, train.dim, **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup = max(1, total_steps // 20)
    # linear warm-up for the first 5% of steps, then cosine decay to 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup)
                                                  * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    # plain types only, so checkpoints load with torch.load's safe default
    saved_args = json.loads(json.dumps(vars(args), default=str))
    start_epoch, best_auc, stale = 0, -1.0, 0
    if args.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_auc, stale = state["epoch"], state["best_auc"], state["stale"]
        print(f"resuming after epoch {start_epoch} (best mean val video AUC {best_auc:.4f})")
    else:
        (run_dir / "args.json").write_text(json.dumps(saved_args, indent=2))

    print(f"{args.run_name}: {args.epochs} epochs x {len(train_loader)} steps on {device}, lr {args.lr}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    for epoch in range(start_epoch, args.epochs):
        if stale >= args.patience:  # checked here too, so resuming a stopped run does not train again
            print(f"no improvement for {args.patience} epochs, stopping")
            break
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        aucs = val_video_aucs(model, val_sets, device, args.batch_size * 4)
        mean_auc = sum(aucs.values()) / len(aucs)
        row = dict(epoch=epoch + 1, train_loss=loss, lr=scheduler.get_last_lr()[0],
                   **{f"val_video_auc_{v}": a for v, a in aucs.items()}, val_video_auc_mean=mean_auc,
                   minutes=(time.time() - start) / 60)
        log = run_dir / "metrics.csv"
        pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"epoch {epoch + 1}: loss {loss:.4f} | val video AUC "
              + ", ".join(f"{v} {a:.4f}" for v, a in aucs.items()) + f" | {row['minutes']:.1f} min", flush=True)

        if mean_auc > best_auc:
            best_auc, stale = mean_auc, 0
            torch.save(dict(model=model.state_dict(), name=args.model, d_in=train.dim, options=options,
                            args=saved_args, epoch=epoch + 1), run_dir / "best.pt")
        else:
            stale += 1
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch + 1, best_auc=best_auc, stale=stale),
                   run_dir / "last.pt")
    print(f"best mean val video AUC {best_auc:.4f}; weights in {run_dir / 'best.pt'}")

    if not args.skip_test:
        score_all(run_dir, batch_size=args.batch_size * 4)


if __name__ == "__main__":
    main()
