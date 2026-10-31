"""
Finds DFDC fakes whose video is identical to their original apart from re-encoding
noise: fakes made by changing only the audio. Our models see frames only, so these
fakes cannot be detected and should be left out of the DFDC results.

Each fake is compared with its `original` (from metadata.json, the index's
source_ids) on every --step-th frame of the first --max-frames, at full resolution
and in grey. The pixel difference is smoothed (Gaussian, --sigma) so single noisy
pixels drop out, and the connected patches where it exceeds --pixel-threshold grey
levels are found. A face swap leaves a face-sized patch (on the Kaggle sample, at
least 0.00025 of the frame, usually 0.001-0.03); re-encoding noise leaves nothing or
specks of a few pixels (at most 0.000024). A fake is flagged when its largest patch,
over all compared frames, is under --threshold of the frame. Fakes with any real
visual change are kept, since a visual model could in principle detect them.

Fakes whose original is not in the index (the Kaggle sample has only some of them)
cannot be checked; they are listed with status "no_original" and kept.

Run on the GPU machine from the repo root (CPU only, a few minutes):
    python -m src.preprocessing.dfdc_audio_only --data-root ~/data

Writes:
    {data_root}/processed/dfdc_audio_only.csv  one row per DFDC fake: video_id, source_id, status,
                                               frames, largest_patch, patch_box (x, y, w, h of the
                                               largest patch), p9999 (99.99th percentile of the
                                               difference, grey levels), audio_only (True = leave out)
Prints the fakes with the smallest patches, so the gap at the threshold can be checked by eye.
"""

import argparse
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def resolve(path: str, data_root: Path) -> str:
    """The index stores absolute paths; if the data has moved, look for the same file under data_root/raw."""
    p = Path(path)
    if p.exists() or "raw" not in p.parts:
        return str(p)
    return str(data_root.joinpath(*p.parts[p.parts.index("raw"):]))


def read_frames(path: str, max_frames: int, step: int) -> list[np.ndarray]:
    """Every step-th of the first max_frames frames, grey, full resolution."""
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while i < max_frames and cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return frames


def largest_patch(diff: np.ndarray, sigma: float, pixel_threshold: float) -> tuple[float, tuple | None]:
    """Share of the frame covered by the largest connected patch of change, and its box (x, y, w, h)."""
    mask = (cv2.GaussianBlur(diff, (0, 0), sigma) > pixel_threshold).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n == 1:  # background only
        return 0.0, None
    k = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return float(stats[k, cv2.CC_STAT_AREA] / diff.size), tuple(int(v) for v in stats[k, :4])


def compare_group(job: tuple[str, str, list[tuple[str, str]], argparse.Namespace]) -> list[dict]:
    """Compare every fake made from one original with that original (read once)."""
    source_id, source_path, fakes, args = job
    cv2.setNumThreads(1)
    try:
        original = read_frames(source_path, args.max_frames, args.step)
    except Exception as e:  # noqa: BLE001 - one bad file should not stop the run
        return [dict(video_id=v, source_id=source_id, status="error", message=f"original: {e}") for v, _ in fakes]
    rows = []
    for video_id, path in fakes:
        try:
            fake = read_frames(path, args.max_frames, args.step)
            n = min(len(fake), len(original))
            if n == 0:
                rows.append(dict(video_id=video_id, source_id=source_id, status="error", message="no frames"))
                continue
            best, box, p9999 = -1.0, None, 0.0
            for a, b in zip(fake[:n], original[:n]):
                if a.shape != b.shape:
                    a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
                diff = cv2.absdiff(a, b)
                p9999 = max(p9999, float(np.percentile(diff, 99.99)))
                share, patch_box = largest_patch(diff, args.sigma, args.pixel_threshold)
                if share > best:
                    best, box = share, patch_box
            rows.append(dict(video_id=video_id, source_id=source_id, status="compared", frames=n,
                             largest_patch=best, patch_box=box, p9999=p9999))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(video_id=video_id, source_id=source_id, status="error", message=str(e)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--index", type=Path, default=None, help="default: {data_root}/processed/index.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: {data_root}/processed/dfdc_audio_only.csv")
    parser.add_argument("--max-frames", type=int, default=300, help="compare within the first N frames")
    parser.add_argument("--step", type=int, default=30, help="compare every N-th frame")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing of the difference, pixels")
    parser.add_argument("--pixel-threshold", type=float, default=6.0,
                        help="smoothed difference (grey levels) above which a pixel counts as changed")
    parser.add_argument("--threshold", type=float, default=0.0001,
                        help="flag a fake when its largest patch of change is under this share of the frame")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    index = pd.read_csv(args.index or data_root / "processed" / "index.csv", dtype={"source_ids": str})
    dfdc = index[index["dataset"] == "dfdc"]
    reals = {r.video_id.split("/")[-1]: resolve(r.path, data_root) for r in dfdc[dfdc["label"] == 0].itertuples()}
    fakes = dfdc[dfdc["label"] == 1]

    rows, jobs = [], []
    for source_id, group in fakes.groupby("source_ids"):
        pairs = [(r.video_id, resolve(r.path, data_root)) for r in group.itertuples()]
        if source_id in reals:
            jobs.append((source_id, reals[source_id], pairs, args))
        else:
            rows += [dict(video_id=v, source_id=source_id, status="no_original") for v, _ in pairs]
    print(f"{len(fakes)} DFDC fakes: {len(fakes) - len(rows)} with their original in the index "
          f"({len(jobs)} originals), {len(rows)} without", flush=True)

    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(compare_group, jobs), 1):
            rows += result
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} originals done", flush=True)

    out = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)
    out["audio_only"] = (out["status"] == "compared") & (out["largest_patch"] < args.threshold)
    path = args.out or data_root / "processed" / "dfdc_audio_only.csv"
    out.to_csv(path, index=False)

    print("\n" + out["status"].value_counts().to_string())
    for _, r in out[out["status"] == "error"].iterrows():
        print(f"error {r['video_id']}: {r['message']}")
    compared = out[out["status"] == "compared"].sort_values("largest_patch")
    if len(compared):
        print(f"\nsmallest patches of change (share of the frame); flagged below {args.threshold}:")
        print(compared.head(12)[["video_id", "source_id", "largest_patch", "p9999", "patch_box", "audio_only"]]
              .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\n{int(out['audio_only'].sum())} of {len(compared)} checked fakes flagged audio-only "
          f"({len(out) - len(compared)} not checked); wrote {path}")


if __name__ == "__main__":
    main()
