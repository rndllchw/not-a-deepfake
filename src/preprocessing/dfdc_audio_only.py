"""
Finds DFDC fakes whose video is (near-)identical to their original: fakes made by
changing only the audio. Our models see frames only, so these fakes cannot be
detected and should be left out of the DFDC results.

Each fake is compared with its `original` (from metadata.json, the index's
source_ids) on every --step-th frame of the first --max-frames. Frames are turned
grey, shrunk to --width pixels wide and blurred, so re-encoding noise mostly
disappears; a pixel counts as changed if it differs by more than --pixel-threshold
grey levels. A face swap changes the face region (around 1% of the frame) in every
frame; an audio-only fake changes nothing. A fake is flagged when even its most
changed frame has under --threshold of its pixels changed.

Fakes whose original is not in the index (the Kaggle sample has only some of them)
cannot be checked; they are listed with status "no_original" and kept.

Run on the GPU machine from the repo root (CPU only, a few minutes):
    python -m src.preprocessing.dfdc_audio_only --data-root ~/data

Writes:
    {data_root}/processed/dfdc_audio_only.csv  one row per DFDC fake: video_id, source_id, status,
                                               frames, changed_max, changed_median, absdiff_median,
                                               audio_only (True = leave out)
Prints the spread of changed_max, so the threshold can be checked by eye: the
audio-only fakes should sit apart, near 0.
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


def read_frames(path: str, max_frames: int, step: int, width: int) -> list[np.ndarray]:
    """Every step-th of the first max_frames frames, grey, shrunk to width pixels and blurred."""
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while i < max_frames and cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height = round(grey.shape[0] * width / grey.shape[1])
            frames.append(cv2.GaussianBlur(cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA),
                                           (5, 5), 0))
        i += 1
    cap.release()
    return frames


def compare_group(job: tuple[str, str, list[tuple[str, str]], argparse.Namespace]) -> list[dict]:
    """Compare every fake made from one original with that original (read once)."""
    source_id, source_path, fakes, args = job
    cv2.setNumThreads(1)
    try:
        original = read_frames(source_path, args.max_frames, args.step, args.width)
    except Exception as e:  # noqa: BLE001 - one bad file should not stop the run
        return [dict(video_id=v, source_id=source_id, status="error", message=f"original: {e}") for v, _ in fakes]
    rows = []
    for video_id, path in fakes:
        try:
            fake = read_frames(path, args.max_frames, args.step, args.width)
            n = min(len(fake), len(original))
            if n == 0:
                rows.append(dict(video_id=video_id, source_id=source_id, status="error", message="no frames"))
                continue
            changed, absdiff = [], []
            for a, b in zip(fake[:n], original[:n]):
                if a.shape != b.shape:
                    a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
                d = cv2.absdiff(a, b)
                changed.append(float((d > args.pixel_threshold).mean()))
                absdiff.append(float(d.mean()))
            rows.append(dict(video_id=video_id, source_id=source_id, status="compared", frames=n,
                             changed_max=max(changed), changed_median=float(np.median(changed)),
                             absdiff_median=float(np.median(absdiff))))
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
    parser.add_argument("--step", type=int, default=10, help="compare every N-th frame")
    parser.add_argument("--width", type=int, default=480, help="frames are shrunk to this width first")
    parser.add_argument("--pixel-threshold", type=int, default=25, help="grey levels for a pixel to count as changed")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="flag a fake when its most changed frame has under this share of pixels changed")
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
    out["audio_only"] = (out["status"] == "compared") & (out["changed_max"] < args.threshold)
    path = args.out or data_root / "processed" / "dfdc_audio_only.csv"
    out.to_csv(path, index=False)

    print("\n" + out["status"].value_counts().to_string())
    for _, r in out[out["status"] == "error"].iterrows():
        print(f"error {r['video_id']}: {r['message']}")
    compared = out[out["status"] == "compared"]
    if len(compared):
        print("\nchanged_max (share of pixels changed in the most changed frame), compared fakes:")
        print(compared["changed_max"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5]).to_string())
        for t in (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01):
            print(f"  under {t:<6}: {(compared['changed_max'] < t).sum()}")
        print("\nlowest 15:")
        print(compared.nsmallest(15, "changed_max")[["video_id", "source_id", "frames", "changed_max",
                                                     "changed_median", "absdiff_median"]].to_string(index=False))
    print(f"\n{int(out['audio_only'].sum())} fakes flagged audio-only (changed_max < {args.threshold}); wrote {path}")


if __name__ == "__main__":
    main()
