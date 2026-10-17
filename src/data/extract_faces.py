"""
Extracts face crops from every video in the index.

For each video, takes --windows runs of --window-len consecutive frames, spread
evenly through the video. The temporal models need consecutive frames; the
frame models train on the same crops. Faces are detected with RetinaFace
(ResNet-50, via batch-face) on a whole window at a time, one face is followed
through the window, and a square crop around it is saved.

Run on the GPU machine from the repo root, after build_index, inside tmux:
    python -m src.data.extract_faces --data-root ~/data
Quick test on a few random videos first:
    python -m src.data.extract_faces --data-root ~/data --limit 20

Writes:
    {data_root}/faces/<video_id>/<frame>.jpg  one crop per frame, e.g. faces/ffpp/real/000/00012.jpg
    {data_root}/faces/<video_id>/faces.csv    one row per frame; written last, marks the video done
    {data_root}/processed/faces.csv           all done videos' rows, joined with label and split

Safe to stop (Ctrl+C) and rerun: videos that already have faces.csv are skipped.
"""

import argparse
import multiprocessing
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# RetinaFace's 5 landmarks, in its output order (the person's right eye comes first)
LANDMARKS = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
DETECT_MAX_SIDE = 960  # frames are shrunk to this for detection only; crops come from the full frame
MIN_IOU = 0.2  # a face in the next frame must overlap the followed face this much to count as the same person


def window_starts(n_frames: int, n_windows: int, window_len: int) -> list[int]:
    """First frame of each window. The video is cut into n equal parts with a
    window in the middle of each, so windows never overlap; short videos get
    fewer windows (one window covering everything if shorter than window_len)."""
    if n_frames <= window_len:
        return [0]
    n = max(1, min(n_windows, n_frames // window_len))
    part = n_frames / n
    return [min(max(0, round((i + 0.5) * part - window_len / 2)), n_frames - window_len)
            for i in range(n)]


def count_frames(path: str) -> int:
    """Frame count from the header, or by decoding the whole video if the header has none."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n


def read_windows(path: str, starts: list[int], window_len: int):
    """Yields (window, [(frame number, frame), ...]) for each window. Decodes from
    the start instead of seeking, because seeking in H.264 is not always frame-exact.
    A window is shorter (or missing) if the video ends earlier than its header says."""
    cap = cv2.VideoCapture(path)
    i = 0
    for w, start in enumerate(starts):
        frames = []
        while i < start + window_len and cap.grab():
            if i >= start:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append((i, frame))
            i += 1
        if frames:
            yield w, frames
    cap.release()


class FaceDetector:
    """RetinaFace (ResNet-50) from batch-face, run on batches of same-size frames."""

    def __init__(self, device: str, score_threshold: float, batch_size: int):
        from batch_face import RetinaFace  # imports torch; only needed in the workers
        self.net = RetinaFace(gpu_id=-1 if device == "cpu" else 0, network="resnet50")
        self.score_threshold, self.batch_size = score_threshold, batch_size

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Faces in each frame as rows of [x, y, w, h, 5 landmark (x, y) pairs, score],
        in frame pixels. Frames must be BGR, as OpenCV reads them."""
        h, w = frames[0].shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        if scale < 1:
            size = (round(w * scale), round(h * scale))
            frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
        results = self.net(frames, threshold=self.score_threshold, chunk_size=self.batch_size)
        out = []
        for faces in results:  # each face: box (x1, y1, x2, y2), landmarks (5, 2), score
            rows = [[box[0], box[1], box[2] - box[0], box[3] - box[1], *kps.ravel(), score]
                    for box, kps, score in faces]
            faces = np.array(rows, np.float32).reshape(-1, 15)
            faces[:, :14] /= scale
            out.append(faces)
        return out


class Landmarks68:
    """The standard 68 face landmarks (iBUG layout: jaw 0-16, brows 17-26, nose 27-35,
    eyes 36-47, mouth 48-67) from FAN (face-alignment), for a face RetinaFace found."""

    def __init__(self, device: str):
        import face_alignment
        # FAN's own detector is never run (we pass boxes in); choosing "retinaface" tells it
        # how to size its crop from a RetinaFace-style box.
        self.net = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device,
                                                flip_input=False, face_detector="retinaface",
                                                compile=False)

    def __call__(self, frame: np.ndarray, face: np.ndarray) -> tuple[np.ndarray, float]:
        """(68, 2) points in frame pixels, and their mean confidence (0-1). Frame is BGR."""
        x, y, w, h = face[:4]
        # float64: FAN puts values derived from the box into torch tensors, which reject np.float32
        box = np.array([x, y, x + w, y + h, face[14]], np.float64)
        points, scores, _ = self.net.get_landmarks_from_image(
            np.ascontiguousarray(frame[..., ::-1]), detected_faces=[box], return_landmark_score=True)
        return points[0], float(np.mean(scores[0]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two (x, y, w, h) boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def pick_face(faces: np.ndarray, followed: np.ndarray | None) -> np.ndarray | None:
    """The largest face at the start of a window; after that, the face that
    overlaps most with the last one kept, so the crop stays on the same person.
    None if no face, or none overlaps enough (frame is recorded without a crop)."""
    if len(faces) == 0:
        return None
    if followed is None:
        return faces[np.argmax(faces[:, 2] * faces[:, 3])]
    overlaps = [iou(f[:4], followed[:4]) for f in faces]
    best = int(np.argmax(overlaps))
    return faces[best] if overlaps[best] >= MIN_IOU else None


def square_crop(frame: np.ndarray, face: np.ndarray, margin: float, size: int):
    """Square crop centred on the face box, enlarged by `margin`, resized to size x size.
    Parts outside the frame are filled black. Returns the crop and its (x, y, side) in the frame."""
    x, y, w, h = face[:4]
    side = round(max(w, h) * (1 + margin))
    x0, y0 = round(x + w / 2 - side / 2), round(y + h / 2 - side / 2)
    fh, fw = frame.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - fw, y0 + side - fh)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    crop = frame[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interp), (x0, y0, side)


_args = None
_detector = None
_landmarks68 = None


def init_worker(args: argparse.Namespace) -> None:
    global _args, _detector, _landmarks68
    import torch
    # one process per CPU already; extra threads only compete
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _args = args
    _detector = FaceDetector(args.device, args.score_threshold, args.batch_size)
    _landmarks68 = Landmarks68(args.device)


def extract_video(job: tuple[str, str]) -> dict:
    """Crops faces from one video. Never raises, so one bad file cannot stop the run."""
    video_id, path = job
    try:
        out_dir = _args.faces_dir / video_id
        if (out_dir / "faces.csv").exists():
            return dict(video_id=video_id, status="skipped")
        n_frames = count_frames(path)
        if n_frames == 0:
            return dict(video_id=video_id, status="error", message="no frames could be read")
        out_dir.mkdir(parents=True, exist_ok=True)

        rows, points68 = [], []
        for window, frames in read_windows(path, window_starts(n_frames, _args.windows, _args.window_len),
                                           _args.window_len):
            detections = _detector([frame for _, frame in frames])
            followed = None  # each window starts again from the largest face
            for (f, frame), faces in zip(frames, detections):
                face = pick_face(faces, followed)
                row = dict(video_id=video_id, window=window, frame=f, crop="")
                points = np.full((68, 2), np.nan, np.float32)
                if face is not None:
                    followed = face
                    crop, (cx, cy, side) = square_crop(frame, face, _args.margin, _args.size)
                    name = f"{f:05d}.jpg"
                    cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, _args.jpeg_quality])
                    to_crop = _args.size / side
                    row.update(crop=f"{video_id}/{name}", score=round(float(face[14]), 3),
                               crop_x=cx, crop_y=cy, crop_side=side)
                    for k, point in enumerate(LANDMARKS):
                        row[f"{point}_x"] = round(float(face[4 + 2 * k] - cx) * to_crop, 1)
                        row[f"{point}_y"] = round(float(face[5 + 2 * k] - cy) * to_crop, 1)
                    frame_points, row["lm68_score"] = _landmarks68(frame, face)
                    points = (frame_points - [cx, cy]) * to_crop
                rows.append(row)
                points68.append(points)

        if not rows:
            return dict(video_id=video_id, status="error", message="no frames could be decoded")
        # 68-point landmarks in crop pixels, one (68, 2) entry per faces.csv row (NaN if no face)
        np.save(out_dir / "landmarks68.npy", np.stack(points68).astype(np.float32))
        # Write to a temporary name first, so a half-written file never marks the video done.
        tmp = out_dir / "faces.csv.tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, out_dir / "faces.csv")
        n_faces = sum(bool(r["crop"]) for r in rows)
        return dict(video_id=video_id, status="done", frames=len(rows), faces=n_faces)
    except Exception as e:
        return dict(video_id=video_id, status="error", message=f"{type(e).__name__}: {e}")


def merge(index: pd.DataFrame, faces_dir: Path, out: Path) -> None:
    """Concatenates every finished video's faces.csv and adds its label and split."""
    parts = [pd.read_csv(p, dtype={"crop": str})
             for v in index["video_id"] if (p := faces_dir / v / "faces.csv").exists()]
    if not parts:
        print("no finished videos to merge yet")
        return
    faces = pd.concat(parts, ignore_index=True)
    info = index[["video_id", "dataset", "method", "label", "split"]]
    faces = faces.merge(info, on="video_id", how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    faces.to_csv(out, index=False)
    print(f"wrote {len(faces)} frames from {len(parts)}/{len(index)} videos to {out}")
    has_face = faces["crop"].notna()  # empty crop = no face found in that frame
    print(f"frames with a face: {has_face.mean():.1%}")
    print(has_face.groupby([faces["dataset"], faces["method"]]).mean().map("{:.1%}".format).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--index", type=Path, default=None, help="default: {data_root}/processed/index.csv")
    parser.add_argument("--windows", type=int, default=2, help="windows of consecutive frames per video")
    parser.add_argument("--window-len", type=int, default=32, help="frames per window")
    parser.add_argument("--size", type=int, default=380, help="crop size in pixels (square)")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="extra space around the face box, as a share of its longer side")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="minimum RetinaFace face score")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16, help="frames per detector batch")
    parser.add_argument("--workers", type=int, default=4,
                        help="processes decoding videos; each loads its own detector on the GPU")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process this many random videos (fixed seed), for test runs")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    args.faces_dir = data_root / "faces"
    index = pd.read_csv(args.index or data_root / "processed" / "index.csv", dtype={"source_ids": str})
    todo = index.sample(n=min(args.limit, len(index)), random_state=0) if args.limit else index
    jobs = list(zip(todo["video_id"], todo["path"]))

    # Download the model weights once here (on the CPU), so the workers find them cached.
    FaceDetector("cpu", args.score_threshold, args.batch_size)
    Landmarks68("cpu")

    print(f"{len(jobs)} videos, {args.workers} workers on {args.device}, writing to {args.faces_dir}")
    start, counts, errors = time.time(), {"done": 0, "skipped": 0, "error": 0}, []
    frames = faces = 0
    # spawn, not fork: CUDA cannot be used in a forked child
    with multiprocessing.get_context("spawn").Pool(args.workers, initializer=init_worker,
                                                   initargs=(args,)) as pool:
        for i, r in enumerate(pool.imap_unordered(extract_video, jobs), 1):
            counts[r["status"]] += 1
            if r["status"] == "error":
                errors.append(r["video_id"])
                print(f"error {r['video_id']}: {r['message']}", flush=True)
            elif r["status"] == "done":
                frames, faces = frames + r["frames"], faces + r["faces"]
            if i % 50 == 0 or i == len(jobs):
                rate = (time.time() - start) / max(1, counts["done"] + counts["error"])
                left = (len(jobs) - i) * rate / 3600
                print(f"{i}/{len(jobs)} videos | {counts} | faces in {faces}/{frames} new frames "
                      f"| {rate:.1f} s/video, ~{left:.1f} h left", flush=True)

    if errors:
        print(f"{len(errors)} videos failed and will be retried on the next run: {errors[:10]}")
    merge(index, args.faces_dir, data_root / "processed" / "faces.csv")


if __name__ == "__main__":
    main()
