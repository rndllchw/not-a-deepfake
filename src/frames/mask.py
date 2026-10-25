"""
Face mask: keeps the face and blacks out everything around it.

This is the "mask" input setting compared against the plain crop ("no-mask")
for the baseline frame models. A run trained with --mask sees masked crops in
training, validation and testing; predict reads the setting from the run, so a
model is always scored the way it was trained.

The face region is the outline of the 68 landmarks, extended up over the
forehead (the landmarks stop at the eyebrows) and grown by GROW of the face
size, so the edge of a swapped face, where blending artefacts are, stays
visible. Hair, neck and background are removed.

Preview what the mask keeps, on a few random crops:
    python -m src.frames.mask --data-root ~/data
writes {data_root}/previews/mask/*.jpg: crop | crop with 68 landmarks | masked crop
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

GROW = 0.08  # share of the face size the mask is grown by, to keep the face's edge
FOREHEAD = 0.35  # how far above the eyebrows the mask reaches, as a share of the brow-to-chin distance
BROWS = slice(17, 27)  # iBUG 68-point indices
NOSE_TOP, CHIN = 27, 8


def face_region(shape: tuple, points: np.ndarray) -> np.ndarray:
    """(H, W) uint8 mask, 1 on the face: convex hull of the 68 landmarks plus a
    forehead band, grown by GROW. Follows head rotation, since the forehead is
    placed along the chin-to-nose direction."""
    up = points[NOSE_TOP] - points[CHIN]  # points from chin to top of nose
    forehead = points[BROWS] + FOREHEAD * up
    outline = cv2.convexHull(np.concatenate([points, forehead]).round().astype(np.int32))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, outline, 1)
    face_size = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    k = 2 * max(1, round(GROW * face_size / 2)) + 1
    return cv2.dilate(mask, np.ones((k, k), np.uint8))


def apply_mask(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The crop with everything outside the face region set to black."""
    return img * face_region(img.shape, points)[..., None]


def main() -> None:
    from src.frames.dataset import load_frames  # here, not at the top: dataset imports this module

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=Path, default=Path("~/data"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=12, help="number of crops to preview")
    parser.add_argument("--out", type=Path, default=None, help="default: {data_root}/previews/mask")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    out = args.out or data_root / "previews" / "mask"
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(data_root, args.split)
    for _, row in frames.sample(n=min(args.n, len(frames)), random_state=0).iterrows():
        img = cv2.imread(str(data_root / "faces" / row["crop"]))
        points = np.load(data_root / "faces" / row["video_id"] / "landmarks68.npy")[row["lm_row"]]
        marked = img.copy()
        for x, y in points.round().astype(int):
            cv2.circle(marked, (x, y), 2, (0, 255, 0), -1)
        name = row["crop"].replace("/", "_")
        cv2.imwrite(str(out / name), np.hstack([img, marked, apply_mask(img, points)]))
    print(f"wrote {min(args.n, len(frames))} previews to {out}")


if __name__ == "__main__":
    main()
