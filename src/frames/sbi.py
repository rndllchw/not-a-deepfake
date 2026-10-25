"""
Self-Blended Images (Shiohara & Yamasaki, CVPR 2022).

Makes a training fake from one real face crop: a slightly altered copy of the
face (colour, sharpness, a small shift and scale) is blended back onto the
original inside a soft mask over the face. The result has the blending
boundary and colour mismatch that face swaps leave, but no change of identity,
so a model trained on it learns those general artefacts rather than the
fingerprints of one manipulation method.

Simplified from the official code: the mask is the convex hull of the 68
landmarks (the paper also extends it over the forehead), and the colour and
sharpness changes are a smaller set.
"""

import cv2
import numpy as np

BLEND_RATIOS = [0.25, 0.5, 0.75, 1, 1, 1]  # from the paper; full-strength blends are most common


def face_mask(shape: tuple, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soft mask (H, W) in 0-1 over the face: the landmarks' convex hull, grown or
    shrunk a little, then blurred so the blending edge is a gradient."""
    mask = np.zeros(shape[:2], np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(points.round().astype(np.int32)), 1.0)
    face_size = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    k = 2 * int(rng.integers(1, max(2, int(0.05 * face_size)))) + 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.dilate(mask, kernel) if rng.random() < 0.5 else cv2.erode(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.02, 0.08) * face_size)
    mask /= max(float(mask.max()), 1e-6)  # blurring lowers the centre; bring it back to 1
    return mask * rng.choice(BLEND_RATIOS)


def alter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A copy of the face with shifted colour, changed sharpness and a small
    shift and scale, so its edges and tones do not match the original exactly."""
    x = img.astype(np.float32) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-15, 15, 3)
    x = np.clip(x, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        x = blurred
    else:  # unsharp mask
        amount = rng.uniform(0.3, 1.0)
        x = cv2.addWeighted(x, 1 + amount, blurred, -amount, 0)
    h, w = img.shape[:2]
    scale = rng.uniform(0.95, 1.05)
    dx, dy = rng.uniform(-0.03, 0.03, 2) * (w, h)
    shift = np.float32([[scale, 0, (1 - scale) * w / 2 + dx], [0, scale, (1 - scale) * h / 2 + dy]])
    return cv2.warpAffine(x, shift, (w, h), borderMode=cv2.BORDER_REFLECT)


def self_blend(img: np.ndarray, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A fake made from the real crop `img` (BGR uint8) and its (68, 2) landmarks in
    crop pixels. As in the paper, either the pasted face or the background is the
    altered copy."""
    source, target = alter(img, rng), img
    if rng.random() < 0.5:
        source, target = target, source
    mask = face_mask(img.shape, points, rng)[..., None]
    return (mask * source + (1 - mask) * target).round().astype(np.uint8)
