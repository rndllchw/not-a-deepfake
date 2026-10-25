"""
Video-call quality degradations for face crops.

Calls are compressed hard, sent at low resolution, and often blurry or badly
lit, which removes the fine artefacts many detectors rely on. Training on
degraded crops stops a model depending on those artefacts; the same functions,
with a fixed seed per crop, make the call-quality test sets.

All functions take and return BGR uint8 images, as OpenCV reads them.
"""

import cv2
import numpy as np


def lighting(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Too dark or too bright, with lower contrast, as from a poor webcam."""
    x = img.astype(np.float32) / 255
    x = x ** rng.uniform(0.6, 1.6)  # gamma above 1 darkens
    x = (x - 0.5) * rng.uniform(0.6, 1.1) + 0.5 + rng.uniform(-0.15, 0.1)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Camera sensor noise."""
    noisy = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Out of focus, or smeared by movement."""
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))


def low_resolution(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shrunk to 64-160 px and enlarged again, as when a small call tile is shown full size."""
    h, w = img.shape[:2]
    side = int(rng.integers(64, 161))
    small = cv2.resize(img, (side, round(side * h / w)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def compression(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Strong JPEG compression (quality 20-60). Its 8x8 block artefacts are close to
    those of a low-bitrate video codec, and it is much faster to apply per frame."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 61))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Call quality: lighting, noise, blur and low resolution each applied with
    probability 0.5, in the order camera then sender, then always compressed."""
    for step in (lighting, noise, blur, low_resolution):
        if rng.random() < 0.5:
            img = step(img, rng)
    return compression(img, rng)
