"""
Video-call quality degradations for face crops.

Calls are compressed hard, sent at low resolution, and often blurry or badly
lit, which removes the fine artefacts many detectors rely on. Training on
degraded crops stops a model depending on those artefacts.

The call-quality test sets draw the settings once per window (see dataset.py),
so they hold steady over its consecutive frames, as a real call's do; only the
sensor noise changes from frame to frame. Settings that changed every frame
would add flicker that no real call has, which a temporal model could pick up.

All functions take and return BGR uint8 images, as OpenCV reads them.
"""

import cv2
import numpy as np


def lighting(img: np.ndarray, gamma: float, contrast: float, brightness: float) -> np.ndarray:
    """Too dark or too bright, with lower contrast, as from a poor webcam."""
    x = img.astype(np.float32) / 255
    x = x ** gamma  # gamma above 1 darkens
    x = (x - 0.5) * contrast + 0.5 + brightness
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Camera sensor noise: its strength is a setting, the noise itself is new every frame."""
    noisy = img + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Out of focus, or smeared by movement."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def low_resolution(img: np.ndarray, side: int) -> np.ndarray:
    """Shrunk to `side` px wide and enlarged again, as when a small call tile is shown full size."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (side, round(side * h / w)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def compression(img: np.ndarray, quality: int) -> np.ndarray:
    """Strong JPEG compression. Its 8x8 block artefacts are close to those of a
    low-bitrate video codec, and it is much faster to apply per frame."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def call_settings(rng: np.random.Generator) -> dict:
    """One call's conditions: lighting, noise, blur and low resolution are each on
    with probability 0.5, each with its own strength; compression is always on."""
    settings = {}
    if rng.random() < 0.5:
        settings["lighting"] = dict(gamma=rng.uniform(0.6, 1.6), contrast=rng.uniform(0.6, 1.1),
                                    brightness=rng.uniform(-0.15, 0.1))
    if rng.random() < 0.5:
        settings["noise"] = dict(sigma=rng.uniform(2, 8))
    if rng.random() < 0.5:
        settings["blur"] = dict(sigma=rng.uniform(0.5, 2.0))
    if rng.random() < 0.5:
        settings["low_resolution"] = dict(side=int(rng.integers(64, 161)))
    settings["compression"] = dict(quality=int(rng.integers(20, 61)))
    return settings


def apply_call(img: np.ndarray, settings: dict, rng: np.random.Generator) -> np.ndarray:
    """Degrades one frame with call_settings, in the order camera then sender.
    rng only draws the sensor noise."""
    if "lighting" in settings:
        img = lighting(img, **settings["lighting"])
    if "noise" in settings:
        img = noise(img, settings["noise"]["sigma"], rng)
    if "blur" in settings:
        img = blur(img, **settings["blur"])
    if "low_resolution" in settings:
        img = low_resolution(img, **settings["low_resolution"])
    return compression(img, **settings["compression"])


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Call quality for a crop on its own, with settings of its own (training augmentation)."""
    return apply_call(img, call_settings(rng), rng)
