"""
Evaluation metrics (Step 7), shared by every model.

Video scores are the mean of the video's frame scores; the temporal models
(Step 4) replace that mean with their own window scores.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def binary_metrics(labels, probs, threshold: float = 0.5, max_fpr: float = 0.01) -> dict:
    """AUC, accuracy and F1 at `threshold`, and the share of fakes caught when at
    most `max_fpr` of real faces are flagged (TPR at low FPR). label 1 = fake."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return dict(n=len(labels))  # AUC needs both real and fake
    fpr, tpr, _ = roc_curve(labels, probs)
    pred = probs >= threshold
    return dict(n=len(labels), auc=float(roc_auc_score(labels, probs)), acc=float(accuracy_score(labels, pred)),
                f1=float(f1_score(labels, pred)), tpr_at_fpr=float(tpr[np.searchsorted(fpr, max_fpr, side="right") - 1]))


def video_scores(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per video: its label, dataset, method and mean frame `prob`."""
    return frames.groupby("video_id").agg(label=("label", "first"), dataset=("dataset", "first"),
                                          method=("method", "first"), prob=("prob", "mean"))


def frame_and_video_metrics(frames: pd.DataFrame) -> dict:
    """binary_metrics over frames and over videos, keys prefixed frame_ and video_.
    `frames` needs video_id, label, dataset, method and prob columns."""
    videos = video_scores(frames)
    out = {f"frame_{k}": v for k, v in binary_metrics(frames["label"], frames["prob"]).items()}
    out.update({f"video_{k}": v for k, v in binary_metrics(videos["label"], videos["prob"]).items()})
    return out
