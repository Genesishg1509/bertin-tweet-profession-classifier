"""Evaluation utilities: metrics and threshold tuning.

The original coursework notebook found the F1-optimal decision threshold on
the validation set but never applied it when generating the final test
predictions -- it used the default argmax (equivalent to threshold 0.5)
instead, silently discarding the improvement it had just measured. This
module fixes that: `find_best_threshold` tunes the threshold, and
`predict_with_threshold` is meant to be applied consistently afterwards,
including on the final test predictions.

Caveat, stated plainly: the threshold is tuned on the same validation set
used to report F1, so the reported number is optimistic. It is the best
threshold FOR this ~1,000-example validation set, not a guarantee on new
data. A more rigorous setup would tune the threshold on a separate split.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def find_best_threshold(y_true, probs, thresholds=None) -> tuple[float, float]:
    """Grid-search the F1-optimal probability threshold.

    Run only on validation data -- running this on test labels would leak
    them into model selection.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        y_pred = (np.asarray(probs) >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="binary")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def predict_with_threshold(probs, threshold: float) -> np.ndarray:
    """Apply a fixed threshold to class-1 probabilities."""
    return (np.asarray(probs) >= threshold).astype(int)


def evaluation_report(y_true, y_pred) -> dict:
    """F1, accuracy, and confusion matrix as a plain dict."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "confusion_matrix": cm.tolist(),
    }
