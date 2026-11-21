import numpy as np
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix)

def expected_calibration_error(y_true, proba, n_bins=10):
    """Average gap between predicted probability and actual disease rate."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(proba, edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return ece

def all_metrics(y_true, proba, threshold=0.5):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    both_classes = len(np.unique(y_true)) > 1
    return {
        "auc": roc_auc_score(y_true, proba) if both_classes else np.nan,
        "bal_acc": balanced_accuracy_score(y_true, pred) if both_classes else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "brier": brier_score_loss(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }