"""Metric definitions and threshold policy shared by validation and testing."""
import math
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def _checked_arrays(y_true, probabilities):
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise ValueError('Labels and probabilities must be matching nonempty vectors.')
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected binary labels and finite probabilities in [0,1].')
    return y.astype(int), p


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y,p = _checked_arrays(y_true, probabilities)
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0,1].')
    decision = (p >= threshold).astype(int)
    return {'threshold':float(threshold), 'sample_count':len(y),
            'precision':float(precision_score(y,decision,zero_division=0)),
            'recall':float(recall_score(y,decision,zero_division=0)),
            'f1':float(f1_score(y,decision,zero_division=0)),
            'accuracy':float(accuracy_score(y,decision)),
            'confusion_matrix':confusion_matrix(y,decision,labels=[0,1]).tolist(),
            'class_counts':{'0':int((y==0).sum()),'1':int((y==1).sum())}}


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y,p = _checked_arrays(y_true,probabilities)
    if set(y) != {0,1}:
        raise ValueError('Ranking evaluation requires both target classes.')
    return {**threshold_metrics(y,p,threshold),
            'average_precision':float(average_precision_score(y,p)),
            'roc_auc':float(roc_auc_score(y,p))}


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, policy: str) -> float:
    y,p = _checked_arrays(y_true,probabilities)
    if policy == 'fixed_0_5':
        return 0.5
    if policy != 'max_f1':
        raise ValueError('Unknown threshold policy.')
    candidates = np.unique(np.r_[0.0,p,1.0])
    scores = []
    for threshold in candidates:
        decision = p >= threshold
        denominator = int(y.sum() + decision.sum())
        score = 2 * int(((y==1)&decision).sum()) / denominator if denominator else 0.0
        scores.append((score,float(threshold)))
    # Tuple ordering implements the documented higher-threshold tie break.
    return max(scores)[1]
