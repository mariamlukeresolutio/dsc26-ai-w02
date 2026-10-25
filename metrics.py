from __future__ import annotations
import math
import numpy as np

def _validate(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty.")

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    if np.isnan(scores).any():
        raise ValueError("Scores cannot contain NaN values.")

    return y_true, scores


def confusion_at(y_true, scores, threshold: float) -> dict:
    y_true, scores = _validate(y_true, scores)

    preds = scores >= threshold

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    cm = confusion_at(y_true, scores, threshold)

    tp = cm["tp"]
    fp = cm["fp"]
    fn = cm["fn"]

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)

    return precision, recall


def precision_at_k(y_true, scores, k: int) -> float:
    y_true, scores = _validate(y_true, scores)

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    n = len(y_true)
    k = min(k, n)

    order = np.argsort(-scores, kind="stable")
    top_k = y_true[order[:k]]

    return float(np.sum(top_k) / k)


def expected_cost(
    y_true,
    scores,
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    cm = confusion_at(y_true, scores, threshold)

    return (
        cm["fp"] * cost_fp
        + cm["fn"] * cost_fn
    )


def best_threshold(
    y_true,
    scores,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    y_true, scores = _validate(y_true, scores)

    candidates = list(np.unique(scores))
    candidates.append(math.inf)

    best_thr = None
    best_cost = math.inf

    for thr in candidates:
        cost = expected_cost(
            y_true,
            scores,
            thr,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        if cost < best_cost:
            best_cost = cost
            best_thr = thr
        elif cost == best_cost and thr > best_thr:
            best_thr = thr

    return float(best_thr)


def threshold_for_precision(
    y_true,
    scores,
    min_precision: float,
) -> float | None:
    y_true, scores = _validate(y_true, scores)

    valid = []

    for thr in np.unique(scores):
        precision, _ = precision_recall_at(
            y_true,
            scores,
            thr,
        )

        if precision >= min_precision:
            valid.append(thr)

    if not valid:
        return None

    return float(min(valid))