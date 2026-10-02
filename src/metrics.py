"""Leaderboard metrics: RMSE and Pearson correlation."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from src.config import SCORE_MAX, SCORE_MIN


def clip_score(pred) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), SCORE_MIN, SCORE_MAX)


def rmse(y_true, y_pred) -> float:
    pred = clip_score(y_pred)
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, dtype=float), pred)))


def pearson(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = clip_score(y_pred)
    if np.std(y) < 1e-8 or np.std(pred) < 1e-8:
        return 0.0
    value = pearsonr(y, pred)[0]
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def evaluate(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "pearson": pearson(y_true, y_pred)}


def objective(y_true, y_pred) -> float:
    """Higher is better.

    Pearson is roughly on [-1, 1]. RMSE on a 0–5 scale is divided by 5 so the
    two terms have a similar range. The leaderboard uses both, and neither
    one alone is a safe selection rule: a well-ranked but badly scaled model
    wins Pearson and loses RMSE.
    """
    scores = evaluate(y_true, y_pred)
    return scores["pearson"] - scores["rmse"] / 5.0
