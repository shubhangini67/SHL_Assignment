"""Figures the notebook shows next to the metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURE_DIR

NAVY = "#1f4e79"
TEAL = "#2a6f7f"


def _style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
    })


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def label_distribution(labels: pd.Series) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(labels, bins=20, color=NAVY, ax=ax, edgecolor="white")
    ax.set_xlabel("Grammar score")
    ax.set_ylabel("Clips")
    ax.set_title("Training label distribution")
    return _save(fig, "label_distribution.png")


def feature_correlations(corr: pd.Series) -> Path:
    _style()
    top = corr.reindex(corr.abs().sort_values(ascending=False).head(15).index)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [NAVY if v > 0 else "#b85c38" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Pearson r with grammar score")
    ax.set_title("Which measurements follow the score")
    return _save(fig, "feature_correlations.png")


def pred_vs_actual(y, pred, rmse: float, pearson: float, title: str, name: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, pred, alpha=0.55, s=22, color=NAVY, edgecolor="none")
    lo = min(float(np.min(y)), float(np.min(pred)), 0)
    hi = max(float(np.max(y)), float(np.max(pred)), 5)
    ax.plot([lo, hi], [lo, hi], color=TEAL, linewidth=1.2)
    ax.set_xlabel("Actual grammar score")
    ax.set_ylabel("Predicted grammar score")
    ax.set_title(title)
    ax.text(
        0.04, 0.96,
        f"RMSE {rmse:.3f}\nPearson r {pearson:.3f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#dddddd"},
    )
    return _save(fig, name)


def residuals(y, pred) -> Path:
    _style()
    resid = np.asarray(pred) - np.asarray(y)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(pred, resid, alpha=0.55, s=22, color=NAVY, edgecolor="none")
    ax.axhline(0, color=TEAL, linewidth=1.2)
    ax.set_xlabel("Cross-validated prediction")
    ax.set_ylabel("Prediction − actual")
    ax.set_title("5-fold residuals")
    return _save(fig, "residuals.png")


def importance(table: pd.DataFrame) -> Path:
    _style()
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance"], color=NAVY, xerr=top["importance_std"], ecolor="#9bb3c9")
    ax.set_xlabel("Permutation importance (RMSE increase)")
    ax.set_title("Which features the model uses")
    return _save(fig, "feature_importance.png")


def score_vs_feature(labels, values, xlabel: str, name: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(values, labels, alpha=0.5, s=22, color=NAVY, edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Grammar score")
    ax.set_title(xlabel + " vs grammar score")
    return _save(fig, name)
