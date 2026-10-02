"""Cross-validated regression and a two-view blend.

View 1 is the rubric features (a tree or a linear model).
View 2 is a regularized regression on text embeddings, which picks up
vocabulary and clause patterns the hand-built features miss.

The blend weight is chosen without looking at the fold being scored.
That keeps the cross-validated Pearson and RMSE honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from src.config import N_SPLITS, SCORE_MAX, SCORE_MIN, SEED
from src.data import speaker_groups
from src.metrics import evaluate, objective


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop = {"audio_id"}
    columns = [c for c in frame.columns if c not in drop]
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    keep = []
    for col in matrix.columns:
        values = matrix[col]
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return matrix[keep], keep


def make_folds(y: np.ndarray, audio_ids: list[str], n_splits: int = N_SPLITS):
    groups = speaker_groups(audio_ids)
    if groups is not None:
        n_groups = len(set(groups))
        splits = min(n_splits, n_groups)
        folder = GroupKFold(n_splits=splits)
        return list(folder.split(np.zeros(len(y)), y, groups)), "group"

    # Likert scores are discrete. Stratifying on the rounded score keeps every
    # fold in the same score range, which stabilises Pearson on a small set.
    rounded = np.rint(np.asarray(y)).astype(int)
    counts = pd.Series(rounded).value_counts()
    if counts.size >= 2 and int(counts.min()) >= n_splits:
        folder = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(folder.split(np.zeros(len(y)), rounded)), "stratified"
    folder = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(folder.split(np.zeros(len(y)))), "kfold"


def _hgb(**kwargs) -> HistGradientBoostingRegressor:
    params = dict(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_depth=3,
        min_samples_leaf=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=SEED,
    )
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def _fit_predict_tabular(name: str, x_train, y_train, x_valid):
    if name == "ridge":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 4, 15))),
        ])
    elif name == "extra_trees":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=5,
                random_state=SEED,
                n_jobs=-1,
            )),
        ])
    elif name == "hgb_shallow":
        model = _hgb(max_depth=2, min_samples_leaf=20, learning_rate=0.05, l2_regularization=1.0)
    elif name == "hgb":
        model = _hgb()
    elif name == "hgb_deep":
        model = _hgb(max_depth=4, min_samples_leaf=10, learning_rate=0.03, l2_regularization=2.0)
    else:
        raise KeyError(name)
    model.fit(x_train, y_train)
    return model.predict(x_valid), model


def _fit_predict_text(x_train, y_train, x_valid):
    n_comp = int(min(48, x_train.shape[0] - 1, x_train.shape[1]))
    n_comp = max(2, n_comp)
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
    z_train = svd.fit_transform(x_train)
    z_valid = svd.transform(x_valid)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-2, 4, 15))),
    ])
    model.fit(z_train, y_train)
    return model.predict(z_valid), (svd, model)


def _baseline_oof(y, folds) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in folds:
        pred[valid_idx] = float(np.mean(y[train_idx]))
    return pred


def cross_validate(x: pd.DataFrame, y: np.ndarray, emb: np.ndarray, audio_ids: list[str]):
    folds, fold_kind = make_folds(y, audio_ids)
    print(f"Cross-validation: {len(folds)} {fold_kind} folds.")
    names = ["ridge", "extra_trees", "hgb_shallow", "hgb", "hgb_deep"]
    oof = {name: np.zeros(len(y), dtype=float) for name in names}
    oof["text"] = np.zeros(len(y), dtype=float)
    x_values = x.to_numpy(dtype=float)
    text_ok = np.nanstd(emb) > 1e-6

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"  fold {fold_id}/{len(folds)}")
        for name in names:
            pred, _ = _fit_predict_tabular(name, x_values[train_idx], y[train_idx], x_values[valid_idx])
            oof[name][valid_idx] = pred
        if text_ok:
            text_pred, _ = _fit_predict_text(emb[train_idx], y[train_idx], emb[valid_idx])
            oof["text"][valid_idx] = text_pred
        else:
            oof["text"][valid_idx] = float(np.mean(y[train_idx]))

    oof["mean_baseline"] = _baseline_oof(y, folds)
    scores = {name: evaluate(y, pred) for name, pred in oof.items()}
    tabular_name = max(names, key=lambda name: objective(y, oof[name]))
    blended, weight = _honest_blend(oof[tabular_name], oof["text"], y, folds)
    calibrated, slope, intercept = _honest_calibrate(blended, y, folds)
    use_calibration = objective(y, calibrated) >= objective(y, blended)
    chosen = calibrated if use_calibration else blended
    scores["blend"] = evaluate(y, blended)
    scores["blend_calibrated"] = evaluate(y, calibrated)
    return {
        "oof": oof,
        "oof_blend": blended,
        "oof_final": chosen,
        "scores": scores,
        "tabular_name": tabular_name,
        "blend_weight_tabular": weight,
        "calibration": {"slope": slope, "intercept": intercept, "used": use_calibration},
        "folds": folds,
        "fold_kind": fold_kind,
        "columns": list(x.columns),
    }


def _best_weight(first, second, y) -> float:
    best_w, best_score = 1.0, -1e9
    for weight in np.linspace(0.0, 1.0, 21):
        pred = weight * first + (1.0 - weight) * second
        score = objective(y, pred)
        if score > best_score:
            best_score = score
            best_w = float(weight)
    return best_w


def _honest_blend(tabular, text, y, folds):
    """Pick the blend weight on the other folds, then score the held-out fold."""
    blended = np.zeros(len(y), dtype=float)
    weights = []
    for _, valid_idx in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[valid_idx] = False
        weight = _best_weight(tabular[mask], text[mask], y[mask])
        blended[valid_idx] = weight * tabular[valid_idx] + (1.0 - weight) * text[valid_idx]
        weights.append(weight)
    return blended, float(np.median(weights))


def _fit_ab(pred, y) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(pred) < 1e-6:
        return 1.0, 0.0
    design = np.vstack([pred, np.ones_like(pred)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    if slope <= 0:
        return 1.0, 0.0
    return float(slope), float(intercept)


def _honest_calibrate(pred, y, folds):
    calibrated = np.zeros(len(y), dtype=float)
    slopes, intercepts = [], []
    for _, valid_idx in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[valid_idx] = False
        slope, intercept = _fit_ab(pred[mask], y[mask])
        calibrated[valid_idx] = slope * pred[valid_idx] + intercept
        slopes.append(slope)
        intercepts.append(intercept)
    return (
        np.clip(calibrated, SCORE_MIN, SCORE_MAX),
        float(np.mean(slopes)),
        float(np.mean(intercepts)),
    )


def fit_full(x: pd.DataFrame, y: np.ndarray, emb: np.ndarray, test_x: pd.DataFrame, test_emb: np.ndarray, cv: dict):
    """Refit on every training clip and score both the training set and the test set.

    Training-set error is optimistic. It is still required in the submission
    notebook. Cross-validated numbers in `cv` are the ones to trust.
    """
    name = cv["tabular_name"]
    weight = cv["blend_weight_tabular"]
    columns = cv["columns"]
    x_values = x[columns].to_numpy(dtype=float)
    test_values = test_x[columns].to_numpy(dtype=float)

    train_tab, tab_model = _fit_predict_tabular(name, x_values, y, x_values)
    test_tab = tab_model.predict(test_values)

    if np.nanstd(emb) > 1e-6:
        train_text, (svd, text_model) = _fit_predict_text(emb, y, emb)
        test_text = text_model.predict(svd.transform(test_emb))
    else:
        train_text = np.full(len(y), float(np.mean(y)))
        test_text = np.full(len(test_values), float(np.mean(y)))

    train_pred = weight * train_tab + (1.0 - weight) * train_text
    test_pred = weight * test_tab + (1.0 - weight) * test_text
    if cv["calibration"]["used"]:
        slope = cv["calibration"]["slope"]
        intercept = cv["calibration"]["intercept"]
        train_pred = slope * train_pred + intercept
        test_pred = slope * test_pred + intercept
    train_pred = np.clip(train_pred, SCORE_MIN, SCORE_MAX)
    test_pred = np.clip(test_pred, SCORE_MIN, SCORE_MAX)
    return train_pred, test_pred, tab_model


def permutation_importance_table(model, x: pd.DataFrame, y: np.ndarray, folds) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    train_idx, valid_idx = folds[-1]
    values = x.to_numpy(dtype=float)
    # Refit on one split so importance is measured on clips the model did not see.
    if hasattr(model, "fit"):
        # `model` was fit on all of the data. Fit a fresh clone on the train fold.
        from sklearn.base import clone
        estimator = clone(model)
        estimator.fit(values[train_idx], y[train_idx])
    else:
        estimator = model
    result = permutation_importance(
        estimator,
        values[valid_idx],
        y[valid_idx],
        n_repeats=8,
        random_state=SEED,
        scoring="neg_root_mean_squared_error",
    )
    table = pd.DataFrame({
        "feature": list(x.columns),
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance", ascending=False)
    return table
