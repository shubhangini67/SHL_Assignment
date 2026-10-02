"""End-to-end run: audio in, submission.csv and the required training RMSE out."""

from __future__ import annotations

import json
import os
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# The Xet download path currently 404s for some public models, including GPT-2.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, DATA_DIR, OUTPUT_DIR
from src.data import load_competition
from src.features import build_features
from src.metrics import evaluate
from src.model import cross_validate, feature_matrix, fit_full, permutation_importance_table
from src.plots import (
    feature_correlations,
    importance,
    label_distribution,
    pred_vs_actual,
    residuals,
    score_vs_feature,
)


def _align(frame: pd.DataFrame, features: pd.DataFrame, embeddings: np.ndarray):
    saved_ids = features["audio_id"].astype(str).tolist()
    position = {audio_id: i for i, audio_id in enumerate(saved_ids)}
    missing = [audio_id for audio_id in frame["audio_id"].astype(str) if audio_id not in position]
    if missing:
        raise RuntimeError(f"{len(missing)} ids are missing from the feature cache, for example {missing[:3]}")
    order = [position[audio_id] for audio_id in frame["audio_id"].astype(str)]
    ordered = features.set_index("audio_id").loc[frame["audio_id"].astype(str).tolist()].reset_index()
    return ordered, embeddings[order]


def _banner(train_scores: dict, cv_scores: dict) -> str:
    line = "=" * 64
    text = "\n".join([
        line,
        f"TRAINING RMSE (required): {train_scores['rmse']:.4f}",
        f"TRAINING Pearson r:       {train_scores['pearson']:.4f}",
        f"CV RMSE:                  {cv_scores['rmse']:.4f}",
        f"CV Pearson r:             {cv_scores['pearson']:.4f}",
        line,
        "Training RMSE is the in-sample error of the final model.",
        "CV RMSE and CV Pearson are the estimates of test-set performance.",
    ])
    print("\n" + text + "\n")
    return text


def run(data_dir: Path | None = None) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_competition(DATA_DIR if data_dir is None else data_dir)
    label_distribution(data.train["label"])

    train_features, test_features, train_emb, test_emb, previews = build_features(
        data.train, data.test, CACHE_DIR
    )
    train_features, train_emb = _align(data.train, train_features, train_emb)
    test_features, test_emb = _align(data.test, test_features, test_emb)

    matrix, columns = feature_matrix(train_features)
    test_matrix = test_features.reindex(columns=columns)
    labels = data.train["label"].to_numpy(dtype=float)

    correlations = matrix.corrwith(pd.Series(labels, index=matrix.index))
    correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    feature_correlations(correlations)
    correlations.to_csv(OUTPUT_DIR / "feature_correlations.csv", header=["pearson_r"])

    for feature, filename, xlabel in (
        ("gec_wer", "gec_wer_vs_score.png", "Grammar-correction edit rate"),
        ("cola_mean", "cola_vs_score.png", "Mean grammatical-acceptability probability"),
        ("fragment_rate", "fragment_rate_vs_score.png", "Incomplete-sentence rate"),
        ("clause_per_sentence", "clauses_vs_score.png", "Dependent clauses per sentence"),
    ):
        if feature in matrix.columns:
            score_vs_feature(labels, matrix[feature], xlabel, filename)

    print("\nFitting models with cross-validation.")
    cv = cross_validate(matrix, labels, train_emb, data.train["audio_id"].astype(str).tolist())
    print("\nOut-of-fold scores")
    for name, scores in cv["scores"].items():
        print(f"  {name:20s}  RMSE {scores['rmse']:.4f}   Pearson {scores['pearson']:.4f}")
    print(
        f"Selected tabular model: {cv['tabular_name']}  "
        f"blend weight on tabular features: {cv['blend_weight_tabular']:.2f}  "
        f"calibration used: {cv['calibration']['used']}"
    )

    train_pred, test_pred, model = fit_full(matrix, labels, train_emb, test_matrix, test_emb, cv)
    # sample_submission.csv names 99 clips that are not in the published audio.
    # A model has nothing to score there, so those rows get the training mean.
    has_audio = data.test["audio_path"].map(lambda p: isinstance(p, str) and Path(p).is_file())
    n_missing_audio = int((~has_audio).sum())
    if n_missing_audio:
        fallback = float(np.mean(labels))
        print(
            f"{n_missing_audio} submission files have no wav in the competition bundle. "
            f"Those scores are the training-set mean ({fallback:.4f})."
        )
        test_pred = np.asarray(test_pred, dtype=float).copy()
        test_pred[~has_audio.to_numpy()] = fallback
    train_scores = evaluate(labels, train_pred)
    cv_scores = evaluate(labels, cv["oof_final"])
    banner = _banner(train_scores, cv_scores)

    pred_vs_actual(
        labels, cv["oof_final"], cv_scores["rmse"], cv_scores["pearson"],
        "Cross-validated predictions", "pred_vs_actual_cv.png",
    )
    pred_vs_actual(
        labels, train_pred, train_scores["rmse"], train_scores["pearson"],
        "Training-set predictions (in-sample)", "pred_vs_actual_train.png",
    )
    residuals(labels, cv["oof_final"])

    try:
        importance_table = permutation_importance_table(model, matrix, labels, cv["folds"])
        importance_table.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        importance(importance_table)
    except Exception:
        print("Permutation importance failed. Correlation plot is still available.")
        traceback.print_exc()

    submission = data.sample_submission.copy()
    if len(test_pred) != len(submission):
        raise RuntimeError(
            f"Predicted {len(test_pred)} scores but sample_submission.csv has {len(submission)} rows."
        )
    submission[data.target_column] = np.round(test_pred, 5)
    _validate_submission(submission, data.sample_submission, data.id_column, data.target_column)
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False, float_format="%.5f")

    oof = pd.DataFrame({
        "audio_id": data.train["audio_id"],
        "label": labels,
        "oof_prediction": cv["oof_final"],
        "train_prediction": train_pred,
    })
    oof.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    train_preview = (previews or {}).get("train", {})
    examples = []
    ranked = oof.sort_values("label")
    for kind, subset in (("low", ranked.head(3)), ("high", ranked.tail(3))):
        for row in subset.itertuples(index=False):
            preview = train_preview.get(str(row.audio_id), {})
            examples.append({
                "band": kind,
                "audio_id": str(row.audio_id),
                "label": float(row.label),
                "oof_prediction": float(row.oof_prediction),
                "whisper": preview.get("whisper", ""),
                "grammar_transcript": preview.get("grammar", ""),
            })
    (OUTPUT_DIR / "examples.json").write_text(json.dumps(examples, indent=2))

    metrics = {
        "training_rmse": train_scores["rmse"],
        "training_pearson": train_scores["pearson"],
        "cv_rmse": cv_scores["rmse"],
        "cv_pearson": cv_scores["pearson"],
        "model_scores": cv["scores"],
        "tabular_model": cv["tabular_name"],
        "blend_weight_tabular": cv["blend_weight_tabular"],
        "calibration": cv["calibration"],
        "fold_kind": cv["fold_kind"],
        "n_train": int(len(labels)),
        "n_test": int(len(test_pred)),
        "n_missing_audio": n_missing_audio,
        "n_features": int(matrix.shape[1]),
        "target_column": data.target_column,
        "id_column": data.id_column,
        "banner": banner,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _stamp_notebook(metrics)
    print(f"Wrote {submission_path}")
    return metrics


def _validate_submission(submission: pd.DataFrame, sample: pd.DataFrame, id_column: str, target_column: str) -> None:
    """Refuse a file Kaggle would reject or score against the wrong rows."""
    if list(submission.columns) != list(sample.columns):
        raise RuntimeError(
            f"Submission columns {list(submission.columns)} do not match {list(sample.columns)}."
        )
    if len(submission) != len(sample):
        raise RuntimeError(f"Submission has {len(submission)} rows; sample has {len(sample)}.")
    got_ids = submission[id_column].astype(str).tolist()
    expected_ids = sample[id_column].astype(str).tolist()
    if got_ids != expected_ids:
        raise RuntimeError("Submission ids are not in the same order as sample_submission.csv.")
    scores = pd.to_numeric(submission[target_column], errors="coerce")
    if scores.isna().any():
        raise RuntimeError("Submission has missing scores.")
    if ((scores < 0) | (scores > 5)).any():
        raise RuntimeError("Submission scores fall outside 0 to 5.")


def _stamp_notebook(metrics: dict) -> None:
    """Write the required training RMSE into the notebook file itself.

    A print in a cell is not enough: the competition asks for the number
    in the submitted notebook, including when the run was started from
    run_pipeline.py rather than from Jupyter.
    """
    import nbformat

    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "notebooks" / "SHL_Grammar_Scoring_Engine.ipynb"
    if not path.exists():
        return
    source = "\n".join([
        "## Evaluation results",
        "",
        f"**TRAINING RMSE (required): {metrics['training_rmse']:.4f}**",
        "",
        f"**TRAINING Pearson r: {metrics['training_pearson']:.4f}**",
        "",
        f"**5-fold CV RMSE: {metrics['cv_rmse']:.4f}**",
        "",
        f"**5-fold CV Pearson r: {metrics['cv_pearson']:.4f}**",
        "",
        (
            f"Tabular model: `{metrics['tabular_model']}`. "
            f"Blend weight on the rubric features: {metrics['blend_weight_tabular']:.2f}. "
            f"Calibration used: {metrics['calibration']['used']}. "
            f"Folds: {metrics['fold_kind']}. "
            f"Features: {metrics['n_features']}. "
            f"Train clips: {metrics['n_train']}. "
            f"Test clips: {metrics['n_test']}."
        ),
        "",
        "The training RMSE is the error of the submitted model on the training clips. The cross-validated RMSE and Pearson estimate the leaderboard.",
        "",
        (
            f"Submission rows with no published wav: {metrics.get('n_missing_audio', 0)}. "
            "Those scores are the training-set mean, because the audio is not in the competition bundle."
        ),
    ])
    notebook = nbformat.read(path, as_version=4)
    replaced = False
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## Evaluation results"):
            cell.source = source
            replaced = True
            break
    if not replaced:
        notebook.cells.append(nbformat.v4.new_markdown_cell(source))
    nbformat.write(notebook, path)
    print(f"Wrote the training RMSE into {path.name}")


if __name__ == "__main__":
    run()
