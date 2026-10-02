"""Find the csvs and match each row to its wav.

The labels in test.csv are random. I never use them as scores.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR


class DataNotFoundError(FileNotFoundError):
    """Raised when the Kaggle files are not in data/ yet."""


@dataclass
class CompetitionData:
    train: pd.DataFrame          # audio_id, label, audio_path, submission_id
    test: pd.DataFrame           # audio_id, audio_path, submission_id
    sample_submission: pd.DataFrame
    id_column: str
    target_column: str
    data_dir: Path


def ensure_unzipped(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for archive in data_dir.glob("*.zip"):
        marker = data_dir / f".unzipped_{archive.stem}"
        if marker.exists():
            continue
        print(f"Unzipping {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
        marker.write_text("ok")


def _find_csv(data_dir: Path, name: str) -> Path:
    hits = [
        p for p in data_dir.rglob(name)
        if "__MACOSX" not in p.as_posix() and not p.name.startswith(".")
    ]
    if not hits:
        raise DataNotFoundError(
            f"Could not find {name} under {data_dir}. "
            "Download the competition data from Kaggle and unzip it into the data/ folder."
        )
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


def _index_audio(data_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in data_dir.rglob("*"):
        if path.suffix.lower() != ".wav":
            continue
        if "__MACOSX" in path.as_posix():
            continue
        index.setdefault(path.name, []).append(path)
        index.setdefault(path.stem, []).append(path)
    return index


def _match_rate(series: pd.Series, index: dict[str, list[Path]]) -> float:
    keys = set(index)
    if not keys or series.empty:
        return 0.0
    as_name = series.astype(str).map(lambda s: Path(s.strip()).name)
    as_stem = series.astype(str).map(lambda s: Path(s.strip()).stem)
    return float(max(as_name.isin(keys).mean(), as_stem.isin(keys).mean()))


def _detect_id_column(train: pd.DataFrame, sample: pd.DataFrame, index: dict) -> str:
    candidates = [c for c in sample.columns if c in train.columns] or list(train.columns)
    best_col, best_score = candidates[0], -1.0
    for col in candidates:
        score = _match_rate(train[col], index)
        if score > best_score:
            best_col, best_score = col, score
    if best_score < 0.5:
        # The id column may not be in the sample file under the same header.
        for col in train.columns:
            score = _match_rate(train[col], index)
            if score > best_score:
                best_col, best_score = col, score
    if best_score < 0.5:
        raise DataNotFoundError(
            "The CSV ids do not match the wav filenames. "
            f"Best match was column '{best_col}' at {best_score:.0%}."
        )
    return best_col


def _detect_target_column(train: pd.DataFrame, sample: pd.DataFrame, id_column: str) -> str:
    others = [c for c in sample.columns if c != id_column]
    for col in others:
        if col in train.columns and pd.api.types.is_numeric_dtype(train[col]):
            return col
    numeric = [
        c for c in train.columns
        if c != id_column and pd.api.types.is_numeric_dtype(train[c])
    ]
    if not numeric:
        raise DataNotFoundError("No numeric grammar-score column was found in train.csv.")
    in_range = []
    for col in numeric:
        values = pd.to_numeric(train[col], errors="coerce")
        share = ((values >= 0) & (values <= 5)).mean()
        in_range.append((share, col))
    in_range.sort(reverse=True)
    return in_range[0][1]


def _resolve(file_id: str, index: dict[str, list[Path]], split: str) -> Path | None:
    raw = str(file_id).strip()
    name = Path(raw).name
    stem = Path(raw).stem
    candidates = list(index.get(name, [])) or list(index.get(stem, []))
    unique: list[Path] = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    preferred = [p for p in unique if split in p.as_posix().lower()]
    return preferred[0] if preferred else unique[0]


def load_competition(data_dir: Path = DATA_DIR) -> CompetitionData:
    ensure_unzipped(data_dir)
    if not data_dir.exists():
        raise DataNotFoundError(f"Missing data directory: {data_dir}")

    train_path = _find_csv(data_dir, "train.csv")
    test_path = _find_csv(data_dir, "test.csv")
    sample_path = _find_csv(data_dir, "sample_submission.csv")
    index = _index_audio(data_dir)
    if not index:
        raise DataNotFoundError(
            f"No .wav files found under {data_dir}. Unzip the audio with the CSVs."
        )

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    # String ids preserve values like '001.wav' instead of rewriting them as numbers.
    train_ids = pd.read_csv(train_path, dtype=str)
    test_ids = pd.read_csv(test_path, dtype=str)
    sample_ids = pd.read_csv(sample_path, dtype=str)

    id_column = _detect_id_column(train_raw, sample, index)
    target_column = _detect_target_column(train_raw, sample, id_column)

    def _build(frame: pd.DataFrame, id_frame: pd.DataFrame, split: str, with_label: bool) -> pd.DataFrame:
        rows = []
        for i in range(len(frame)):
            submission_id = str(id_frame.iloc[i][id_column])
            audio = _resolve(submission_id, index, split)
            row = {
                "audio_id": Path(submission_id).stem,
                "submission_id": submission_id,
                "audio_path": str(audio) if audio is not None else None,
            }
            if with_label:
                row["label"] = float(frame.iloc[i][target_column])
            rows.append(row)
        out = pd.DataFrame(rows)
        missing = int(out["audio_path"].isna().sum())
        if missing:
            print(f"Warning: {missing} {split} rows have no matching wav.")
        if missing == len(out):
            raise DataNotFoundError(f"None of the {split} ids matched a wav file.")
        return out

    train = _build(train_raw, train_ids, "train", with_label=True)
    # Fill sample_submission.csv in its own row order.
    # Some names are training clips. Some wavs are missing. I do not read test.csv labels.
    test = _build(sample, sample_ids, "test", with_label=False)
    if test["submission_id"].duplicated().any():
        print("Duplicate submission ids. Keeping the first wav for each id.")
        test = test.drop_duplicates("submission_id", keep="first")
    ordered = sample_ids[[id_column]].copy()
    ordered.columns = ["submission_id"]
    test = ordered.merge(test, on="submission_id", how="left")
    if len(test) != len(sample_ids):
        raise DataNotFoundError(
            f"Submission alignment failed: sample has {len(sample_ids)} rows, "
            f"joined test has {len(test)}."
        )
    if test["submission_id"].astype(str).tolist() != sample_ids[id_column].astype(str).tolist():
        raise DataNotFoundError("Joined test ids do not match sample_submission.csv.")

    print(
        f"Loaded {len(train)} train and {len(test)} test clips. "
        f"id='{id_column}'  target='{target_column}'  "
        f"label mean={train['label'].mean():.3f}  std={train['label'].std():.3f}"
    )
    print(
        "I am not using the labels in test.csv. They are random."
    )
    return CompetitionData(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        sample_submission=sample_ids,
        id_column=id_column,
        target_column=target_column,
        data_dir=data_dir,
    )


def speaker_groups(audio_ids: list[str]) -> list[str] | None:
    """Use a filename prefix as a speaker id when the prefix clearly repeats.

    A random hash prefix fails this check because each value appears once.
    Grouping matters: two clips from the same speaker leak into both sides of
    a random split and make cross-validation look better than the test set.
    """
    prefixes = [str(a).split("_")[0] for a in audio_ids]
    counts = pd.Series(prefixes).value_counts()
    n = len(prefixes)
    if counts.empty:
        return None
    if not (15 <= len(counts) <= n / 2):
        return None
    if counts.min() < 2 or counts.max() > 0.2 * n:
        return None
    print(f"Speaker-style ids detected ({len(counts)} groups). Using grouped folds.")
    return prefixes
