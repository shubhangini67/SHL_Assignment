"""Build one feature row per clip and cache it.

The grammar transcript is the literal CTC text when that pass succeeded.
Whisper text is the fallback, and it is always the source of timing features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, CACHE_VERSION
from src.language_features import delivery_features, load_spacy, normalize_ctc_case, syntax_features
from src.text_signals import cola_and_gec, embeddings, perplexities
from src.transcribe import transcribe_literal, transcribe_whisper


def _grammar_text(audio_id: str, whisper: dict, literal: dict) -> str:
    literal_text = (literal.get(audio_id) or "").strip()
    if len(literal_text.split()) >= 5:
        return literal_text
    return (whisper.get(audio_id, {}).get("text") or "").strip()


def _version_dir(cache_dir: Path) -> Path:
    folder = cache_dir / CACHE_VERSION
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _table_cache(split: str, cache_dir: Path) -> Path:
    return _version_dir(cache_dir) / f"features_{split}.csv"


def _emb_cache(split: str, cache_dir: Path) -> Path:
    return _version_dir(cache_dir) / f"embeddings_{split}.npy"


def build_split(frame: pd.DataFrame, cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, np.ndarray, dict]:
    whisper = transcribe_whisper(frame, cache_dir)
    literal, acoustic = transcribe_literal(frame, cache_dir)
    ids = frame["audio_id"].tolist()
    # Literal transcript, with capitals restored to normal case for the grammar models.
    grammar_texts = [normalize_ctc_case(_grammar_text(i, whisper, literal)) for i in ids]
    whisper_texts = [(whisper.get(i, {}).get("text") or "") for i in ids]

    print("Parsing transcripts for sentence-structure features.")
    nlp = load_spacy()
    rows = []
    for audio_id, whisper_text, grammar_text in zip(ids, whisper_texts, grammar_texts):
        row = {"audio_id": audio_id}
        row.update(delivery_features(whisper.get(audio_id, {})))
        # Sentence structure needs punctuation. Whisper has it; the CTC transcript
        # usually does not, so a 60 s literal transcript would be one fake sentence.
        structure_text = whisper_text.strip() or grammar_text
        row.update(syntax_features(structure_text, nlp))
        row.update(acoustic.get(audio_id, {}))
        row["literal_word_count"] = float(len((literal.get(audio_id) or "").split()))
        row["used_literal_transcript"] = float(
            len((literal.get(audio_id) or "").split()) >= 5
        )
        rows.append(row)
    hand = pd.DataFrame(rows)

    try:
        neural = cola_and_gec(ids, grammar_texts, correct=True)
    except Exception as exc:
        print(f"Acceptability / grammar-correction step failed ({exc}). Those features will be empty.")
        neural = pd.DataFrame({"audio_id": ids})
    # Acceptability of the Whisper text minus acceptability of the literal text.
    # Whisper often rewrites ungrammatical speech, so this gap is a candidate
    # repair signal. The regression keeps it only when it predicts the score.
    whisper_changed = [w.strip() != g.strip() for w, g in zip(whisper_texts, grammar_texts)]
    if any(whisper_changed):
        try:
            whisper_neural = cola_and_gec(ids, whisper_texts, correct=False)
            whisper_neural = whisper_neural.rename(columns={
                "cola_mean": "whisper_cola_mean",
                "cola_min": "whisper_cola_min",
                "cola_std": "whisper_cola_std",
                "cola_low_rate": "whisper_cola_low_rate",
            })
            keep = ["audio_id", "whisper_cola_mean", "whisper_cola_min", "whisper_cola_std", "whisper_cola_low_rate"]
            neural = neural.merge(whisper_neural[keep], on="audio_id", how="left")
        except Exception as exc:
            print(f"Whisper acceptability step failed ({exc}).")
    try:
        ppl = perplexities(ids, grammar_texts)
    except Exception as exc:
        print(f"Perplexity step failed ({exc}).")
        ppl = pd.DataFrame({"audio_id": ids, "gpt2_perplexity": np.nan})
    try:
        emb = embeddings(ids, grammar_texts)
    except Exception as exc:
        print(f"Embedding step failed ({exc}).")
        emb = np.zeros((len(ids), 32), dtype=np.float32)

    features = hand.merge(neural, on="audio_id", how="left").merge(ppl, on="audio_id", how="left")
    # Log perplexity is easier for a tree to split, and a few pathological
    # transcripts would otherwise dominate the raw exponential value.
    features["gpt2_log_perplexity"] = np.log(pd.to_numeric(features["gpt2_perplexity"], errors="coerce").clip(lower=1))
    if "whisper_cola_mean" in features.columns and "cola_mean" in features.columns:
        features["cola_repair_gap"] = features["whisper_cola_mean"] - features["cola_mean"]
    features = features.replace([np.inf, -np.inf], np.nan)

    transcripts = {
        audio_id: {
            "whisper": whisper_texts[i][:1200],
            "grammar": grammar_texts[i][:1200],
        }
        for i, audio_id in enumerate(ids)
    }
    return features, emb, transcripts


def build_features(train: pd.DataFrame, test: pd.DataFrame, cache_dir: Path = CACHE_DIR):
    train_csv, train_npy = _table_cache("train", cache_dir), _emb_cache("train", cache_dir)
    test_csv, test_npy = _table_cache("test", cache_dir), _emb_cache("test", cache_dir)
    transcript_path = cache_dir / CACHE_VERSION / "transcript_preview.json"

    if train_csv.exists() and train_npy.exists() and test_csv.exists() and test_npy.exists():
        cached_train = pd.read_csv(train_csv, dtype={"audio_id": str})
        cached_test = pd.read_csv(test_csv, dtype={"audio_id": str})
        same_train = set(cached_train["audio_id"].astype(str)) == set(train["audio_id"].astype(str))
        same_test = set(cached_test["audio_id"].astype(str)) == set(test["audio_id"].astype(str))
        if same_train and same_test and len(cached_train) == len(train) and len(cached_test) == len(test):
            print("Feature cache found. Delete cache/ to recompute from audio.")
            import json
            previews = json.loads(transcript_path.read_text()) if transcript_path.exists() else {}
            return cached_train, cached_test, np.load(train_npy), np.load(test_npy), previews
        print("Feature cache does not match the current CSV files. Recomputing.")

    print("\n=== Training clips ===")
    train_features, train_emb, train_text = build_split(train, cache_dir)
    print("\n=== Test clips ===")
    test_features, test_emb, test_text = build_split(test, cache_dir)

    train_features.to_csv(train_csv, index=False)
    test_features.to_csv(test_csv, index=False)
    np.save(train_npy, train_emb)
    np.save(test_npy, test_emb)
    import json
    transcript_path.write_text(json.dumps({"train": train_text, "test": test_text}))
    return train_features, test_features, train_emb, test_emb, {"train": train_text, "test": test_text}
