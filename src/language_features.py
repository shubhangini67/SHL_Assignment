"""Counts I can explain from the marking guide.

Sentence shape, mistakes, unfinished sentences, and self-corrections.
I do not use spectral features for those.
"""

from __future__ import annotations

import re

import numpy as np

from src.config import PAUSE_SEC

FILLERS = {"um", "uh", "er", "ah", "hmm", "uhh", "umm", "erm"}
REPAIR_PHRASES = (
    "i mean",
    "sorry",
    "let me",
    "i meant",
    "no no",
    "wait no",
    "i should say",
    "what i meant",
)
# Words that start a dependent clause. I skip "that" because it is usually just "that".
SUBORDINATORS = {
    "because", "although", "though", "unless", "whereas", "whether",
    "while", "which", "who", "if", "when", "since", "before", "after",
}
CLAUSE_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl"}
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "PART", "AUX"}
DANGLING_ENDINGS = {"and", "or", "but", "to", "the", "a", "an", "of", "because", "if", "so"}


def normalize_ctc_case(text: str) -> str:
    """wav2vec2 writes in capitals. Acceptability models are trained on normal case."""
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 8:
        return text
    if sum(ch.isupper() for ch in letters) / len(letters) < 0.75:
        return text
    lowered = text.lower()
    return lowered[:1].upper() + lowered[1:]


def tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9'\s]", " ", (text or "").lower())
    return [tok for tok in cleaned.split() if tok]


def _rate(count: float, denom: float) -> float:
    return float(count) / float(denom) if denom else 0.0


def _dependency_depth(token) -> int:
    depth = 0
    seen = set()
    while token.head is not token and token.i not in seen and depth < 40:
        seen.add(token.i)
        token = token.head
        depth += 1
    return depth


def delivery_features(transcript: dict) -> dict:
    """Timing, hesitations, and self-repairs from the Whisper word timestamps."""
    segments = transcript.get("segments") or []
    words = []
    for segment in segments:
        for word in segment.get("words") or []:
            token = re.sub(r"[^a-z0-9']", "", (word.get("word") or "").lower())
            if not token:
                continue
            words.append({
                "token": token,
                "start": float(word.get("start") or 0.0),
                "end": float(word.get("end") or 0.0),
                "probability": word.get("probability"),
            })

    tokens = [w["token"] for w in words]
    n = len(tokens)
    text_tokens = tokenize(transcript.get("text") or "")
    if n == 0:
        tokens = text_tokens
        n = len(tokens)

    gaps = []
    durations = []
    for left, right in zip(words, words[1:]):
        gap = right["start"] - left["end"]
        if gap > 0:
            gaps.append(gap)
        dur = left["end"] - left["start"]
        if dur > 0:
            durations.append(dur)
    pauses = [g for g in gaps if g >= PAUSE_SEC]

    spoken = float(sum(durations))
    probs = [w["probability"] for w in words if w["probability"] is not None]
    logprobs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
    no_speech = [s.get("no_speech_prob") for s in segments if s.get("no_speech_prob") is not None]
    compression = [s.get("compression_ratio") for s in segments if s.get("compression_ratio") is not None]

    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    fillers = sum(1 for tok in tokens if tok in FILLERS)
    lowered = " ".join(tokens)
    repairs = sum(lowered.count(phrase) for phrase in REPAIR_PHRASES)

    false_starts = 0
    for i in range(max(0, n - 3)):
        bigram = (tokens[i], tokens[i + 1])
        window = tokens[i + 2:i + 8]
        for j in range(len(window) - 1):
            if (window[j], window[j + 1]) == bigram:
                false_starts += 1
                break

    return {
        "n_words": float(n),
        "speech_rate_wps": _rate(n, _duration(segments, words)),
        "articulation_rate": _rate(n, spoken),
        "pause_rate_per_word": _rate(len(pauses), n),
        "pause_ratio": _rate(sum(pauses), _duration(segments, words)),
        "mean_pause_sec": float(np.mean(pauses)) if pauses else 0.0,
        "max_pause_sec": float(np.max(pauses)) if pauses else 0.0,
        "mean_word_dur": float(np.mean(durations)) if durations else 0.0,
        "std_word_dur": float(np.std(durations)) if durations else 0.0,
        "filler_rate": _rate(fillers, n),
        "immediate_repeat_rate": _rate(repeats, n),
        "false_start_rate": _rate(false_starts, n),
        "repair_marker_rate": _rate(repairs, n),
        "mean_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
        "mean_word_prob": float(np.mean(probs)) if probs else 0.0,
        "mean_no_speech_prob": float(np.mean(no_speech)) if no_speech else 0.0,
        "mean_compression_ratio": float(np.mean(compression)) if compression else 0.0,
        "unique_word_ratio": _rate(len(set(tokens)), n),
        "ends_incomplete": float(bool(tokens) and tokens[-1] in DANGLING_ENDINGS),
        "whisper_is_english": float((transcript.get("language") or "en") == "en"),
        "transcript_missing": float(n == 0),
    }


def _duration(segments, words) -> float:
    if words:
        return max(0.0, words[-1]["end"] - words[0]["start"])
    if segments:
        return max(0.0, float(segments[-1].get("end") or 0) - float(segments[0].get("start") or 0))
    return 0.0


def syntax_features(text: str, nlp) -> dict:
    """Parse statistics. spaCy's parser is the sentence-structure half of the rubric."""
    tokens = tokenize(text)
    n = len(tokens)
    empty = {
        "n_sentences": 0.0,
        "mlu_words": 0.0,
        "std_sentence_len": 0.0,
        "fragment_rate": 1.0 if n == 0 else 0.0,
        "short_sentence_rate": 0.0,
        "mean_dep_depth": 0.0,
        "max_dep_depth": 0.0,
        "clause_per_sentence": 0.0,
        "conj_per_sentence": 0.0,
        "subordinate_marker_rate": _rate(sum(t in SUBORDINATORS for t in tokens), n),
        "lexical_density": 0.0,
        "guiraud_ttr": _rate(len(set(tokens)), np.sqrt(n)) if n else 0.0,
        "avg_word_len": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "long_word_rate": _rate(sum(len(t) > 6 for t in tokens), n),
        "function_word_rate": 0.0,
    }
    if n == 0:
        return empty

    doc = nlp(text if text.strip() else " ")
    sentences = list(doc.sents) or [doc]
    lengths = []
    fragments = 0
    short = 0
    depths = []
    clauses = 0
    conjs = 0
    content = 0
    function = 0
    kept = 0
    for sent in sentences:
        words = [t for t in sent if not t.is_punct and not t.is_space]
        if not words:
            continue
        lengths.append(len(words))
        has_verb = any(t.pos_ in {"VERB", "AUX"} for t in words)
        fragments += int(not has_verb)
        short += int(len(words) <= 3)
        depths.extend(_dependency_depth(t) for t in words)
        clauses += sum(t.dep_ in CLAUSE_DEPS for t in words)
        conjs += sum(t.dep_ == "conj" for t in words)
        content += sum(t.pos_ in CONTENT_POS for t in words)
        function += sum(t.pos_ in FUNCTION_POS for t in words)
        kept += len(words)

    n_sent = len(lengths) or 1
    empty.update({
        "n_sentences": float(len(lengths)),
        "mlu_words": float(np.mean(lengths)) if lengths else 0.0,
        "std_sentence_len": float(np.std(lengths)) if lengths else 0.0,
        "fragment_rate": _rate(fragments, len(lengths)),
        "short_sentence_rate": _rate(short, len(lengths)),
        "mean_dep_depth": float(np.mean(depths)) if depths else 0.0,
        "max_dep_depth": float(np.max(depths)) if depths else 0.0,
        "clause_per_sentence": _rate(clauses, n_sent),
        "conj_per_sentence": _rate(conjs, n_sent),
        "lexical_density": _rate(content, kept),
        "function_word_rate": _rate(function, kept),
    })
    return empty


def load_spacy():
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    # I only need the parser and the part-of-speech tags.
    if "ner" in nlp.pipe_names:
        nlp.disable_pipes("ner")
    return nlp
