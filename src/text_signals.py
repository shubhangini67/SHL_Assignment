"""Neural text measurements: acceptability, correction effort, perplexity, embeddings.

Models are loaded one at a time. A 16 GB machine cannot hold Whisper, wav2vec2,
RoBERTa, T5, and GPT-2 together, and it does not need to.
"""

from __future__ import annotations

import gc
import os
import re

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import COLA_MODEL, EMB_MODEL, GEC_MODEL, PPL_MODEL


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def release(model) -> None:
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for part in parts:
        words = part.split()
        if not words:
            continue
        if len(words) <= 32:
            sentences.append(part.strip())
            continue
        # CTC transcripts have no punctuation, so long stretches are windowed.
        for start in range(0, len(words), 20):
            window = " ".join(words[start:start + 20]).strip()
            if window:
                sentences.append(window)
    return sentences


def _word_edit_rate(source: str, target: str) -> float:
    import difflib

    src = source.lower().split()
    tgt = target.lower().split()
    if not src:
        return 0.0
    matcher = difflib.SequenceMatcher(a=src, b=tgt, autojunk=False)
    edits = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            edits += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            edits += i2 - i1
        elif tag == "insert":
            edits += j2 - j1
    return edits / len(src)


def cola_and_gec(ids: list[str], texts: list[str], correct: bool = True) -> pd.DataFrame:
    """Acceptability and, when correct=True, grammar-correction edit rate per clip."""
    dev = device()
    sentence_rows = []
    for audio_id, text in zip(ids, texts):
        sentences = split_sentences(text)
        if not sentences:
            sentences = [""]
        for sentence in sentences:
            sentence_rows.append((audio_id, sentence))

    print(f"CoLA acceptability on {dev} ({COLA_MODEL}).")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cola_tok = AutoTokenizer.from_pretrained(COLA_MODEL)
    cola = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL).to(dev)
    cola.eval()
    # CoLA's positive class is "acceptable". Some checkpoints only name it LABEL_1,
    # which is the GLUE convention, so 1 is the fallback.
    accept_index = 1
    for idx, name in cola.config.id2label.items():
        label = str(name).lower()
        if "unaccept" in label:
            continue
        if "accept" in label:
            accept_index = int(idx)
    accept_probs = []
    batch = 16
    nonempty = [row[1] if row[1] else "." for row in sentence_rows]
    with torch.no_grad():
        for start in tqdm(range(0, len(nonempty), batch), desc="CoLA"):
            chunk = nonempty[start:start + batch]
            encoded = cola_tok(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            encoded = {k: v.to(dev) for k, v in encoded.items()}
            logits = cola(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, accept_index].detach().cpu().numpy()
            accept_probs.extend(float(p) for p in probs)
    release(cola)
    del cola_tok

    corrected = [""] * len(nonempty)
    if correct:
        print(f"Grammar correction on {dev} ({GEC_MODEL}).")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as SeqTokenizer

        gec_tok = SeqTokenizer.from_pretrained(GEC_MODEL)
        gec = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL).to(dev)
        gec.eval()
        corrected = []
        gec_batch = 8
        with torch.no_grad():
            for start in tqdm(range(0, len(nonempty), gec_batch), desc="GEC"):
                chunk = ["grammar: " + s for s in nonempty[start:start + gec_batch]]
                encoded = gec_tok(
                    chunk, return_tensors="pt", padding=True, truncation=True, max_length=96
                )
                encoded = {k: v.to(dev) for k, v in encoded.items()}
                generated = gec.generate(**encoded, max_new_tokens=96, num_beams=2)
                decoded = gec_tok.batch_decode(generated, skip_special_tokens=True)
                corrected.extend(s.strip() for s in decoded)
        release(gec)
        del gec_tok

    per_id: dict[str, dict] = {
        audio_id: {"cola": [], "wer": [], "changed": []} for audio_id in ids
    }
    for (audio_id, source), prob, fix in zip(sentence_rows, accept_probs, corrected):
        if not source:
            continue
        per_id[audio_id]["cola"].append(prob)
        if correct:
            wer = _word_edit_rate(source, fix)
            per_id[audio_id]["wer"].append(wer)
            per_id[audio_id]["changed"].append(float(wer > 0.02))

    rows = []
    for audio_id in ids:
        cola_vals = per_id[audio_id]["cola"]
        wer_vals = per_id[audio_id]["wer"]
        changed = per_id[audio_id]["changed"]
        rows.append({
            "audio_id": audio_id,
            "cola_mean": float(np.mean(cola_vals)) if cola_vals else np.nan,
            "cola_min": float(np.min(cola_vals)) if cola_vals else np.nan,
            "cola_std": float(np.std(cola_vals)) if cola_vals else np.nan,
            "cola_low_rate": float(np.mean([c < 0.5 for c in cola_vals])) if cola_vals else np.nan,
            "gec_wer": float(np.mean(wer_vals)) if wer_vals else np.nan,
            "gec_wer_max": float(np.max(wer_vals)) if wer_vals else np.nan,
            "gec_sent_changed_rate": float(np.mean(changed)) if changed else np.nan,
        })
    return pd.DataFrame(rows)


def perplexities(ids: list[str], texts: list[str]) -> pd.DataFrame:
    dev = device()
    print(f"GPT-2 perplexity on {dev}.")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL)
    model = GPT2LMHeadModel.from_pretrained(PPL_MODEL).to(dev)
    model.eval()
    values = []
    with torch.no_grad():
        for text in tqdm(texts, desc="perplexity"):
            if not (text or "").strip():
                values.append(np.nan)
                continue
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            encoded = {k: v.to(dev) for k, v in encoded.items()}
            if encoded["input_ids"].shape[1] < 2:
                values.append(np.nan)
                continue
            loss = model(**encoded, labels=encoded["input_ids"]).loss
            values.append(float(torch.exp(loss).detach().cpu()))
    release(model)
    return pd.DataFrame({"audio_id": ids, "gpt2_perplexity": values})


def embeddings(ids: list[str], texts: list[str]) -> np.ndarray:
    """Mean-pooled MiniLM vectors, L2-normalized, aligned to ids."""
    dev = device()
    print(f"Text embeddings on {dev} ({EMB_MODEL}).")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
    model = AutoModel.from_pretrained(EMB_MODEL).to(dev)
    model.eval()
    vectors = []
    batch = 16
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch), desc="embeddings"):
            chunk = [t if (t or "").strip() else "." for t in texts[start:start + batch]]
            encoded = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            encoded = {k: v.to(dev) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    release(model)
    if not vectors:
        return np.zeros((len(ids), 384), dtype=np.float32)
    return np.vstack(vectors)

