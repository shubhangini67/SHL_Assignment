"""Two transcripts.

Whisper is good at timing, but it cleans up bad grammar. wav2vec2 stays
closer to what was said. I use Whisper for pauses and wav2vec2 for grammar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.config import CACHE_VERSION, LITERAL_MODEL, USE_LITERAL_ASR, WHISPER_REPO


def _cache_file(cache_dir: Path, kind: str, audio_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in audio_id)[:180]
    folder = cache_dir / CACHE_VERSION / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe}.json"


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text())


def _slim_whisper(result: dict) -> dict:
    segments = []
    for segment in result.get("segments") or []:
        words = []
        for word in segment.get("words") or []:
            probability = word.get("probability")
            words.append({
                "word": word.get("word", ""),
                "start": float(word.get("start") or 0.0),
                "end": float(word.get("end") or 0.0),
                "probability": None if probability is None else float(probability),
            })
        segments.append({
            "start": float(segment.get("start") or 0.0),
            "end": float(segment.get("end") or 0.0),
            "text": segment.get("text") or "",
            "avg_logprob": float(segment.get("avg_logprob") or 0.0),
            "no_speech_prob": float(segment.get("no_speech_prob") or 0.0),
            "compression_ratio": float(segment.get("compression_ratio") or 0.0),
            "words": words,
        })
    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language") or "",
        "segments": segments,
    }


def _usable_audio(path) -> bool:
    if path is None:
        return False
    try:
        if path != path:  # NaN
            return False
    except Exception:
        return False
    text = str(path)
    if not text or text == "nan":
        return False
    return Path(text).is_file()


def transcribe_whisper(rows, cache_dir: Path, repo: str = WHISPER_REPO) -> dict[str, dict]:
    """rows: dataframe with audio_id and audio_path. Returns id -> transcript."""
    pending = []
    transcripts: dict[str, dict] = {}
    for row in rows.itertuples(index=False):
        path = _cache_file(cache_dir, "whisper", row.audio_id)
        if path.exists():
            transcripts[row.audio_id] = _json_load(path)
        elif _usable_audio(row.audio_path):
            pending.append(row)

    if not pending:
        print(f"Whisper transcripts loaded from cache ({len(transcripts)} files).")
        return transcripts

    import mlx_whisper

    print(f"Transcribing {len(pending)} files with {repo}. Cached files are skipped.")
    for row in tqdm(pending, desc="Whisper"):
        try:
            result = mlx_whisper.transcribe(
                row.audio_path,
                path_or_hf_repo=repo,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
                verbose=None,
            )
            payload = _slim_whisper(result)
        except Exception as exc:  # one bad file should not stop the run
            print(f"Whisper failed on {row.audio_id}: {exc}")
            payload = {"text": "", "language": "", "segments": [], "error": str(exc)}
        _json_dump(_cache_file(cache_dir, "whisper", row.audio_id), payload)
        transcripts[row.audio_id] = payload
    return transcripts


def _read_mono(path: str) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def _to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(sample_rate, 16000)
    return resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)


def acoustic_from_array(audio: np.ndarray, sample_rate: int) -> dict:
    """Cheap recording-quality controls, computed from the same read as the CTC pass.

    These are not grammar features. They stop the model treating a quiet or
    noisy recording as poor grammar.
    """
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    empty = {
        "duration_sec": duration,
        "rms_mean": 0.0,
        "rms_std": 0.0,
        "silence_ratio": 1.0,
        "zcr_mean": 0.0,
    }
    frame = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    if frame < 8 or len(audio) < frame:
        return empty
    count = 1 + (len(audio) - frame) // hop
    # Strided frames avoid building a full sliding-window copy of a 60 s file.
    stride = audio.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, frame),
        strides=(hop * stride, stride),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    threshold = max(0.005, float(np.percentile(rms, 20)) * 0.5)
    return {
        "duration_sec": duration,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "silence_ratio": float(np.mean(rms < threshold)),
        "zcr_mean": float(np.mean(zcr)),
    }


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def transcribe_literal(rows, cache_dir: Path, model_name: str = LITERAL_MODEL) -> tuple[dict, dict]:
    """Return (id -> literal text, id -> acoustic features)."""
    texts: dict[str, str] = {}
    acoustics: dict[str, dict] = {}
    pending = []
    for row in rows.itertuples(index=False):
        text_path = _cache_file(cache_dir, "literal", row.audio_id)
        ac_path = _cache_file(cache_dir, "acoustic", row.audio_id)
        if text_path.exists() and ac_path.exists():
            texts[row.audio_id] = _json_load(text_path).get("text", "")
            acoustics[row.audio_id] = _json_load(ac_path)
        elif _usable_audio(row.audio_path):
            pending.append(row)

    if not pending:
        print(f"Literal transcripts loaded from cache ({len(texts)} files).")
        return texts, acoustics

    if not USE_LITERAL_ASR:
        print("Literal ASR disabled. Acoustic features only.")
        for row in tqdm(pending, desc="Acoustic"):
            audio, sample_rate = _read_mono(row.audio_path)
            acoustic = acoustic_from_array(audio, sample_rate)
            _json_dump(_cache_file(cache_dir, "acoustic", row.audio_id), acoustic)
            _json_dump(_cache_file(cache_dir, "literal", row.audio_id), {"text": ""})
            texts[row.audio_id] = ""
            acoustics[row.audio_id] = acoustic
        return texts, acoustics

    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = _device()
    print(f"Literal ASR on {len(pending)} files with {model_name} ({device}).")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()
    chunk = 15 * 16000

    for row in tqdm(pending, desc="wav2vec2"):
        try:
            audio, sample_rate = _read_mono(row.audio_path)
            acoustic = acoustic_from_array(audio, sample_rate)
            wav = _to_16k(audio, sample_rate)
            pieces = []
            for start in range(0, len(wav), chunk):
                piece = wav[start:start + chunk]
                if len(piece) < 400:
                    continue
                inputs = processor(piece, sampling_rate=16000, return_tensors="pt")
                values = inputs.input_values.to(device)
                with torch.no_grad():
                    logits = model(values).logits
                token_ids = torch.argmax(logits, dim=-1)
                pieces.append(processor.batch_decode(token_ids)[0])
            text = " ".join(p.strip() for p in pieces if p.strip())
        except Exception as exc:
            print(f"Literal ASR failed on {row.audio_id}: {exc}")
            text = ""
            acoustic = {
                "duration_sec": 0.0,
                "rms_mean": 0.0,
                "rms_std": 0.0,
                "silence_ratio": 1.0,
                "zcr_mean": 0.0,
            }
        _json_dump(_cache_file(cache_dir, "literal", row.audio_id), {"text": text})
        _json_dump(_cache_file(cache_dir, "acoustic", row.audio_id), acoustic)
        texts[row.audio_id] = text
        acoustics[row.audio_id] = acoustic

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return texts, acoustics
