"""Paths, model names, and the settings I train with."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Change this if I change the features, so old cache files are not reused.
CACHE_VERSION = "v2"

# Whisper on Apple silicon. I use it for word times and punctuation.
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"

# wav2vec2 does not clean up bad grammar. I measure grammar on this text.
LITERAL_MODEL = "facebook/wav2vec2-base-960h"
USE_LITERAL_ASR = True

# "Is this sentence grammatical?"
COLA_MODEL = "textattack/roberta-base-CoLA"

# Grammar correction. I count how many words it changes.
GEC_MODEL = "vennify/t5-base-grammar-correction"

# Small text embedding, used only if the 5-fold check says it helps.
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How predictable the text is. A side clue, not the grammar score.
PPL_MODEL = "gpt2"

SEED = 42
N_SPLITS = 5
SCORE_MIN = 0.0
SCORE_MAX = 5.0

# A gap longer than this, between words, counts as a pause.
PAUSE_SEC = 0.25
