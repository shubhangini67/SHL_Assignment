"""Paths, model names, and training settings.

Everything that changes the result lives here so the notebook and the
command-line runner stay in lockstep.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Bump this when the feature definition changes so stale caches are ignored.
CACHE_VERSION = "v2"

# Apple-silicon build of Whisper. Turbo is the large-v3 decoder distilled
# into a faster model, so 45–60 s clips stay accurate without a multi-hour run.
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"

# CTC model. Unlike Whisper it has no language-model decoder, so it does not
# silently repair ungrammatical speech. Grammar features are computed on this
# literal transcript. Whisper is still used for punctuation, timing, and fluency.
LITERAL_MODEL = "facebook/wav2vec2-base-960h"
USE_LITERAL_ASR = True

# Sentence-level "is this grammatical?" classifier trained on the Corpus of
# Linguistic Acceptability. This is the closest public model to the rubric.
COLA_MODEL = "textattack/roberta-base-CoLA"

# Grammar-error correction. The edit rate between the transcript and the
# corrected text is the amount of grammar that had to be fixed.
GEC_MODEL = "vennify/t5-base-grammar-correction"

# Residual text signal (vocabulary and clause patterns rules do not list).
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Predictability of the transcript. Secondary cue, not a grammar measure.
PPL_MODEL = "gpt2"

SEED = 42
N_SPLITS = 5
SCORE_MIN = 0.0
SCORE_MAX = 5.0

# Pause longer than this, between words, counts as a hesitation.
PAUSE_SEC = 0.25
