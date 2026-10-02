# SHL Hiring Assessment 2026 — Grammar Scoring Engine

Scores a 45–60 second spoken response for grammar on a continuous 0–5 scale. The leaderboard metrics are Pearson correlation and RMSE.

The grammar rubric is about sentence structure, grammatical mistakes, incomplete sentences, self-correction, and control of complex grammar. The model measures those things from the words, instead of treating the clip as a spectrogram.

## What the notebook reports

`notebooks/SHL_Grammar_Scoring_Engine.ipynb` is the submission notebook. It prints:

- **TRAINING RMSE**, which the competition requires. This is the in-sample error.
- **5-fold CV RMSE and Pearson**, which estimate the leaderboard.

The file to upload to Kaggle is `outputs/submission.csv`.

## Run

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Put `train.csv`, `test.csv`, `sample_submission.csv`, and the wav files in `data/` (a Kaggle zip in that folder is unpacked automatically). Or, with a valid `~/.kaggle/kaggle.json`:

```bash
python scripts/download_competition.py
python run_pipeline.py
```

The first run transcribes the 769 training clips and caches the result in `cache/`. On an Apple M3 that pass is about 14 seconds per clip, so the first run takes several hours. A later run only refits the regression.

`test.csv` labels are not used. The scored file is `sample_submission.csv` (204 rows). 99 of those filenames are not in the published audio; those rows receive the training-set mean.
