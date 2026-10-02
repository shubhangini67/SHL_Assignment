# SHL Hiring Assessment 2026 — Grammar Scoring Engine

I score a spoken answer for grammar, from 0 to 5. The clips are about a minute long. The leaderboard uses Pearson correlation and RMSE.

The marking guide is about sentence structure, grammar mistakes, unfinished sentences, and fixing yourself. I measure those from the words. I do not treat the clip as a spectrogram.

## Notebook

`notebooks/SHL_Grammar_Scoring_Engine.ipynb` is what I submit. It prints:

- **TRAINING RMSE**. The task asks for this. It is the error on the training clips, so it looks a bit too good.
- **5-fold CV RMSE and Pearson**. These are closer to the leaderboard.

The Kaggle file is `outputs/submission.csv`.

## How to run

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Put `train.csv`, `test.csv`, `sample_submission.csv`, and the wav files in `data/`. A zip in that folder is unpacked on its own. If `~/.kaggle/kaggle.json` is set up:

```bash
python scripts/download_competition.py
python run_pipeline.py
```

The first run transcribes the 769 training clips and saves them in `cache/`. On my Mac that is about 14 seconds per clip, so the first run takes several hours. The next run only fits the model again.

I do not use the labels in `test.csv`. The file I fill is `sample_submission.csv` (204 rows). 99 of those names are not in the audio zip. Those rows get the average training score.
