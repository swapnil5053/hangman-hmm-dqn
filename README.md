# Intelligent Hangman (HMM + optional DQN)

## Setup

1. Create venv and install deps
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Place your dataset
- Put `corpus.txt` and `test_words.txt` into `data/`.

## Run Baseline (HMM Greedy)
Corpus (sampled 2000 games):
```
python -m src.evaluate --mode baseline --n_games 2000 --lives 6 --seed 42 --outdir plots --data_dir data --test data/corpus.txt
```
Test set (use full test_words.txt):
```
$n=(Get-Content data\test_words.txt | Measure-Object -Line).Lines
python -m src.evaluate --mode baseline --n_games $n --lives 6 --seed 42 --outdir plots --data_dir data --test data/test_words.txt
```

## Train DQN (optional)
```
python -m src.evaluate --mode dqn --episodes 2000 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay 0.995 --seed 42 --lives 6 --outdir plots --data_dir data --test data/corpus.txt
```

Evaluate DQN (corpus, hybrid top-k enabled in code):
```
python -m src.evaluate --mode dqn_eval --n_games 2000 --lives 6 --seed 42 --outdir plots --data_dir data --test data/corpus.txt
```

Evaluate DQN (test set, strict or hybrid depending on code path):
```
$n=(Get-Content data\test_words.txt | Measure-Object -Line).Lines
python -m src.evaluate --mode dqn_eval --n_games $n --lives 6 --seed 42 --outdir plots --data_dir data --test data/test_words.txt
```

## Generate Report
Baseline consolidated (no viva):
```
python -m src.evaluate --mode report --n_games 2000 --seed 42 --outdir plots --data_dir data --test data/corpus.txt
```

DQN 1‑page summary (reads latest eval metrics):
```
python -m src.evaluate --mode dqn_summary --outdir plots --data_dir data
```

Final analysis PDF from Markdown:
```
python scripts/md_to_pdf.py Analysis_Report.md Final_Analysis_Report.pdf
```

Outputs:
- Plots in `plots/`
- Baseline CSV in `data/baseline_results.csv`
- DQN CSV in `data/dqn_results.csv`
- Reports at project root:
  - `Analysis_Report.pdf` (baseline consolidated)
  - `DQN_Summary_Report.pdf` (1‑page HMM vs DQN)
  - `Final_Analysis_Report.pdf` (final consolidated with corpus vs test)

## Import Usage
In notebooks or scripts, you can import the package APIs:
```
from src import BigramLM, HangmanEnv, greedy_play, DQNAgent
```

## Notebooks
Open the notebooks in `notebooks/` for step-by-step demo:
- 01_data_and_hmm.ipynb
- 02_baseline_greedy.ipynb
- 03_train_dqn.ipynb (optional)
- 04_generate_report.ipynb

## Project Structure
```
intelligent_hangman/
  data/
    corpus.txt
    test_words.txt
  src/
    utils.py
    hmm_oracle.py
    baseline_greedy.py
    hangman_env.py
    dqn_agent.py
    evaluate.py
  notebooks/
    01_data_and_hmm.ipynb
    02_baseline_greedy.ipynb
    03_train_dqn.ipynb
    04_generate_report.ipynb
  Analysis_Report.pdf
  DQN_Summary_Report.pdf
  Final_Analysis_Report.pdf
```
