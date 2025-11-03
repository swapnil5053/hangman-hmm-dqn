# Intelligent Hangman — Simple Project Guide

This document explains what the project does, how the models work, and the end‑to‑end workflow in simple, practical terms.

## 1) What is this project?

An AI agent that plays Hangman. It learns from a word list (the corpus) and tries to guess letters with as few mistakes as possible. It has two components:

- A probabilistic letter oracle (HMM Bigram) used by a Greedy Baseline agent.
- An optional Reinforcement Learning (DQN) agent that learns a policy to pick letters.

## 2) Data and folders

- Put your dataset in `data/`.
  - `data/corpus.txt`: large list of words used to learn patterns.
  - `data/test_words.txt`: words used for evaluation (optional). If not provided, we sample from the corpus.
- Outputs:
  - CSV results in `data/`
  - Plots in `plots/`
  - Final report as `Analysis_Report.pdf` in project root

## 3) The HMM Bigram Oracle (core idea)

- We treat each word as a sequence of letters (plus start and end markers).
- The model counts how often letter B follows letter A (bigram counts) and converts those counts into probabilities with smoothing.
- At game time, for a masked word like `_ p p _ e`, we:
  1. Find all candidate words in the corpus that match the pattern and do not contain already-wrong letters.
  2. Build a letter posterior by two signals:
     - Frequency: which letters fill the unknown positions among candidates.
     - Context: bigram probabilities for neighbors (previous and next if known).
  3. Combine both into a normalized letter distribution and mask out already-guessed letters.
- This gives `P(letter | pattern, context)` — our “oracle output”.

## 4) Greedy Baseline Agent

- At each step, the agent picks the highest‑probability letter from the oracle that it has not guessed yet.
- If the guess is correct, letters get revealed; if not, a life is lost.
- The agent tracks mistakes (wrong guesses) and repeats (should be ~0 with proper masking).

## 5) Hangman Environment (for RL and simulation)

- State vector includes:
  - Masked word encoding (one‑hot over letters, padded to a max length)
  - Guessed letters mask (26‑dim binary)
  - Oracle letter probabilities (26‑dim)
  - Lives remaining (normalized)
- Step(action = letter index) returns next state, reward, and whether the word was solved or lives ran out.

## 6) DQN Agent (optional)

- Uses a small MLP to predict Q‑values for the 26 letter actions.
- ε‑greedy policy for exploration; replay buffer + target network for stability.
- Reward shaping:
  - +1 correct guess
  - −2 wrong guess
  - +10 solve
  - −10 lose
  - −0.1 per step (encourages fast solves)
- Goal: learn to pick letters that minimize mistakes and finish quickly.

## 7) End‑to‑End Workflow

1. Prepare data in `data/` (corpus and optional test file).
2. Train + evaluate baseline:
   - `python -m src.evaluate --mode baseline --n_games 1000 --lives 6 --seed 42 --outdir plots --data_dir data`
   - If you want to sample from the corpus, pass a non‑existent test file: `--test NOFILE`.
3. (Optional) Train RL/DQN:
   - `python -m src.evaluate --mode dqn --episodes 2000 --seed 42 --outdir plots --data_dir data`
4. Generate report:
   - `python -m src.evaluate --mode report --n_games 1000 --seed 42 --outdir plots --data_dir data`

## 8) What to look at after runs

- `data/baseline_results.csv`: per‑game results (word, success, wrong guesses, guesses sequence).
- `data/dqn_results.csv` (if trained): per‑episode reward, success, epsilon.
- `plots/`: letter frequency, wrong‑guess histograms, DQN training curves.
- `Analysis_Report.pdf`: summary compiled into a single PDF.
- `RUN_SUMMARY.md`: concise log of recent actions and final metrics.

## 9) Typical Metrics

- Success Rate (%) = wins / games × 100
- Wrong guesses (total and average per game)
- Repeated guesses (should be near 0)
- Final Score = (SuccessRate × 2000) − (Wrong × 5) − (Repeated × 2)

## 10) Why this works (simple intuition)

- English words have local patterns (e.g., vowels between consonants). The bigram model captures these letter‑to‑letter tendencies.
- Conditioning on the current pattern (revealed letters) filters out impossible words.
- Combining bigram context with candidate frequencies yields smarter letter choices than plain global frequency.
- RL (DQN) can further learn which letters to try earlier to reduce mistakes when lives are limited.

## 11) Limitations & Improvements

- Bigram context is local; longer context (trigrams) or subword features could help on rare patterns.
- RL benefits more on large, diverse corpora and longer training (e.g., 10k+ episodes).
- Add tie‑breaking using positional priors or smarter sampling of hard words.

## 12) Reproducing our final baseline

- With 50,000‑word corpus, 2,000 games:
  - Success ~95%, Avg wrong ~1.98, Repeats ~0, Final Score ~170,220
- Commands:
  - `python -m src.evaluate --mode baseline --n_games 2000 --lives 6 --seed 42 --outdir plots --data_dir data --test data/corpus.txt`
  - `python -m src.evaluate --mode report --n_games 2000 --seed 42 --outdir plots --data_dir data --test data/corpus.txt`

## 13) Quick Demo Script (copy/paste)

```
# From project root (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.evaluate --mode baseline --n_games 1000 --lives 6 --seed 42 --outdir plots --data_dir data --test NOFILE
python -m src.evaluate --mode report --n_games 1000 --seed 42 --outdir plots --data_dir data --test NOFILE
```

## 14) VIVA

### Problem & Goal

- We built an AI that plays Hangman and tries to solve words with the fewest wrong guesses.
- It learns patterns from a list of English words and uses them to choose letters smartly.

### Data & Cleaning

- We load words from `data/corpus.txt`, lowercase them, keep only alphabetic letters, and remove duplicates.
- We also group words by length to quickly match the current puzzle pattern.

### HMM/Bigram Intuition

- We treat words as letter sequences and count how often each letter follows another (bigram counts).
- These counts are converted into probabilities with smoothing so that rare letters still have non‑zero probability.
- During a game, we use both the pattern match (which words fit `_ p p _ e`) and the bigram context (neighbors like previous/next letter) to score letters.

### Oracle Output

- The oracle returns a probability for each letter (a–z) given the current pattern and guessed letters.
- Already guessed letters are masked to zero so we never repeat guesses.

### Greedy Baseline Strategy

- At each step, we choose the letter with the highest oracle probability that we haven’t tried.
- If the guess is correct, letters are revealed; if not, we lose a life.
- We track wrong guesses and (ideally zero) repeated guesses.

### Reinforcement Learning (optional)

- The RL agent (DQN) sees a state vector: masked word one‑hot + guessed mask + oracle probabilities + lives.
- It learns Q‑values (usefulness) for each letter action and uses ε‑greedy exploration to try new letters.
- Rewards encourage correct letters, fast solves, and penalize wrong guesses or losing.

### Why probabilistic beats naïve frequency

- Global letter frequency ignores the word pattern; the oracle conditions on what we already know.
- This avoids silly guesses that don’t fit the visible positions and improves accuracy.

### Metrics we report

- Success Rate (% of solved words), Total Wrong Guesses, Total Repeated Guesses, and a Final Score.
- We also show plots: letter frequency and which letters cause most errors.

### Results summary (example from our run)

- On 2,000 games sampled from a 50k‑word corpus, we achieved about 95% success.
- Average wrong guesses were about 1.98 per game with zero repeats.
- The final score summarizes high success and low mistake counts.

### What the report shows

- The approach and dataset, the key metrics, and visualizations of errors and training (if RL used).
- The report is auto‑generated for consistency and quick sharing.

### Limitations & future work

- Bigram context is local; we can test trigrams or subword models to capture longer patterns.
- More training data and RL training time can further reduce mistakes.
- We could add heuristics (e.g., vowels first in early steps) blended with the oracle.
