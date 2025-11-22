# Intelligent Hangman

An AI that learns to play Hangman by modeling letter patterns in English words. I combined an HMM (to learn which letters follow each other) with a simple greedy agent and an optional DQN to see if reinforcement learning could improve performance. Spoiler: the HMM approach completely dominated.

## The Idea

Most Hangman AIs just guess the most frequent letters. This one tries to be smarter by learning that certain letters follow each other. For example, if you see `_ING`, it knows a `T` is likely nearby because `TING`, `LING`, `RING`, etc. are common.

I built two approaches:
- **HMM + Greedy**: Uses a probabilistic model trained on 50k words to pick the best letter each turn
- **DQN**: A reinforcement learning agent trained to pick actions (letter guesses) based on game state

The HMM approach achieved 95% win rate. The DQN... didn't work as well (see results below).

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # Mac/Linux

pip install -r requirements.txt
```

Then add your datasets to the `data/` folder:
- `corpus.txt` — Training words for the HMM
- `test_words.txt` — Words to test on

## Running It

**HMM baseline (what actually works):**
```bash
python -m src.evaluate --mode baseline --n_games 1000 --lives 6 --seed 42 --outdir plots --data_dir data
```

Outputs a CSV with results, plots, and a PDF report.

**DQN training (optional, experimental):**
```bash
python -m src.evaluate --mode dqn --episodes 2000 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay 0.995 --seed 42 --outdir plots --data_dir data
```

## Results

The HMM approach crushed it. The DQN... didn't.

| Model | Win Rate | Wrong Guesses | Score |
|-------|----------|---------------|-------|
| HMM + Greedy | 95% | 3,956 | 170,220 |
| DQN (2000 episodes) | 11.15% | 11,672 | −36,060 |

The scoring formula is: `(Win% × 2000) − (Wrong × 5) − (Repeated × 2)`

I think the DQN struggled because the state space is huge (you need to track which letters have been guessed, the current pattern, remaining lives, etc.) and I didn't have enough training time or a good reward structure. The HMM's simplicity was actually its strength—it just learned the underlying patterns in English text.

## What's in Here

- `src/hmm_oracle.py` — The HMM model that learns letter transitions
- `src/baseline_greedy.py` — Simple greedy agent that uses HMM probabilities
- `src/dqn_agent.py` — PyTorch DQN implementation (experimental)
- `src/hangman_env.py` — Game environment
- `notebooks/` — Step-by-step walkthroughs if you want to understand the approach

## Project Structure

```
intelligent_hangman/
├── data/
│   ├── corpus.txt
│   ├── test_words.txt
├── src/
│   ├── utils.py
│   ├── hmm_oracle.py
│   ├── baseline_greedy.py
│   ├── hangman_env.py
│   ├── dqn_agent.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_data_and_hmm.ipynb
│   ├── 02_baseline_greedy.ipynb
│   ├── 03_train_dqn.ipynb
│   └── 04_generate_report.ipynb
├── plots/
└── README.md
```

## Key Takeaway

This was a good lesson in not over-engineering. A probabilistic model trained on real data outperformed a learned policy by a huge margin. Sometimes the simpler approach that understands your domain (English letter frequencies) beats something more complex (DQN) that needs way more tuning and data.

## Using It As a Library

```python
from src import HMMOracle, HangmanEnv, DQNAgent

oracle = HMMOracle(corpus_file="data/corpus.txt")
env = HangmanEnv(word_list="data/test_words.txt")

# Get best guess
next_letter = oracle.predict(current_pattern, guessed_letters)
```

---

Built to explore probabilistic modeling and RL on a fun problem.
