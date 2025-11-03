# Intelligent Hangman — DQN Overview (Simple Guide)

This guide explains the Reinforcement Learning (DQN) part of the Hangman agent in the same simple, sectioned style as the baseline overview.

## 1) What is the DQN part?
- A learning agent that tries to pick letters smarter over time by maximizing long‑term reward.
- Goal: fewer wrong guesses, faster solves, comparable or better performance than the HMM+Greedy baseline.

## 2) Data & Outputs
- Data comes from `data/corpus.txt` (50k words). We train and evaluate on sampled words from this corpus.
- Outputs for DQN training:
  - `data/dqn_results.csv` (episode, reward, success, epsilon)
  - `plots/dqn_reward.png`, `plots/dqn_success_rate.png` (training curves)
  - `data/dqn_agent.pth` (saved model weights)

## 3) State, Action, and Model
- **State vector** (concatenated):
  1. Masked word one‑hot (max_len × 26, flattened).
  2. Guessed letters mask (26‑dim binary).
  3. HMM oracle probabilities (26‑dim) for letters a–z.
  4. Lives left (1‑dim, normalized).
- **Actions**: 26 letters (a–z).
- **Model (QNet)**: MLP with layers 256 → 256 (ReLU), output 26 Q‑values (one per letter).

## 4) Rewards (why the agent learns)
- +1 correct guess, −2 wrong guess.
- +10 win (word solved), −10 lose (no lives left).
- −0.1 per step to solve faster.

## 5) Training Setup
- Method: DQN with replay buffer and target network.
- Exploration: ε‑greedy (epsilon_start=1.0 → epsilon_end=0.05 using decay=0.995).
- Optimizer: Adam (lr=1e‑3). Lives per game: 6.
- Episodes: 2000 in our run.

## 6) How to run
```
# From project root (venv active)
python -m src.evaluate --mode dqn --episodes 2000 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay 0.995 --seed 42 --lives 6 --outdir plots --data_dir data --test data/corpus.txt
python -m src.evaluate --mode dqn_eval --n_games 2000 --lives 6 --seed 42 --outdir plots --data_dir data --test data/corpus.txt
python -m src.evaluate --mode dqn_summary --outdir plots --data_dir data
```

## 7) Evaluation & Metrics
- After training, we evaluate with ε=0 (greedy policy) on 2000 games.
- Metrics: Success Rate (%), Total Wrong, Total Repeated, Final Score.
- Final Score = (SuccessRate × 2000) − (Wrong × 5) − (Repeated × 2)

## 8) Results (our run)
### Corpus-sampled (data/corpus.txt), 2000 games
- DQN (Hybrid top‑k=5 eval): Success 58.45%, Wrong 9,093, Repeated 0, Final Score 71,435.0
- Baseline HMM+Greedy: Success 95.0%, Wrong 3,956, Repeated 0, Final Score 170,220.0

### Test set (data/test_words.txt), full set (2000 words)
- DQN (Strict eval): Success 13.85%, Wrong 11,529, Repeated 0, Final Score −29,945.0
- Baseline HMM+Greedy (Top‑k expected‑reveals): Success 32.90%, Wrong 10,423, Repeated 0, Final Score 13,685.0

## 9) What this means
- The HMM+Greedy baseline is very strong out‑of‑the‑box for Hangman.
- With 2000 episodes, DQN did not catch up; it needs more training and tuning to leverage the oracle effectively.

### Score vs Accuracy (important)
- Judges compute Final Score = (SuccessRate × 2000) − (Wrong × 5) − (Repeated × 2).
- Accuracy (success rate) matters, but heavy penalties for wrong guesses can make scores negative on the held‑out test.
- This explains why many teams’ DQN scores are negative on test even if accuracy is non‑zero.

## 10) Why DQN under‑performed (likely reasons)
- Short training relative to the vocabulary size and pattern diversity.
- The MLP must learn positional patterns from a flattened state; that takes more data/time.
- ε‑greedy exploration still tries poor letters early; more structure can help.

## 11) How to improve DQN (practical tips)
- Train longer: 10k–50k episodes; evaluate periodically.
- Curriculum: start with easy/short words, then ramp up.
- Prioritized replay; larger buffer.
- Stronger or tuned rewards (e.g., bonus for streaks, penalty for repeats).
- Better state encoding (positional signals) or hybrid policy:
  - Limit actions to top‑k letters suggested by the HMM oracle and pick among them with QNet.

## 12) Files you’ll use
- Training logs/data: `data/dqn_results.csv`
- Curves: `plots/dqn_reward.png`, `plots/dqn_success_rate.png`
- Weights: `data/dqn_agent.pth`
- Summary PDF: `DQN_Summary_Report.pdf`

## 13) Takeaway
- Use HMM+Greedy for a strong baseline and demo stability.
- Use DQN as a bonus feature: show learning curves, discuss improvements, and plan for larger‑scale training.
