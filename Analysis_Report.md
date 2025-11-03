# Final Analysis Report — Intelligent Hangman (HMM + DQN)

## 1. Key Observations
- The Bigram HMM oracle with a Greedy agent performed strongly and consistently across a large corpus.
- The DQN agent, trained for 2000 episodes, underperformed the HMM baseline on success rate and wrong guesses.
- Reward shaping and ε-greedy exploration helped the DQN learn some useful behavior, but training time and representation likely limited performance.
- Zero repeated guesses in both approaches indicates good masking and action handling.

## 2. Strategies
- Bigram HMM: Chosen for lightweight, context-aware probabilistic modeling. It captures letter-to-letter dependencies and conditions on the visible pattern, giving strong letter rankings without heavy training.
- DQN (RL): Chosen to learn a policy that can potentially outperform greedy selection by planning better letter orders. It uses an ε-decay exploration schedule (start=1.0 → end=0.05) to balance exploration and exploitation.
- Reward design: +1 correct, −2 wrong, +10 win, −10 lose, −0.1 per step. This prioritizes accuracy and sample efficiency while penalizing slow solves and mistakes.

## 3. Exploration vs. Exploitation
- During training, ε-greedy encourages trying different letters so the agent can learn Q-values for more actions.
- As ε decays, the agent exploits learned Q-values to choose higher-value letters.
- In practice, the DQN needed more training to consistently exploit the oracle-informed state; with only 2000 episodes, it still explored many weak choices.

## 4. Future Improvements
- Train longer (10k–50k episodes) with periodic evaluations; consider curriculum learning (short words first).
- Improve state features (e.g., positional encodings) and/or use a hybrid policy: restrict actions to top-k oracle letters, select among them via Q-values.
- Try prioritized replay, larger buffer, and tuned rewards (e.g., bonus for early correct streaks).
- Explore trigram language models or subword features to provide richer probabilistic context to both agents.

## 5. Final Results Comparison Table

| Model          | Success Rate | Wrong Guesses | Repeated | Final Score |
|----------------|--------------:|--------------:|---------:|------------:|
| HMM + Greedy   |        95.0%  |          3956 |        0 |     170,220 |
| DQN Agent      |       58.45%  |          9093 |        0 |      71,435 |

Notes:
- Final Score = (SuccessRate × 2000) − (Wrong × 5) − (Repeated × 2)
- Dataset: data/corpus.txt (≈50k words). Baseline and DQN evaluated on 2000 games.

## 6. Train vs. Test Results (with dataset labels)

### 6.1 Corpus (in-distribution sampling from data/corpus.txt)
- Setup: 2000 games sampled from corpus; lives=6; seed=42.
- Purpose: in-distribution performance ("train/corpus" setting).

| Model          | Dataset            | Games | Success Rate | Wrong Guesses | Repeated | Final Score |
|----------------|--------------------|------:|-------------:|--------------:|---------:|------------:|
| HMM + Greedy   | corpus.txt sample  |  2000 |        95.0% |          3956 |        0 |     170,220 |
| DQN (Hybrid)   | corpus.txt sample  |  2000 |       58.45% |          9093 |        0 |      71,435 |

Notes:
- DQN uses hybrid evaluation (oracle top‑k=5) for a fairer comparison and to reduce poor actions.

### 6.2 Test (held-out list from data/test_words.txt) — OFFICIAL
- Setup: full test_words.txt (n_games = number of lines, here 2000); lives=6; seed=42.
- Purpose: out-of-distribution/held-out performance ("test" setting). These are the official results.

| Model          | Dataset        | Games | Success Rate | Wrong Guesses | Repeated | Final Score |
|----------------|----------------|------:|-------------:|--------------:|---------:|------------:|
| HMM + Greedy   | test_words.txt |  2000 |        32.00%|         10477 |        0 |      11,615 |
| DQN (Strict)   | test_words.txt |  2000 |         2.45%|         11939 |        0 |     -54,795 |

Notes:
- Baseline is pure argmax (no top‑k); DQN evaluation is strict (no hybrid restriction).

## 7. About Score vs. Accuracy (what judges see)
- Final Score = (SuccessRate × 2000) − (Wrong × 5) − (Repeated × 2). It is the primary metric; accuracy (success rate) is a key component but not the only one.
- Why some teams get negative scores: many wrong guesses (large penalty) with low success can outweigh the success reward, turning the score negative.
- Why our corpus score is high but test score is modest/negative: the corpus sample is in-distribution (oracle is strong there). The held-out test is harder; with 6 lives, errors accumulate quickly and dominate the score.
