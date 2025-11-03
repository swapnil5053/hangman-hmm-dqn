# 🧠 Intelligent Hangman — HMM + DQN Hybrid AI

> An AI that plays **Hangman intelligently** using probabilistic modeling (HMM) and reinforcement learning (DQN).  
> It learns contextual letter patterns from a 50,000-word corpus to guess words with minimal mistakes.

---

## 🚀 Overview
This project builds an **Intelligent Hangman Assistant** that:
- Learns **letter sequences** using a *Hidden Markov Model (HMM)*.
- Plays efficiently using a **Greedy Baseline Agent** guided by probability.
- Optionally trains a **Deep Q-Network (DQN)** agent that learns an optimal guessing policy through rewards.

🎯 **Goal:** Maximize success rate while minimizing wrong and repeated guesses.

---

## ⚙️ Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate         # (Windows)
# source .venv/bin/activate      # (Mac/Linux)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Place corpus.txt and test_words.txt inside the data/ folder
```

---

## 🧩 Run Baseline (HMM + Greedy Agent)

```bash
python -m src.evaluate --mode baseline --n_games 1000 --lives 6 --seed 42 --outdir plots --data_dir data
```

**Outputs**

* 📊 `data/baseline_results.csv`
* 🖼️ Plots in `plots/`
* 📄 Report → `Analysis_Report.pdf`

---

## 🤖 Train DQN (Optional Reinforcement Learning Agent)

```bash
python -m src.evaluate --mode dqn --episodes 2000 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay 0.995 --seed 42 --outdir plots --data_dir data
```

**Outputs**

* 🧮 `data/dqn_results.csv`
* 🧠 `data/dqn_agent.pth`
* 📉 Training curves in `plots/`
* 📘 Summary → `DQN_Summary_Report.pdf`

---

## 📈 Example Metrics

| Model                | Success Rate | Wrong Guesses | Repeated | Final Score |
| -------------------- | ------------ | ------------- | -------- | ----------- |
| HMM + Greedy         | **95.0%**    | 3956          | 0        | **170,220** |
| DQN Agent (2000 ep.) | 11.15%       | 11,672        | 0        | −36,060     |

**Scoring Formula:**
`Final Score = (SuccessRate × 2000) − (Wrong × 5) − (Repeated × 2)`

> **⚠️ Note:** If success rate is used as a fraction (e.g., `0.32` instead of `32`), the score calculation changes:  
> `0.32 × 2000 = 640` → `Final Score = 640 − 52,385 = −51,745`  
> The table above uses **percentage form** (e.g., `95` for 95%) for scoring.

---

## 🧠 Notebooks (Step-by-Step Demo)

| Notebook | Description |
|-----------|--------------|
| 🧾 01_data_and_hmm.ipynb | Data loading & HMM oracle training |
| 🧩 02_baseline_greedy.ipynb | Baseline agent evaluation |
| ⚡ 03_train_dqn.ipynb | DQN training and performance plots |
| 📊 04_generate_report.ipynb | Generates analysis & comparison reports |

---

## 🧭 Import Usage (APIs)

```python
from src import HMMOracle, HangmanEnv, DQNAgent
```

---

## 🗂️ Project Structure

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
├── Analysis_Report.pdf
└── README.md
```

---

## 💡 Insights

* The **HMM Bigram Model** captures English letter-to-letter dependencies.
* The **DQN Agent** learns through rewards to minimize mistakes and solve faster.
* Together, they blend **probabilistic reasoning** with **strategic decision-making**.

---

## 👥 Credits

**Developed by:** Swapnil Kumar  
**For:** *Intelligent Hangman Challenge*  
**Domain:** Machine Learning • Probabilistic Reasoning • Reinforcement Learning
