# HW: RL Fine-Tuning a LoRA-BERT Buzzer (REINFORCE)

## Big Picture

In the **prompt fine-tuning homework**, you did **supervised fine-tuning (SFT)** and implemented a **LoRA-adapted DistilBERT buzzer** (“LoRABERT”) that learns to **buzz vs. wait** from labeled training data.

In **this homework**, we keep the same buzzer model family (DistilBERT + your LoRA layers), but we now **fine-tune with reinforcement learning (RL)** using the simplest policy-gradient method: **REINFORCE**. 🎯

We intentionally keep the environment simple and stable:

- **Only the TF‑IDF guesser** is allowed (no DSPy / Ollama / other guessers).
- **Only LoRA parameters** should be trainable (base encoder frozen).
- RL is done with a **single-episode rollout per question** and a **return** computed from a reward function.

---

## What You Are Implementing: REINFORCE

The buzzer is treated as a stochastic **policy** over two actions:

- `WAIT` (0)
- `BUZZ` (1)

At each step (partial question), the policy samples an action and receives reward. We update parameters via:

\[
\nabla_\theta \; \mathbb{E}[R] \approx \sum_t (R - b)\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\]

Where:

- \(R\): episode return (question-level reward)
- \(b\): baseline to reduce variance (provided/scaffolded)
- optional entropy bonus encourages exploration

### References (pick one to cite in your writeup)
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, Policy Gradient / REINFORCE.
- A short REINFORCE overview: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html (policy gradients)

---

## You Must Reuse Your LoRA Code ✅

You should reuse your SFT homework implementation of:

- `LoRALayer`
- `LinearLoRA`

These are already present in the RL template and should look extremely similar to your previous LoRABERT setup. Reuse is part of the point: **small parameter count, fast updates, stable training**.

---

## Starter Files

You will work with:

- `rl_lorabert_buzzer.py` — RL fine-tuning buzzer using REINFORCE (**you edit this**)
- `tfidf_guesser.py` — TF‑IDF guesser (**you submit this**, usually reused from prior HW)
- `lorabert_buzzer.py` — SFT version (reference only; do not submit)

---

## What To Do (TODOs)

Open `rl_lorabert_buzzer.py` and complete the sections marked **TODO** (and optionally the **EXTRA CREDIT** parts).

### Required TODOs

1) **Reward function** (`_rl_reward`)
- Must reward:
  - correct buzz (scaled by “how early” you buzz)
  - penalize wrong buzz
  - small penalty for waiting (to prevent infinite waiting)
- The template includes an `expected_win_probability(...)` helper.

2) **Rollout stop condition** (`_rl_rollout`)
- When the policy chooses **BUZZ**, the episode should **end immediately** (stop rolling out additional steps).

3) **REINFORCE loss** (`_rl_loss`)
- Implement the policy gradient objective using:
  - log-probabilities of chosen actions
  - advantage \(A = R - b\)
  - optional entropy bonus (already scaffolded)

4) **Advantage computation** (`train`)
- Compute \(A = R - b\) (baseline is scaffolded).

---

## Running the Code

### Sanity run (small limit first)
```bash
python3 rl_lorabert_buzzer.py \
  --questions=../data/qanta.buzztrain.json.gz \
  --secondary_questions=../data/qanta.buzzdev.json.gz \
  --buzzer_guessers=Tfidf \
  --buzzer_type=rl_lorabert \
  --limit=100
```

Once it runs end-to-end, remove `--limit` (or increase it) for a real run.

---

## Only TF‑IDF Guesser (IMPORTANT)

For this RL homework:

✅ Allowed: `Tfidf`  
❌ Not allowed: DSPy/Ollama/other guessers

This keeps the RL environment consistent and prevents “moving target” behavior.

---

## Good Enough ✅

A “good enough” submission must **improve over the baseline** on the dev set:

- `buzzer_acc` improves over baseline (**acc = x**)
- `expected_wins` improves over baseline (**expected_wins = x**)
- `buzz_ratio` is reasonable / not degenerate (**buzz_ratio = X**)

Replace the `x` / `X` with your measured numbers (from logs / leaderboard).

---

## What To Submit (ONLY these)

Submit these two files:

1. `rl_lorabert_buzzer.py`
2. `tfidf_guesser.py`

### Models: “all TF‑IDF” (same convention as prior HW)
Your submission must include (or be compatible with) the same TF‑IDF artifacts used previously (e.g., pickled vectorizer/index/etc.). The key constraint is that the system must run using **TF‑IDF as the only guesser**.

---

## Extra Credit (+5 points total) ⭐

These are the extra-credit items explicitly mentioned in the code. You can earn up to **5 points** total by implementing one or more:

1) **Discount factor \(\gamma\)**  
   - Make `gamma` actually affect the return (e.g., discounted sum of step rewards).

2) **Optimizer improvements**  
   - Try different optimizers or tuned hyperparameters (LR, weight decay, schedules, gradient clipping).

3) **Reward shaping**  
   - Improve the reward signal (e.g., better wait penalty, scaling, normalization, stronger wrong-buzz penalty, etc.) while keeping it “honest”.

---

## Tips / Common Failure Modes

- **Always buzzing immediately** → increase wrong-buzz penalty, add/raise wait penalty, or add entropy annealing.
- **Never buzzing** → increase correct buzz reward, reduce wait penalty magnitude.
- **NaNs / instability** → lower LR, clip gradients, ensure logprobs/returns are finite.
- **Make sure episodes end on BUZZ** (otherwise you learn from inconsistent trajectories).

Good luck, and keep it simple: **REINFORCE + LoRA + TF‑IDF** 🚀
