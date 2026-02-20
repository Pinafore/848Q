# HW: RL Fine-Tuning a LoRA-BERT Buzzer (REINFORCE)

## Big Picture

In the **prompt fine-tuning homework**, you did **supervised fine-tuning (SFT)** and implemented a **LoRA-adapted DistilBERT buzzer** (“LoRABERT”) that learns to **buzz vs. wait** from labeled training data.

In **this homework**, we keep the same buzzer model family (DistilBERT + your LoRA layers), but we now **fine-tune with reinforcement learning (RL)** using the simplest policy-gradient method: **REINFORCE**.

We intentionally keep the environment simple and stable:

- **Only the TF‑IDF guesser** is allowed (reducing evaluation complexity).
- finetuning **LoRA finetuning parameters**.
- RL is done with a **single-episode rollout per question** and a **return** computed from a reward function.

## What You Are Implementing: REINFORCE

The buzzer is treated as a stochastic **policy** over two actions:

- `WAIT` (0)
- `BUZZ` (1)

At each step (partial question), the policy samples an action and receives reward. We update parameters via:

$$
\nabla_\theta \; \mathbb{E}[R] \approx \sum_t (R - b)\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

Where:

- $$R$$: episode return (question-level reward)
- $$b$$: baseline to reduce variance (provided/scaffolded)
- optional entropy bonus encourages exploration

## What You Have To Do


You will work with `rl_lorabert_buzzer.py` for RL fine-tuning buzzer using REINFORCE

where you can use your previous homework implementation of:

- `LoRALayer`
- `LinearLoRA`

Here are functions you will need to implement:

* **Reward function** (`_rl_reward`)
  - Must reward:
    - correct buzz (scaled by “how early” you buzz)
    - penalize wrong buzz
    - small penalty for waiting (to prevent infinite waiting)
  - The template includes an `expected_win_probability(...)` helper.

* **Rollout stop condition** (`_rl_rollout`)
  - When the policy chooses **BUZZ**, the episode should **end immediately** (stop rolling out additional steps).

* **REINFORCE loss** (`_rl_loss`)
  - Implement the policy gradient objective using:
    - log-probabilities of chosen actions
    - advantage \(A = R - b\)
    - optional entropy bonus (already scaffolded)

* **Advantage computation** (`train`)
  - Compute \(A = R - b\) (baseline is scaffolded).

After you done implementation, you can test your training code
```bash
python3 rl_lorabert_test.py
```

and then train your model when everything works!

```bash
python3 rl_lorabert_buzzer.py \
  --questions=../data/qanta.buzztrain.json.gz \
  --secondary_questions=../data/qanta.buzzdev.json.gz \
  --buzzer_guessers=Tfidf \
  --buzzer_type=rl_lorabert \
  --limit=100
```

Once it runs end-to-end perfectly, remove `--limit` (or increase it)

## Good Enough 

A good enough submission is simple, just have `Expected Win` >= x on training, and `Expected Win` >= x on eval. You can tinker our reward when the model buzzes correctly, play with penalties, or train with more epochs, anything that you can to improve the metric!

## What To Submit (ONLY these)

Submit following files on Gradescope:

1. `rl_lorabert_buzzer.py` ; don't forget to set your best params in this file!
2. `tfidf_guesser.py`
3. `lorabert_rl.model` 
4. Your `TfidfGuesser.answers.pkl`, `TfidfGuesser.questions.pkl`, `TfidfGuesser.tfidf.pkl` and `TfidfGuesser.vectorizer.pkl` (a tfidf guesser)

## Extra Credit

These are the extra-credit items explicitly mentioned in the code. You can earn up to **5 points** by implementing one or more:

* rank top 10 in the leaderboard by EW!

* Additional reward signals 

## FAQ


