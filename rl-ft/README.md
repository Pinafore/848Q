## Big Picture

In the **prompt fine-tuning homework**, you did **supervised fine-tuning (SFT)** and implemented a **LoRA-adapted DistilBERT buzzer** (“LoRABERT”) that learns to **buzz vs. wait** from labeled training data.

In **this homework**, we keep the same buzzer model family (DistilBERT + your LoRA layers), but we now **fine-tune with reinforcement learning (RL)** using the simplest policy-gradient method: **REINFORCE**.

We intentionally keep the environment simple and stable:

- **only TF‑IDF guesser** is allowed at first (reducing evaluation complexity).
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

* **Reward function** (`_rl_reward`) Must reward:
  - correct buzz (scaled by “how early” you buzz)
  - penalize wrong buzz
  - small penalty for waiting (to prevent infinite waiting)
  - The template includes an `expected_win_probability(...)` helper.

* **Rollout stop condition** (`_rl_rollout`)
  - When the policy chooses **BUZZ**, the episode should **end immediately** (stop rolling out additional steps).

* **REINFORCE loss** (`_rl_loss`) Implement the policy gradient objective using:
  - log-probabilities of chosen actions
  - advantage $$A$$
  - entropy reg 

* **Advantage computation** (`train`)
  - Compute $$A = R - b$$

After you done implementation, you can test your training code,
```bash
python3 rl_lorabert_test.py
```
which you will see all tests passed like below if your model RL right!

```bash
(rl-ft-env) root@MSI:/home/wwongkam/848Q/rl-ft# python3 rl_lorabert_test.py
Setting up logging
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 89/89 [00:00<00:00, 2325.76it/s, Materializing param=classifier.weight]
.Setting up logging
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 89/89 [00:00<00:00, 2608.34it/s, Materializing param=classifier.weight]
.Setting up logging
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 89/89 [00:00<00:00, 2529.70it/s, Materializing param=classifier.weight]
.Setting up logging
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 89/89 [00:00<00:00, 2444.76it/s, Materializing param=classifier.weight]
.Setting up logging
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 89/89 [00:00<00:00, 2646.87it/s, Materializing param=classifier.weight]
.Setting up logging
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 89/89 [00:00<00:00, 2920.12it/s, Materializing param=classifier.weight]
.
----------------------------------------------------------------------
Ran 6 tests in 6.752s

OK
```

Then, you will need to create a folder `models` before training. Once you train end-to-end and see your first result, you may adjust your RL train params (e.g., num epochs, correct/incorrect buzz reward) and increase more training data using `--limit`.

```bash
(rl-ft-env) root@MSI:/home/wwongkam/848Q/rl-ft# python3 rl_lorabert_buzzer.py \
  --questions=../data/qanta.buzztrain.json.gz \
  --secondary_questions=../data/qanta.buzzdev.json.gz \
  --buzzer_guessers=Tfidf \
  --buzzer_type=rl_lorabert \
  --limit=100
Setting up logging
Loading buzzer
INFO:root:Buzzer using run length 100
INFO:root:Initializing guesser of type Tfidf
INFO:root:Loading 596496 questions and 596496 answers
INFO:root:Loading 596496 questions and 596496 answers
INFO:root:Adding Tfidf to Buzzer (total guessers=1)
Initializing features: []
dataset: ../data/qanta.buzztrain.json.gz
INFO:root:Loading questions from ../data/qanta.buzztrain.json.gz
INFO:root:Read 100 questions
INFO:root:Loading questions from ../data/qanta.buzzdev.json.gz
INFO:root:Read 100 questions
INFO:root:Generating runs of length 100 from 100 questions
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 100582.83it/s]
INFO:root:Generating guesses for 807 new question
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 807/807 [02:40<00:00,  5.03it/s]
INFO:root:Generating runs of length 100 from 100 questions
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 80412.27it/s]
INFO:root:Generating guesses for 810 new question
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 810/810 [02:42<00:00,  4.97it/s]
[RL] Epoch 1/1: avg_return=1.3817, avg_steps=4.42
[RL] Epoch 1: dev expected_wins=0.0310
Questions Right: 8 (out of 100) Accuracy: 0.62  Buzz ratio: 0.04 Expected Wins: 0.031000
```

**Hint:** if you train multiple epochs, we suggest implementing to save your best model (with most return or with most dev EW)

After the full training is done, you can again evaluate your RL model on dev dataset.
```bash
python3 eval.py \
  --evaluate buzzer \
  --questions ../data/qanta.buzzdev.json.gz \
  --buzzer_guessers=Tfidf \
  --buzzer_type=rl_lorabert \
  --num_guesses 1
  --limit=100
```

## Good Enough 

A good enough submission is simple, just have `Expected Win` >= 0.05 on eval (you can try when your `Expected Win` >= 0.05 on dev when training). You can change our reward when the model buzzes correctly, play with penalties, or train with more epochs, anything that you can to improve the metric!

Note: we can lower EW baseline, just let us know if this baseline is too difficult and require too much of GPU/CPU computation!

## What To Submit 

Submit following files on Gradescope:

1. `rl_lorabert_buzzer.py` ; don't forget to set your best params in this file!
2. `tfidf_guesser.py`
3. `parameters.py` ; since you may set params differently than our defaults
4. `lorabert_rl.model` 
5. Your `TfidfGuesser.answers.pkl`, `TfidfGuesser.questions.pkl`, `TfidfGuesser.tfidf.pkl` and `TfidfGuesser.vectorizer.pkl` (a tfidf guesser)

## Extra Credit

You can earn up to **5 points** by acheive one of below and attach `analysis.pdf` to state what you've done and why.

* rank top 10 in the leaderboard by EW!

* implement additional reward signals than EW * `correct_buzz_reward`. 

* use gpt-cached guesser (gpr guesser) which is another light-weight guesser that we allow. 

Note, since gpr guesser requires a cache `../models/buzzdev_gpr_cache` to eval dev dataset, the internal eval on dev during **training** wouldn't work and always `WAIT`. You will need to do a full train, then eval later.

```bash
(rl-ft-env) root@MSI:/home/wwongkam/848Q/rl-ft# python3 rl_lorabert_buzzer.py   --questions=../data/qanta.buzztrain.json.gz   --secondary_questions=../data/qanta.buzzdev.json.gz --buzzer_guessers=gpr --gpr_guesser_filename=../models/buzztrain_gpr_cache --buzzer_type=rl_lorabert   --limit=100 --rl_lorabert_buzzer_rl_epochs 5 
Setting up logging
Loading buzzer
INFO:root:Buzzer using run length 100
INFO:root:Initializing guesser of type gpr
INFO:root:Loading gpr guesser
INFO:root:125288 entries added to cache
INFO:root:125288 entries added to cache
INFO:root:Adding gpr to Buzzer (total guessers=1)
Initializing features: []
dataset: ../data/qanta.buzztrain.json.gz
INFO:root:Loading questions from ../data/qanta.buzztrain.json.gz
INFO:root:Read 100 questions
INFO:root:Loading questions from ../data/qanta.guessdev.json.gz
INFO:root:Read 100 questions
INFO:root:Generating runs of length 100 from 100 questions
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 87527.21it/s]
INFO:root:Generating guesses for 807 new question
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 807/807 [00:00<00:00, 249212.44it/s]
INFO:root:Generating runs of length 100 from 100 questions
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 86107.66it/s]
INFO:root:Generating guesses for 807 new question
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 807/807 [00:00<00:00, 511763.43it/s]
[RL] Epoch 1/5: avg_return=1.9464, avg_steps=4.06
[RL] Epoch 1: dev expected_wins=0.0000
Questions Right: 0 (out of 100) Accuracy: 0.89  Buzz ratio: -0.06 Expected Wins: 0.000000
[RL] Epoch 2/5: avg_return=4.6468, avg_steps=5.06
[RL] Epoch 2: dev expected_wins=0.0000
Questions Right: 0 (out of 100) Accuracy: 0.00  Buzz ratio: -0.50 Expected Wins: 0.000000
[RL] Epoch 3/5: avg_return=5.9407, avg_steps=4.59
[RL] Epoch 3: dev expected_wins=0.0000
Questions Right: 0 (out of 100) Accuracy: 0.00  Buzz ratio: -0.50 Expected Wins: 0.000000
[RL] Epoch 4/5: avg_return=5.9839, avg_steps=5.43
[RL] Epoch 4: dev expected_wins=0.0000
Questions Right: 0 (out of 100) Accuracy: 0.00  Buzz ratio: -0.50 Expected Wins: 0.000000
[RL] Epoch 5/5: avg_return=6.0071, avg_steps=4.91
[RL] Epoch 5: dev expected_wins=0.0000
Questions Right: 0 (out of 100) Accuracy: 0.00  Buzz ratio: -0.50 Expected Wins: 0.000000
```

Evaluation examples:

```bash
(rl-ft-env) root@MSI:/home/wwongkam/848Q/rl-ft# python3 eval.py   --evaluate buzzer   --questions ../data/qanta.buzzdev.json.gz   --buzzer_guessers=gpr   --buzzer_type=rl_lorabert   --num_guesses 1 --limit 100 --gpr_guesser_filename=../models/buzzdev_gpr_cache
```

```bash
Setting up logging
Loading buzzer
Initializing features: []
dataset: ../data/qanta.buzzdev.json.gz
[-1, -1, -1, -2, -2, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, -1, -1, -1, -1, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, -1, -1, -1, -1, -2, -2, -2, -2, -1, -1, -1, -1, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, -1, -1, -2, -2, -2, -2, 0.38749999999999984, 0.38749999999999984, -1, -1, -1, -1, -2, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, -1, -1, -1, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, -1, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2, -1, -1, -1, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, -1, -1, -1, -2, -2, -2, -2, -2, -1, -1, -1, 0.38749999999999984, -2, 0.38749999999999984, 0.38749999999999984, 0.38749999999999984, -1, -1, -1, -1]
timid 0.34
===================

               guess: Cauldron
              answer: Cauldrons
                  id: 93150
                buzz: 0
            category: Literature
         subcategory: Literature European
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 5541483aea23cc9417e9baf9
              qdb_id: None
             dataset: protobowl
            qanta_id: 93256
       tokenizations: [[0, 175], [176, 391], [392, 562], [563, 674], [675, 769]]
       answer_prompt: or Anatole Thibault
            gameplay: True
                fold: buzzdev
                 run: -0.22 [SEP] Cauldron [SEP] One of these objects is owned by a giant
                      whose wife births a fully armed son every six weeks. That owner of one
                      of these objects, who escapes a plot to roast him alive in an iron
                      house, is named Llasar Llaes Gyfnewid. Along with a staff and a
                      platter, Bran gives one to Matholwch as reparations, which
--------------------
               guess: Henri II de Montmorency
              answer: Louis_XIII_of_France
                  id: 93147
                buzz: 0
            category: History
         subcategory: History World
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 55414839ea23cc9417e9badb
              qdb_id: None
             dataset: protobowl
            qanta_id: 93226
       tokenizations: [[0, 134], [135, 199], [200, 300], [301, 429], [430, 521], [522, 690], [691, 792]]
       answer_prompt: or seraglio; or serail; or saray-i-duhteran; prompt on "Topkapi
                      palace" or "Ottoman palace" or similar answers; prompt on "women" or
                      "women's quarters"
            gameplay: True
                fold: buzzdev
                 run: -0.06 [SEP] Henri II de Montmorency [SEP] During this king's reign,
                      his general Henri II de Montmorency beat the Spanish at the Battle of
                      Veillane
=================
aggressive 0.25
===================

               guess: Zero-grade
              answer: 
                  id: 93153
                buzz: 1
            category: Literature
         subcategory: Literature European
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 5541483eea23cc9417e9bb1f
              qdb_id: None
             dataset: protobowl
            qanta_id: 93294
       tokenizations: [[0, 192], [193, 240], [240, 384], [384, 385], [386, 567], [567, 568], [569, 680], [680, 798]]
       answer_prompt: 
            gameplay: True
                fold: buzzdev
                 run: -0.71 [SEP] Zero-grade [SEP] In Proto-Indo-European studies, this kind
                      of ablaut contrasts with both the "e-grade" and "o-grade" varieties.
                      In English syntax, this form of complementizer is inherent to the
                      sentence "I think they like me." This type of "derivation" is
                      exemplified by using a noun such as "pen" as a verb, as in "I penned
                      it." In the Chomsky hierarchy, unrestricted grammars are also called
                      "Type-[this]". Arabic and Hebrew use this type of copula in sentences
                      lacking a word for "to be." In linguistics, this term
--------------------
               guess: Gaussian Integers
              answer: Perfect_Numbers
                  id: 93144
                buzz: 1
            category: Mythology
         subcategory: Literature Classical
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 55414838ea23cc9417e9bac0
              qdb_id: None
             dataset: protobowl
            qanta_id: 93199
       tokenizations: [[0, 127], [128, 219], [220, 326], [327, 438], [439, 527], [528, 631], [632, 681], [682, 771]]
       answer_prompt: 
            gameplay: True
                fold: buzzdev
                 run: -0.65 [SEP] Gaussian Integers [SEP] For any natural number n, there
                      exists only one of these numbers that can be expressed in the form
                      "n-cubed plus 1". Kanold was the first to show that the amount of
                      these numbers below a given integer n had an asymptotic form of
                      little-O of the square root of n. With the exception of the smallest
                      of
=================
best 0.30
===================

               guess: Louis XIII of France
              answer: Louis_XIII_of_France
                  id: 93147
                buzz: 1
            category: Religion
         subcategory: History European
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 55414839ea23cc9417e9bae2
              qdb_id: None
             dataset: protobowl
            qanta_id: 93233
       tokenizations: [[0, 98], [99, 108], [109, 251], [252, 346], [347, 511], [512, 665], [666, 762]]
       answer_prompt: or Russian Empire; or Russian Federation; or Rossiya; or Rossiyskaya
                      Imperiya; or Rossiyskaya Federatsiya; do not accept "Soviet Union" or
                      "USSR"
            gameplay: True
                fold: buzzdev
                 run: -0.02 [SEP] Louis XIII of France [SEP] During this king's reign, his
                      general Henri II de Montmorency beat the Spanish at the Battle of
                      Veillane and helped Charles Gonzaga, the Duke of Nevers [nuh-VAIR],
                      secure rule over Mantua. The Counts of MontrÃ©sor and Soissons plotted
                      with this king's brother Gaston in a plot to overthrow him. Jean
                      Guiton was mayor of a city that resisted this man's rule, holding out
                      for 14 months until the signing of the Peace of Alais. Concino Concini
                      advised the mother of this king, who acted as his regent until
--------------------
               guess: Hydrogenation
              answer: Hydrogenation
                  id: 93154
                buzz: 1
            category: Science
         subcategory: Science Chemistry
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 5541483fea23cc9417e9bb30
              qdb_id: None
             dataset: protobowl
            qanta_id: 93311
       tokenizations: [[0, 165], [166, 263], [264, 511], [512, 696], [697, 813]]
       answer_prompt: 
            gameplay: True
                fold: buzzdev
                 run: -0.04 [SEP] Hydrogenation [SEP] One reaction of this type reacts
                      alpha, beta-unsaturated carbonyls with Hantzsch esters under amine
                      catalysis. Discoverers of an asymmetric version of this reaction used
                      in the industrial synthesis of L-DOPA from an achiral arene won part
                      of the 2001 Nobel Prize in Chemistry. That asymmetric form of this
                      reaction can be catalyzed by ruthenium-BINAP complexes developed by
                      Noyori. A square-planar tris(triphenylphosphine) rhodium(I) complex
                      was developed in 1966 to homogeneously catalyze this reaction; that is
                      Wilkinson's catalyst. When this reaction is incomplete, it can result
                      in cis-trans isomerization, and thus its "partial" form is responsible
                      for the production of trans fats. For 10 points,
=================
waiting 0.11
===================

               guess: Louis XIII of France
              answer: Louis_XIII_of_France
                  id: 93147
                buzz: 0
            category: Science
         subcategory: Science Chemistry
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 55414839ea23cc9417e9bae0
              qdb_id: None
             dataset: protobowl
            qanta_id: 93231
       tokenizations: [[0, 128], [129, 230], [231, 356], [357, 569], [570, 719], [720, 800]]
       answer_prompt: or binding to a receptor; prompt on protein-protein interaction or PPI
                      or similar answers; anti-prompt on protein activation or inhibition or
                      specific cases of binding; prompt on drug activity or related answers;
                      do not accept "forming a bond", "bonding", or any answer about
                      covalent bonding
            gameplay: True
                fold: buzzdev
                 run: -0.02 [SEP] Louis XIII of France [SEP] During this king's reign, his
                      general Henri II de Montmorency beat the Spanish at the Battle of
                      Veillane and helped Charles Gonzaga, the Duke of Nevers [nuh-VAIR],
                      secure rule over Mantua. The Counts of MontrÃ©sor and Soissons plotted
                      with this king's brother Gaston in a plot to overthrow him. Jean
                      Guiton
--------------------
               guess: The Name of the Rose
              answer: The_Name_of_the_Rose
                  id: 93142
                buzz: 0
            category: Science
         subcategory: Science Other
          tournament: ACF Regionals
          difficulty: College
                year: 2015
            proto_id: 55414838ea23cc9417e9bab3
              qdb_id: None
             dataset: protobowl
            qanta_id: 93186
       tokenizations: [[0, 72], [73, 271], [272, 364], [365, 529], [530, 665], [666, 763]]
       answer_prompt: or solar wind
            gameplay: True
                fold: buzzdev
                 run: -0.00 [SEP] The Name of the Rose [SEP] The narrator of this novel
                      becomes fascinated by the story of Margaret and Dolcino after a
                      lecture on love by Ubertino. To prove his skill, a character in this
                      novel discerns the location, appearance, and name of the horse
                      Brunellus without having ever seen it. A man in this work has a vision
                      of the plot of the Cena Cypriani before discovering how to open a
                      mirror and enter the finis Africae. After
=================
Questions Right: 30 (out of 100) Accuracy: 0.41  Buzz ratio: 0.17 Expected Wins: 0.116250
```

## FAQ


