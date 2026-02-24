import logging
import random
from functools import partial
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import torch
import math
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from buzzer import Buzzer, BuzzerParameters
from lorabert_buzzer import LoRABertBuzzer

@dataclass
class Step:
    text: str
    label: int              # 1 if guess correct; 0 otherwise
    guess: str
    answer: str
    qid: int
    run_length: int
    q_max_length: int


def expected_win_probability(percentage: float) -> float:
    """
    Same function as eval.expected_win_probability (kept local to avoid import cycles).
    Source: https://arxiv.org/pdf/1904.04792
    """
    t = percentage
    return 1 + 0.0775*t - 1.278*t**2 + 0.588*t**3


class LoraBertRLParameters(BuzzerParameters):
    """
    Parameters for RL fine-tuning of LoRA-augmented DistilBERT buzzer.
    """
    def __init__(self, customized_params=None):
        super().__init__()
        self.name = "rl_lorabert_buzzer"
        if customized_params:
            self.params += customized_params
        else:
            self.params += [
                ("rank", int, 16, "Rank of LoRA adaptation"),
                ("base_model", str, "distilbert-base-uncased", "HF model to adapt"),
                ("alpha", float, 1.0, "LoRA alpha"),
                ("rl_lr", float, 5e-4, "RL optimizer learning rate (LoRA params only)"),
                ("rl_epochs", int, 1, "Number of RL epochs"),
                ("gamma", float, 1.0, "Discount factor (1.0 = undiscounted)"),
                ("correct_buzz_reward", float, 5.0, "Per-step reward for correctly buzzing"),
                ("wait_penalty", float, -0.02, "Per-step penalty for choosing WAIT"),
                ("wrong_buzz_penalty", float, -1.0, "Penalty for BUZZ when label=0"),
                ("entropy_coef", float, 0.01, "Entropy bonus coefficient"),
                ("baseline_beta", float, 0.9, "EMA baseline smoothing (0.9 is typical)"),
                ("max_steps", int, 9999, "Safety cap for steps per question"),
                ("seed", int, 0, "Random seed"),
            ]
    def __setitem__(self, key, value):
        assert hasattr(self, key), "Missing %s, options: %s" % (key, dir(self))
        setattr(self, key, value)
           
    def set_defaults(self):
        for parameter, _, default, _ in self.params:
            name = "%s_%s" % (self.name, parameter)
            setattr(self, name, default)    


def initialize_base_model(helper_function=AutoModelForSequenceClassification,
                          model_name="distilbert-base-uncased"):
    """
    Initialize a BERT model and corresponding tokenizer.

    Args:
        helper_function: The huggingface function that returns a BERT model.
        model_name: The name of the BERT model to use.
    """

    model = helper_function.from_pretrained(model_name, num_labels=2)

    # Freeze all parameters by default (we'll unfreeze the classification head).
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classification head so the policy can learn a sensible boundary.
    for head_name in ["pre_classifier", "classifier"]:
        if hasattr(model, head_name):
            for p in getattr(model, head_name).parameters():
                p.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        """
        Initialize a LoRA with two weight matrices whose product is the same as the original parameter matrix.
        """
        super().__init__()

        self.A = None
        self.B = None
        self.alpha = 0

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Complete the initialization of the two weight matrices
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear layer's original result, then add the low-rank delta
        """
        assert x.shape[-1] == self.in_dim, "Input dimension %s does not match input dimension %i" % (str(x.shape), self.in_dim)

        if len(x.shape) == 1:
            delta = torch.zeros(self.out_dim)
            output_dimension = torch.Size([self.out_dim])
        else:
            delta = torch.zeros((x.shape[0], self.in_dim))
            output_dimension = torch.Size((x.shape[0], self.out_dim))

        # Compute the low-rank delta

        return delta


class LinearLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        """
        Initialize a Linear layer with LoRA adaptation.
        """

        super().__init__()
        self.linear = linear

        # Initialize the LoRA layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adatpation.
        """
        result = self.linear(x)

        # Add the LoRA delta
        return result

# TODO(jbg): Get rid of the hardcoded modules so that it generalizes to other models
def add_lora(model: torch.nn.Module, rank: int, alpha: float, 
             modules_to_adapt: Dict[str, List[str]] = {"attention": ["q_lin", "k_lin", "v_lin", "out_lin"], "ffn": ["lin1", "lin2"]}):
    """
    Add LoRA layers to a PyTorch model.  

    Args:
        model: The PyTorch model to adapt.
        rank: The rank of the LoRA matrices.
        alpha: The scaling factor for the LoRA matrices.
        modules_to_adapt: The key of the dictionary is the model component to adapt (e.g., "attention" or "ffn"), and the values are specific linear layers in that component to adapt.  Anything in this dictionary will be adapted, but anything else will remain frozen.
    """
    

    return model


class LoRABertRLBuzzer(LoRABertBuzzer):
    def __init__(self, filename: str, run_length: int, num_guesses: int, cfg):
        super().__init__(filename, run_length, num_guesses)
        self._pipeline = None
        self.model = None
        self.tokenizer = None
        self.cfg = cfg

    def initialize_model(self, model_name, rank, alpha):
        """
        Initialize the model and add LoRA layers.
        """

        self.model, self.tokenizer = initialize_base_model(model_name=model_name)
        add_lora(self.model.distilbert.transformer, rank, alpha)

    def add_data(self, questions):
        """
        We don't have features, so these become noops
        """
        
        None

    def build_features(self, history_length=0, history_depth=0):
        """
        We don't have features, so these become noops
        """
        
        None

    def _build_steps_from_questions(self, questions, answer_field="page") -> List[Step]:
        """
        Convert raw question json, get all the runs, into a list of Step objects.
        """
        from eval import rough_compare

        metadata, answers, runs = self._clean_questions(questions, self.run_length, answer_field)

        # Build guesser outputs (same as original LoRABertBuzzer)
        all_guesses = {}
        for guesser in self._guessers:
            all_guesses[guesser] = self._guessers[guesser].batch_guess(runs, self.num_guesses)
            assert len(all_guesses[guesser]) == len(runs)

        # Compute per-question max length (for percentage)
        qid_to_maxlen: Dict[int, int] = defaultdict(int)
        for meta, run in zip(metadata, runs):
            qid = meta.get("qanta_id", meta.get("id"))
            qid_to_maxlen[qid] = max(qid_to_maxlen[qid], len(run))

        steps: List[Step] = []
        for guesser_id in all_guesses:
            for meta, answer, guesses, run in zip(metadata, answers, all_guesses[guesser_id], runs):
                qid = meta.get("qanta_id", meta.get("id"))
                q_max = qid_to_maxlen[qid]
                run_len = len(run)

                if answer is None:
                    answer = ""

                for guess in guesses:
                    label = 1 if rough_compare(guess["guess"], answer) else 0
                    # Text format matches original LoRABertBuzzer
                    text = "%0.2f [SEP] %s [SEP] %s" % (guess["confidence"], guess["guess"], run)
                    steps.append(Step(text=text, label=label, guess=guess["guess"], answer=answer, qid=qid, run_length=run_len, q_max_length=q_max))

        # Sort by (qid, run_length) to create time-order per question
        steps.sort(key=lambda s: (s.qid, s.run_length))
        return steps

    def _group_steps_by_question(self, steps: List[Step]) -> List[List[Step]]:
        """
        Group steps by question id (Step: qid)

        Args:
            steps: a list of all steps for all questions
        """
        grouped: Dict[int, List[Step]] = defaultdict(list)
        for s in steps:
            grouped[s.qid].append(s)
        # Ensure each question's steps are in time order
        episodes = [grouped[qid] for qid in sorted(grouped.keys())]
        return episodes

    def predict(self, questions):
        if self._pipeline is None:
            from transformers import pipeline

            # TODO: set this up to use GPU based on device
            self._classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

            # TODO: Use explainability to generate some features
            self._classifier.coef_ = [[]]

        episodes = self._rl_episodes(questions)

        predictions = []
        labels = []
        inputs = {}

        assert len(episodes) == len(questions), "Expected one episode per one question, but got %i episodes and %i questions" % (len(episodes), len(questions))

        for traj in episodes:
            for step in traj:
                a, _,_ = self._rl_sample(self._rl_encode(step.text, device=self._rl_device()))
                predictions.append(a.item())
                inputs[len(inputs)] = {
                    "text": step.text,
                    "guess": step.guess,
                    "answer": step.answer,
                    "id": step.qid
                }
                labels.append(step.label)

        for question_index in range(len(questions)):
            questions[question_index]["run"] = inputs[question_index]["text"]
            questions[question_index]["guess"] = inputs[question_index]["guess"]
            questions[question_index]["answer"] = inputs[question_index]["answer"]
            questions[question_index]["id"] = inputs[question_index]["id"]
            
        return predictions, [], [], [pred==label for pred, label in zip(predictions, labels)], questions

    # ---------------------------
    # RL training: 
    # ---------------------------

    def _rl_cfg(self) -> Dict[str, Any]:
        return {
            "lr": self.cfg.rl_lorabert_buzzer_rl_lr,
            "epochs": self.cfg.rl_lorabert_buzzer_rl_epochs,
            "gamma": self.cfg.rl_lorabert_buzzer_gamma,
            "correct_buzz_reward": self.cfg.rl_lorabert_buzzer_correct_buzz_reward,
            "wait_penalty": self.cfg.rl_lorabert_buzzer_wait_penalty,
            "wrong_buzz_penalty": self.cfg.rl_lorabert_buzzer_wrong_buzz_penalty,
            "entropy_coef": self.cfg.rl_lorabert_buzzer_entropy_coef,
            "baseline_beta": self.cfg.rl_lorabert_buzzer_baseline_beta,
            "seed": self.cfg.rl_lorabert_buzzer_seed,
            "max_steps": self.cfg.rl_lorabert_buzzer_max_steps,
        }

    def _rl_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)

    def _rl_device(self) -> torch.device:
        use_cuda = torch.cuda.is_available() and not getattr(self, "no_cuda", False)
        return torch.device("cuda" if use_cuda else "cpu")

    def _rl_trainable_params(self) -> List[torch.nn.Parameter]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found. Did LoRA get attached correctly?")
        return params

    def _rl_optimizer(self, params: List[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
        # EXTRA CREDIT: other optimizer?
        return torch.optim.AdamW(params, lr=lr)

    def _rl_episodes(self, finetune_questions) -> List[List[Step]]:
        """
        Convert questions to RL episodes
        """
        steps = self._build_steps_from_questions(finetune_questions)
        return self._group_steps_by_question(steps)

    def _rl_encode(self, text: str, device: torch.device = 'cuda:0') -> Dict[str, torch.Tensor]:
        return self.tokenizer(text, truncation=True, return_tensors="pt").to(device)

    def _rl_sample(self, enc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from a current policy returning action, log probability and entropy
        """
        logits = self.model(**enc).logits[0]
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy()

    def _rl_reward(self, step: Step, action: int) -> float:
        """
        Return penalty/reward for action
        """
        # TODO: if action = 0 ; WAIT, return wait_penalty
        # if action = 1: 
        # if correct; return ew * correct_buzz_reward 
        # otherwise return wrong_buzz_penalty

        return 0.0

    def _rl_rollout(self, traj: List[Step], device: torch.device):
        """
        rollout steps to sample action in each run (of question)
        """
        logps, ents, rewards, actions = [], [], [], []
        for step in traj[: self.cfg.rl_lorabert_buzzer_max_steps]:
            a, logp, ent = self._rl_sample(self._rl_encode(step.text, device))
            r = self._rl_reward(step, int(a.item()))
            logps.append(logp)
            ents.append(ent)
            rewards.append(r)
            actions.append(a)
            # TODO: break when we buzzes
        return logps, ents, rewards, actions

    def _rl_return(self, rewards: List[float], gamma: float) -> float:
        """
        Calculate return for one episode
        """
        R = 0.0
        # EXTRA CREDIT: maybe try with gamma?
        for r in reversed(rewards):
            R = float(r) + gamma * R
        return float(R)

    def _rl_update_baseline(self, baseline: float, R: float, beta: float) -> float:
        """
        To reduce variance, we need baseline, and here is a baseline update towards a current return
        """
        return beta * baseline + (1.0 - beta) * R

    def _rl_loss(self, logps: List[torch.Tensor], ents: List[torch.Tensor], adv: float, entropy_coef: float) -> torch.Tensor:
        """
        Calculate REINFORCE loss with entropy regularization
        """
        # TODO: reinforce_loss = A * log p 
        reinforce_loss = 0.0
        entropy_reg =  entropy_coef * torch.stack(ents).sum()
        return -(reinforce_loss + entropy_reg)

    def _rl_step(self, opt: torch.optim.Optimizer, params: List[torch.nn.Parameter], loss: torch.Tensor) -> None:
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    def _rl_eval(self, episodes, device) -> float:
        ew = 0.0
        outcomes = {"best": 0, "waiting": 0, "aggressive": 0, "timid": 0}
        neg_on_question = defaultdict(bool)
        for traj in episodes:
            step = traj[-1]
            a, logp, ent = self._rl_sample(self._rl_encode(step.text, device))
            if step.label ==1:
                if a == 1:
                    if not neg_on_question[step.qid]:
                        pct = float(step.run_length) / float(max(1, step.q_max_length))
                        ew += expected_win_probability(pct)
                    outcomes["best"] += 1
                if a==0:
                    outcomes["timid"] += 1
            elif step.label == 0 and a == 1:
                neg_on_question[step.qid] = True
                outcomes["aggressive"] += 1
            elif a == 0:
                outcomes["waiting"] += 1

        return outcomes, ew / len(episodes), sum(outcomes.values())


    def train(self, finetune_questions, dev_questions):
        """
        A training function for on-policy REINFORCE with baseline and entropy regularization.
        """
        cfg = self._rl_cfg()
        self._rl_seed(cfg["seed"])
        device = self._rl_device()
        self.model.to(device)
        self.model.train()

        params = self._rl_trainable_params()
        opt = self._rl_optimizer(params, cfg["lr"])
        episodes = self._rl_episodes(finetune_questions)
        dev_episodes = self._rl_episodes(dev_questions)

        baseline = 0.0
        for ep in range(1, cfg["epochs"] + 1):
            total_R, total_steps = 0.0, 0
            random.shuffle(episodes)

            for traj in episodes:
                # rollout one episode for this question (runs = steps)
                logps, ents, rewards, _ = self._rl_rollout(traj, device)
                # get return for this episode
                R = self._rl_return(rewards, cfg["gamma"])
                baseline = self._rl_update_baseline(baseline, R, cfg["baseline_beta"])
                # TODO: calculate advantage = R - b
                A = 0.0
                # TODO: get REINFORCE loss with entropy reg
                # back prop
                self._rl_step(opt, params, loss)
                total_R += R
                total_steps += len(rewards)

            avg_R = total_R / max(1, len(episodes))
            avg_steps = total_steps / max(1, len(episodes))
            logging.info(f"[RL] Epoch {ep}/{cfg['epochs']}: avg_return={avg_R:.4f}, avg_steps={avg_steps:.2f}")
            
            outcomes, ew, total = self._rl_eval(dev_episodes, device)
            logging.info(f"[RL] Epoch {ep}: dev expected_wins={ew:.4f}")
            logging.info("Questions Right: %i (out of %i) Accuracy: %0.2f  Buzz ratio: %0.2f Expected Wins: %f" %
              (outcomes["best"], # Right
               total,            # Total
               (outcomes["best"] + outcomes["waiting"]) / total if total > 0 else 0, # Accuracy
               (outcomes["best"] - outcomes["aggressive"] * 0.5) / total if total > 0 else 0, # Ratio
               ew))
    def save(self):
        # Save only trainable params (LoRA)
        model_state_dict = {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}
        model_config = {
            "model_state_dict": model_state_dict,
            "model_name": getattr(self, "lorabert_rl_buzzer_base_model", "distilbert-base-uncased"),
            "rank": getattr(self, "lorabert_rl_buzzer_rank", 16),
            "alpha": getattr(self, "lorabert_rl_buzzer_alpha", 1.0),
        }
        torch.save(model_config, "./models/lorabert_rl.model")

    def load(self):
        saved = torch.load("./models/lorabert_rl.model")
        self.model, self.tokenizer = initialize_base_model(model_name=saved["model_name"])
        add_lora(self.model.distilbert.transformer, saved["rank"], saved["alpha"])
        self.model.load_state_dict(saved["model_state_dict"], strict=False)

if __name__ == "__main__":
    import argparse

    from parameters import (
        add_buzzer_params,
        add_question_params,
        add_general_params,
        add_guesser_params,
        load_buzzer,
        load_questions,
        setup_logging,
    )

    parser = argparse.ArgumentParser()
    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)
    question_params = add_question_params(parser)
    add_general_params(parser)
    flags = parser.parse_args()

    setup_logging(flags)

    # IMPORTANT:
    # The buzzer internally calls its attached guessers via flags.buzzer_guessers
    # (see parameters.load_buzzer). Make sure you pass --buzzer_guessers <name>.
    buzzer = load_buzzer(flags, buzzer_params, load=False)

    finetune_questions = load_questions(flags, secondary=False)
    dev_questions = load_questions(flags, secondary=True)

    if finetune_questions and getattr(flags, "limit", -1) > 0:
        finetune_questions = finetune_questions[: flags.limit]
    if dev_questions and getattr(flags, "limit", -1) > 0:
        dev_questions = dev_questions[: flags.limit]

    buzzer.train(finetune_questions, dev_questions)

    # Save learned LoRA parameters
    try:
        buzzer.save()
    except Exception as e:
        logging.warning(f"Save failed: {e}")
