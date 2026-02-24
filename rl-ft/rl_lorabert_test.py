import unittest
import random
import torch

from rl_lorabert_buzzer import LoRABertRLBuzzer, Step, initialize_base_model


class LoraBertRLTest(unittest.TestCase):
    def setUp(self):
        """
        setting up a minimal LoRABertRLBuzzer instance with a tiny random model for testing RL components.
        """
        torch.manual_seed(0)
        random.seed(0)
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
        torch.manual_seed(0) 
        random.seed(0) 
        
        parser = argparse.ArgumentParser() 
        guesser_params = add_guesser_params(parser) 
        buzzer_params = add_buzzer_params(parser) 
        question_params = add_question_params(parser) 
        add_general_params(parser) 
        flags = parser.parse_args() 
        flags.logging_level = "CRITICAL"  # suppress logging during tests
        setup_logging(flags) 
        
        self.buzzer = LoRABertRLBuzzer(filename="", run_length=5, num_guesses=1, cfg=flags) 
        self.buzzer.model, self.buzzer.tokenizer = initialize_base_model(
            model_name="hf-internal-testing/tiny-random-BertForSequenceClassification"
        )

        # Force CPU for tests
        self.buzzer.no_cuda = True
        self.device = torch.device("cpu")
        self.buzzer.model.to(self.device)

        # RL config
        self.cfg = {
            "lr": 1e-2,
            "epochs": 1,
            "gamma": 1.0,
            "correct_buzz_reward": 5.0,
            "wait_penalty": -0.02,
            "wrong_buzz_penalty": -1.0,
            "entropy_coef": 0.01,
            "baseline_beta": 0.9,
            "seed": 0,
            "max_steps": 10,
        }

    # --------------------------
    # Utilities
    # --------------------------

    def util_params_snapshot(self):
        params = [p for p in self.buzzer.model.parameters() if p.requires_grad]
        return [p.detach().clone() for p in params]

    def util_any_param_changed(self, before, after):
        for b, a in zip(before, after):
            if not torch.allclose(b, a):
                return True
        return False

    # --------------------------
    # Tests: return, loss, grads
    # --------------------------

    def test_rl_return_discounted(self):
        """
        whether the return is correctly computed as discounted sum of rewards
        """
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.5
        # Return = 1 + 0.5*(2 + 0.5*3) = 2.75
        R = self.buzzer._rl_return(rewards, gamma)
        self.assertAlmostEqual(R, 2.75, places=6)

    def test_rl_loss_scalar_and_backward(self):
        """
        whether the loss is a scalar tensor and backward() computes gradients without error
        """
        traj = [
            Step(text="0.50 [SEP] guess [SEP] question text aaaa", label=1, qid=0, guess='', answer='', run_length=1, q_max_length=4),
            Step(text="0.40 [SEP] guess [SEP] question text bbbbbbbb", label=0, qid=0, guess='', answer='',run_length=2, q_max_length=4),
        ]

        logps, ents, rewards, actions = self.buzzer._rl_rollout(traj, self.device)
        # Basic sanity checks on outputs
        self.assertEqual(len(logps), len(ents))
        self.assertEqual(len(ents), len(rewards))
        self.assertEqual(len(rewards), len(actions))
        self.assertGreaterEqual(len(rewards), 1)

        R = self.buzzer._rl_return(rewards, self.cfg["gamma"])
        adv = R - 0.0
        loss = self.buzzer._rl_loss(logps, ents, adv, self.cfg["entropy_coef"])

        # Loss should be a scalar tensor
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0, "Loss should be a scalar tensor")

        params = [p for p in self.buzzer.model.parameters() if p.requires_grad]
        for p in params:
            self.assertIsNone(p.grad)

        loss.backward()

        any_grad = False
        for p in params:
            if p.grad is not None and torch.isfinite(p.grad).all() and torch.any(p.grad != 0):
                any_grad = True
                break
        # Check that at least one trainable parameter has a non-zero, finite gradient
        self.assertTrue(any_grad, "Expected some non-zero gradients on trainable params")

    # --------------------------
    # Tests: rollout termination
    # --------------------------

    def test_rollout_stops_on_buzz_when_forced(self):
        """
        Test that the rollout correctly terminates when a BUZZ action is taken
        """
        # Force BUZZ on first step by patching _rl_sample
        def forced_buzz(enc):
            a = torch.tensor(1)  # BUZZ
            logp = torch.tensor(0.0)
            ent = torch.tensor(0.0)
            return a, logp, ent

        original = self.buzzer._rl_sample
        self.buzzer._rl_sample = forced_buzz
        try:
            traj = [
                Step(text="0.5 [SEP] g [SEP] step1", label=1, qid=0, guess='', answer='', run_length=1, q_max_length=3),
                Step(text="0.5 [SEP] g [SEP] step2", label=1, qid=0, guess='', answer='', run_length=2, q_max_length=3),
                Step(text="0.5 [SEP] g [SEP] step3", label=1, qid=0, guess='', answer='', run_length=3, q_max_length=3),
            ]
            logps, ents, rewards, actions = self.buzzer._rl_rollout(traj, self.device)

            self.assertEqual(len(actions), 1)
            self.assertEqual(int(actions[0].item()), 1)
        finally:
            self.buzzer._rl_sample = original

    # --------------------------
    # Tests: reward correctness
    # --------------------------

    def test_rl_reward_wait_penalty(self):
        step = Step(text="t", label=1, qid=0, guess='', answer='',run_length=1, q_max_length=10)
        r = self.buzzer._rl_reward(step, action=0)
        self.assertAlmostEqual(r, self.cfg["wait_penalty"], places=6)

    def test_rl_reward_wrong_buzz_penalty(self):
        step = Step(text="t", label=0, qid=0, guess='', answer='',run_length=5, q_max_length=10)
        r = self.buzzer._rl_reward(step, action=1)
        self.assertAlmostEqual(r, self.cfg["wrong_buzz_penalty"], places=6)

    def test_rl_reward_correct_buzz_positive(self):
        step = Step(text="t", label=1, qid=0, guess='', answer='',run_length=5, q_max_length=10)
        r = self.buzzer._rl_reward(step, action=1)
        self.assertTrue(r > 0.0, "Correct buzz reward should be positive")


if __name__ == "__main__":
    unittest.main()
