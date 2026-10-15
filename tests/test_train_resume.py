"""Resumable DQN training: a stopped-and-resumed run must match an uninterrupted one."""

import importlib.util
import os
import tempfile
import unittest
from unittest import mock

TORCH_AVAILABLE = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    import rl.train_dqn as train_dqn_module
    from rl.train_dqn import TRAINING_STATE_NAME, train_dqn

# Tiny but complete run: warm-up, PER updates, target syncs, validation and best-model saves.
TINY = dict(
    episodes=4, max_steps=15, warmup_steps=40, batch_size=8, val_every=2, val_episodes=1,
    buffer_capacity=200, target_update_steps=10, make_plots=False, final_val_episodes=2,
)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional")
class TestTrainResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _run_dir(self, name):
        d = os.path.join(self.tmp, name)
        os.makedirs(d)
        return d, os.path.join(d, "dqn_pacman.pt"), os.path.join(d, TRAINING_STATE_NAME)

    def test_stop_and_resume_matches_uninterrupted_run(self):
        _, model_a, _ = self._run_dir("a")
        full = train_dqn(save_path=model_a, **TINY)
        self.assertTrue(full["completed"])

        _, model_b, state_b = self._run_dir("b")
        first = train_dqn(save_path=model_b, stop_after=2, **TINY)
        self.assertFalse(first["completed"])
        self.assertEqual(first["episodes_done"], 2)
        self.assertTrue(os.path.exists(state_b))

        resumed = train_dqn(resume=True, checkpoint_path=state_b, make_plots=False, final_val_episodes=2)
        self.assertTrue(resumed["completed"])
        self.assertFalse(os.path.exists(state_b), "training state must be removed after completion")

        self.assertEqual(resumed["metrics"], full["metrics"])
        self.assertEqual(resumed["best_val_episode"], full["best_val_episode"])
        self.assertEqual(resumed["final_eval"], full["final_eval"])
        best_a = torch.load(model_a, weights_only=True)
        best_b = torch.load(model_b, weights_only=True)
        for key in best_a:
            self.assertTrue(torch.allclose(best_a[key], best_b[key], atol=1e-5), key)

    def test_ctrl_c_saves_state_and_resume_completes_without_duplicates(self):
        _, model, state = self._run_dir("c")
        real_reward = train_dqn_module.compute_reward
        calls = {"n": 0}
        interrupt_at = TINY["warmup_steps"] + TINY["max_steps"] + 3  # inside episode 2 at the latest

        def interrupting_reward(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == interrupt_at:
                raise KeyboardInterrupt
            return real_reward(*args, **kwargs)

        with mock.patch.object(train_dqn_module, "compute_reward", interrupting_reward):
            interrupted = train_dqn(save_path=model, **TINY)
        self.assertFalse(interrupted["completed"])
        self.assertTrue(os.path.exists(state))
        saved = torch.load(state, weights_only=True)  # must stay weights_only-loadable
        self.assertEqual(saved["progress"]["episodes_done"], interrupted["episodes_done"])

        resumed = train_dqn(resume=True, checkpoint_path=state, make_plots=False, final_val_episodes=2)
        self.assertTrue(resumed["completed"])
        self.assertEqual([m["episode"] for m in resumed["metrics"]], list(range(1, TINY["episodes"] + 1)))

    def test_final_evaluation_uses_unseen_val_seeds(self):
        from core.seeds import val_seeds

        run_dir, model, _ = self._run_dir("f")
        result = train_dqn(save_path=model, **TINY)
        final = result["final_eval"]
        selection = set(val_seeds(TINY["val_episodes"]))
        self.assertNotIn(final["seeds"][0], selection)
        self.assertEqual(final["seeds"][1] - final["seeds"][0] + 1, TINY["final_val_episodes"])
        self.assertEqual(final["best_checkpoint"]["episode"], result["best_val_episode"])
        for key in ("best_checkpoint", "final_weights"):
            self.assertIn("mean_score", final[key])
        self.assertTrue(os.path.exists(os.path.join(run_dir, "dqn_final_eval.json")))

    def test_epsilon_schedule_follows_run_length(self):
        from rl.train_dqn import epsilon_schedule

        for episodes in (200, 1200):
            eps = [epsilon_schedule(ep, episodes, 1.0, 0.05, 0.6) for ep in range(episodes)]
            self.assertEqual(eps[0], 1.0)
            self.assertTrue(all(a >= b for a, b in zip(eps, eps[1:])), "must not increase")
            self.assertAlmostEqual(eps[int(0.6 * episodes)], 0.05)
            self.assertEqual(eps[-1], 0.05)

    def test_resume_without_state_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            train_dqn(resume=True, checkpoint_path=os.path.join(self.tmp, "missing.pt"))

    def test_agent_loads_current_and_rejects_legacy_checkpoints(self):
        from agents.dqn_agent import DQNAgent
        from rl.dqn_model import PacmanDQN

        current = os.path.join(self.tmp, "current.pt")
        torch.save(PacmanDQN().state_dict(), current)
        agent = DQNAgent(model_path=current, require_weights=True)
        res = agent.decide((9, 15), [(9, 7), (7, 7), (11, 7)], {(1, 1)}, ["left", "right"])
        self.assertIn(res.choice, ["left", "right"])

        legacy = os.path.join(self.tmp, "legacy.pt")  # old 30-channel pooled network layout
        torch.save({"conv.0.weight": torch.zeros(32, 30, 3, 3), "conv.0.bias": torch.zeros(32)}, legacy)
        with self.assertRaisesRegex(RuntimeError, "retrain"):
            DQNAgent(model_path=legacy, require_weights=True)
        fallback = DQNAgent(model_path=legacy)
        self.assertFalse(fallback.model_loaded)
        self.assertIn("[UNTRAINED]", fallback.name)

    def test_replay_buffer_state_round_trip(self):
        import numpy as np
        from rl.train_dqn import ReplayBuffer

        def obs(i):
            return np.full((1, 2, 2), i, np.float32), np.full(2, i / 10, np.float32)

        buf = ReplayBuffer(capacity=3)
        for i in range(5):  # wraps around the ring
            buf.push(obs(i), i % 4, float(i), obs(i + 1), 0.0)
        buf.update_priorities([0], [2.5])
        restored = ReplayBuffer(capacity=3)
        restored.load_state_dict(buf.state_dict())
        self.assertEqual((restored.size, restored.position, restored.max_priority), (buf.size, buf.position, buf.max_priority))
        for name in ("grids", "scalars", "next_grids", "next_scalars"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(buf, name))
        np.testing.assert_array_equal(restored.priorities, buf.priorities)
        np.testing.assert_array_equal(restored.actions, buf.actions)

    def test_vectorized_per_sampling_matches_random_choices(self):
        import random

        import numpy as np
        from rl.train_dqn import ReplayBuffer

        buf = ReplayBuffer(capacity=50, rng=random.Random(7))
        for i in range(37):
            buf.push((np.zeros((1, 2, 2), np.float32), np.zeros(2, np.float32)), i % 4, 0.0,
                     (np.zeros((1, 2, 2), np.float32), np.zeros(2, np.float32)), 0.0)
        buf.priorities[:37] = np.random.default_rng(0).uniform(0.01, 5.0, 37).astype(np.float32)
        probabilities = buf.priorities[:37].astype(np.float64) ** buf.alpha
        probabilities /= probabilities.sum()

        reference_rng = random.Random(7)
        for _ in range(20):
            batch = buf.sample(16, beta=0.5)
            expected = reference_rng.choices(range(37), weights=probabilities.tolist(), k=16)
            self.assertEqual(list(batch["indices"]), expected)

    def test_train_every_stop_and_resume_matches_uninterrupted_run(self):
        cfg = dict(TINY, train_every=2)
        _, model_a, _ = self._run_dir("te_a")
        full = train_dqn(save_path=model_a, **cfg)

        _, model_b, state_b = self._run_dir("te_b")
        train_dqn(save_path=model_b, stop_after=2, **cfg)
        self.assertEqual(torch.load(state_b, weights_only=True)["config"]["train_every"], 2)
        resumed = train_dqn(resume=True, checkpoint_path=state_b, make_plots=False, final_val_episodes=2)
        self.assertEqual(resumed["metrics"], full["metrics"])
        self.assertEqual(resumed["final_eval"], full["final_eval"])

    def test_train_every_must_be_positive(self):
        with self.assertRaises(ValueError):
            train_dqn(save_path=os.path.join(self.tmp, "x.pt"), train_every=0, **TINY)

    def test_beta_schedule_linear_annealing(self):
        from rl.train_dqn import beta_schedule

        self.assertAlmostEqual(beta_schedule(0, 1000, 0.4), 0.4)
        self.assertAlmostEqual(beta_schedule(500, 1000, 0.4), 0.7)
        self.assertAlmostEqual(beta_schedule(1000, 1000, 0.4), 1.0)
        self.assertAlmostEqual(beta_schedule(1500, 1000, 0.4), 1.0)  # clamped at 1.0

        betas = [beta_schedule(s, 1000, 0.4) for s in range(0, 1001, 100)]
        self.assertTrue(all(a <= b for a, b in zip(betas, betas[1:])), "beta must not decrease")

    def test_cosine_lr_annealing(self):
        import math
        from rl.train_dqn import cosine_lr

        lr = 5e-4
        final_ratio = 0.1
        self.assertAlmostEqual(cosine_lr(0, 1000, lr, final_ratio), lr)
        self.assertAlmostEqual(cosine_lr(1000, 1000, lr, final_ratio), lr * final_ratio)
        self.assertAlmostEqual(cosine_lr(1500, 1000, lr, final_ratio), lr * final_ratio)

        # Midpoint cosine value
        expected_mid = lr * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * 0.5)))
        self.assertAlmostEqual(cosine_lr(500, 1000, lr, final_ratio), expected_mid)

        lrs = [cosine_lr(s, 1000, lr, final_ratio) for s in range(0, 1001, 100)]
        self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])), "cosine lr must monotonically decrease")

    def test_soft_update_polyak_averaging(self):
        from rl.train_dqn import soft_update

        linear_target = torch.nn.Linear(4, 2, bias=False)
        linear_policy = torch.nn.Linear(4, 2, bias=False)

        with torch.no_grad():
            linear_target.weight.fill_(1.0)
            linear_policy.weight.fill_(3.0)

        # tau = 0.0 -> target unchanged
        soft_update(linear_target, linear_policy, tau=0.0)
        self.assertTrue(torch.allclose(linear_target.weight, torch.full((2, 4), 1.0)))

        # tau = 0.5 -> target becomes 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        soft_update(linear_target, linear_policy, tau=0.5)
        self.assertTrue(torch.allclose(linear_target.weight, torch.full((2, 4), 2.0)))

        # tau = 1.0 -> target matches policy exactly
        soft_update(linear_target, linear_policy, tau=1.0)
        self.assertTrue(torch.allclose(linear_target.weight, torch.full((2, 4), 3.0)))

    def test_step_based_budget_stop_and_resume_matches(self):
        # Budget defined by total_steps rather than episodes
        cfg = dict(
            total_steps=50, max_steps=15, warmup_steps=20, batch_size=8,
            val_every_steps=25, val_episodes=1, buffer_capacity=200,
            target_tau=0.05, make_plots=False, final_val_episodes=2,
        )
        _, model_a, _ = self._run_dir("step_a")
        full = train_dqn(save_path=model_a, **cfg)
        self.assertTrue(full["completed"])
        self.assertGreaterEqual(full["steps_done"], 50)

        _, model_b, state_b = self._run_dir("step_b")
        first = train_dqn(save_path=model_b, stop_after=2, **cfg)
        self.assertFalse(first["completed"])
        self.assertTrue(os.path.exists(state_b))

        resumed = train_dqn(resume=True, checkpoint_path=state_b, make_plots=False, final_val_episodes=2)
        self.assertTrue(resumed["completed"])
        self.assertEqual(resumed["steps_done"], full["steps_done"])
        self.assertEqual(resumed["metrics"], full["metrics"])
        self.assertEqual(resumed["final_eval"], full["final_eval"])

    def test_invalid_parameters_raise_value_errors(self):
        with self.assertRaises(ValueError):
            train_dqn(save_path=os.path.join(self.tmp, "x.pt"), total_steps=0, **TINY)
        with self.assertRaises(ValueError):
            train_dqn(save_path=os.path.join(self.tmp, "x.pt"), target_tau=-0.1, **TINY)
        with self.assertRaises(ValueError):
            train_dqn(save_path=os.path.join(self.tmp, "x.pt"), target_tau=1.5, **TINY)
        with self.assertRaises(ValueError):
            train_dqn(save_path=os.path.join(self.tmp, "x.pt"), val_every_steps=0, **TINY)
        with self.assertRaises(ValueError):
            train_dqn(save_path=os.path.join(self.tmp, "x.pt"), n_step=0, **TINY)

    def test_n_step_buffer_accumulation_and_flush(self):
        from rl.train_dqn import NStepBuffer
        import numpy as np

        nb = NStepBuffer(n=3, gamma=0.9)
        dummy_mask = np.ones(4, dtype=bool)

        # Step 0: return None (len=1 < 3)
        tr0 = nb.add("s0", 0, 1.0, "s1", False, dummy_mask)
        self.assertIsNone(tr0)

        # Step 1: return None (len=2 < 3)
        tr1 = nb.add("s1", 1, 2.0, "s2", False, dummy_mask)
        self.assertIsNone(tr1)

        # Step 2: return 3-step transition for s0 (1.0 + 0.9*2.0 + 0.9^2*3.0 = 5.23)
        tr2 = nb.add("s2", 2, 3.0, "s3", False, dummy_mask)
        self.assertIsNotNone(tr2)
        s, a, r, ns, d, m = tr2
        self.assertEqual(s, "s0")
        self.assertEqual(a, 0)
        self.assertAlmostEqual(r, 1.0 + 0.9 * 2.0 + 0.81 * 3.0)  # 5.23
        self.assertEqual(ns, "s3")
        self.assertEqual(d, 0.0)

        # Step 3 (terminal death): return 3-step transition for s1 (2.0 + 0.9*3.0 + 0.81*4.0 = 7.94)
        tr3 = nb.add("s3", 3, 4.0, "s4", True, dummy_mask)
        self.assertIsNotNone(tr3)
        s, a, r, ns, d, m = tr3
        self.assertEqual(s, "s1")
        self.assertAlmostEqual(r, 2.0 + 0.9 * 3.0 + 0.81 * 4.0)  # 7.94
        self.assertEqual(ns, "s4")
        self.assertEqual(d, 1.0)

        # Flush remaining 2 items (step 2 and step 3)
        flushed = nb.flush()
        self.assertEqual(len(flushed), 2)
        # Item 1 (from s2): 3.0 + 0.9 * 4.0 = 6.6
        self.assertEqual(flushed[0][0], "s2")
        self.assertAlmostEqual(flushed[0][2], 3.0 + 0.9 * 4.0)
        self.assertEqual(flushed[0][3], "s4")
        self.assertEqual(flushed[0][4], 1.0)
        # Item 2 (from s3): 4.0
        self.assertEqual(flushed[1][0], "s3")
        self.assertAlmostEqual(flushed[1][2], 4.0)
        self.assertEqual(flushed[1][3], "s4")
        self.assertEqual(flushed[1][4], 1.0)

        self.assertEqual(len(nb.buffer), 0)

    def test_per_max_priority_drift_fixed(self):
        import numpy as np
        from rl.train_dqn import ReplayBuffer

        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.push((np.zeros((1, 2, 2)), np.zeros(2)), 0, 0.0, (np.zeros((1, 2, 2)), np.zeros(2)), 0.0)

        # Simulate large error spike (e.g. 20.0 death error)
        buf.update_priorities([0], [20.0])
        self.assertAlmostEqual(buf.max_priority, 20.0 + buf.priority_epsilon, places=4)

        # In previous versions, max_priority would stay at 20.0 forever even after the transition is re-evaluated.
        # Now, updating that transition with low error drops max_priority back down:
        buf.update_priorities([0], [0.2])
        self.assertLess(buf.max_priority, 1.5)

    def test_n_step_stop_and_resume_matches_uninterrupted_run(self):
        cfg = dict(TINY, n_step=3, gamma=0.99, target_tau=0.005)
        _, model_a, _ = self._run_dir("nstep_a")
        full = train_dqn(save_path=model_a, **cfg)
        self.assertTrue(full["completed"])

        _, model_b, state_b = self._run_dir("nstep_b")
        first = train_dqn(save_path=model_b, stop_after=2, **cfg)
        self.assertFalse(first["completed"])
        self.assertTrue(os.path.exists(state_b))
        self.assertEqual(torch.load(state_b, weights_only=True)["config"]["n_step"], 3)

        resumed = train_dqn(resume=True, checkpoint_path=state_b, make_plots=False, final_val_episodes=2)
        self.assertTrue(resumed["completed"])
        self.assertEqual(resumed["metrics"], full["metrics"])
        self.assertEqual(resumed["final_eval"], full["final_eval"])
        best_a = torch.load(model_a, weights_only=True)
        best_b = torch.load(model_b, weights_only=True)
        for key in best_a:
            self.assertTrue(torch.allclose(best_a[key], best_b[key], atol=1e-5), key)


if __name__ == "__main__":
    unittest.main()


