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
            self.assertTrue(torch.equal(best_a[key], best_b[key]), key)

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


if __name__ == "__main__":
    unittest.main()
