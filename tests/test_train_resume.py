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
    buffer_capacity=200, target_update_steps=10, make_plots=False,
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

        resumed = train_dqn(resume=True, checkpoint_path=state_b, make_plots=False)
        self.assertTrue(resumed["completed"])
        self.assertFalse(os.path.exists(state_b), "training state must be removed after completion")

        self.assertEqual(resumed["metrics"], full["metrics"])
        self.assertEqual(resumed["best_val_episode"], full["best_val_episode"])
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

        resumed = train_dqn(resume=True, checkpoint_path=state, make_plots=False)
        self.assertTrue(resumed["completed"])
        self.assertEqual([m["episode"] for m in resumed["metrics"]], list(range(1, TINY["episodes"] + 1)))

    def test_resume_without_state_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            train_dqn(resume=True, checkpoint_path=os.path.join(self.tmp, "missing.pt"))

    def test_replay_buffer_state_round_trip(self):
        import numpy as np
        from rl.train_dqn import ReplayBuffer

        buf = ReplayBuffer(capacity=3)
        for i in range(5):  # wraps around the ring
            buf.push(np.full((2, 2), i, np.float32), i % 4, float(i), np.full((2, 2), i + 1, np.float32), 0.0)
        buf.update_priorities([0], [2.5])
        restored = ReplayBuffer(capacity=3)
        restored.load_state_dict(buf.state_dict())
        self.assertEqual((restored.size, restored.position, restored.max_priority), (buf.size, buf.position, buf.max_priority))
        np.testing.assert_array_equal(restored.states, buf.states)
        np.testing.assert_array_equal(restored.priorities, buf.priorities)
        np.testing.assert_array_equal(restored.actions, buf.actions)


if __name__ == "__main__":
    unittest.main()
