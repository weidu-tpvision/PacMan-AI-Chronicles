import os
import random
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.environment import DEFAULT_MAX_STEPS, Environment
from core.maze_data import GRID_HEIGHT, GRID_WIDTH
from rl.dqn_model import (
    NUM_CHANNELS,
    NUM_SCALARS,
    CompactObservation,
    decode_grids_batch,
    encode_state,
    pack_pellets,
)
from rl.train_dqn import ReplayBuffer, SumTree, observe


class TestCompactReplay(unittest.TestCase):
    def test_compact_observation_protocol(self):
        env = Environment(seed=42)
        obs = observe(env, steps_without_pellet=5, steps_remaining=250, horizon=300)

        # Must support tuple unpacking
        grid, scalars = obs
        self.assertEqual(grid.shape, (NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH))
        self.assertEqual(scalars.shape, (NUM_SCALARS,))
        self.assertEqual(len(obs), 2)

        # Must support index access
        np.testing.assert_array_equal(obs[0], grid)
        np.testing.assert_array_equal(obs[1], scalars)
        with self.assertRaises(IndexError):
            _ = obs[2]

        # Must carry compact metadata
        self.assertTrue(hasattr(obs, "compact"))
        self.assertEqual(len(obs.compact), 9)

    def test_sumtree_properties_and_sampling(self):
        cap = 50
        tree = SumTree(cap)
        self.assertEqual(tree.tree_capacity, 64)  # next power of 2
        self.assertEqual(tree.total_sum, 0.0)
        self.assertEqual(tree.max_value, 0.0)

        # Add single priorities
        tree.update(0, 1.5, alpha=0.6)
        tree.update(1, 0.5, alpha=0.6)
        self.assertAlmostEqual(tree.max_value, 1.5)
        self.assertAlmostEqual(tree.total_sum, (1.5 ** 0.6) + (0.5 ** 0.6))

        # Rebuild tree from array
        priorities = np.array([2.0, 1.0, 0.1, 4.0], dtype=np.float32)
        tree.rebuild(priorities, alpha=0.6)
        self.assertAlmostEqual(tree.max_value, 4.0)
        expected_total = sum(float(p) ** 0.6 for p in priorities)
        self.assertAlmostEqual(tree.total_sum, expected_total, places=5)

        # Sample with draws
        draws = np.array([0.001, expected_total - 0.001])
        indices, scaled = tree.sample(draws, 4)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[1], 3)

    def test_decode_grids_batch_parity(self):
        """Batch decoding from compact representation must exactly match encode_state."""
        env = Environment(seed=123)
        states = []
        for _ in range(50):
            legal = env.get_legal_moves(*env.pacman_pos)
            action = random.choice(legal)
            env.step(action)
            obs = observe(env, steps_without_pellet=3, steps_remaining=100, horizon=300)
            states.append(obs)

        px = np.array([s.compact[0] for s in states], dtype=np.uint8)
        py = np.array([s.compact[1] for s in states], dtype=np.uint8)
        gx = np.array([s.compact[2] for s in states], dtype=np.uint8)
        gy = np.array([s.compact[3] for s in states], dtype=np.uint8)
        p_head = np.array([s.compact[4] for s in states], dtype=np.int8)
        g_head = np.array([s.compact[5] for s in states], dtype=np.int8)
        mode_step = np.array([s.compact[6] for s in states], dtype=np.uint16)
        pellet_bits = np.array([s.compact[7] for s in states], dtype=np.uint8)

        decoded = decode_grids_batch(px, py, gx, gy, p_head, g_head, mode_step, pellet_bits)
        if TORCH_AVAILABLE:
            decoded = decoded.numpy()

        for i, obs in enumerate(states):
            np.testing.assert_array_equal(
                decoded[i],
                obs.grid,
                err_msg=f"Decoded grid mismatch at state index {i}",
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for buffer test")
    def test_compact_replay_buffer_round_trip_and_sample(self):
        buf = ReplayBuffer(capacity=100, rng=random.Random(42))
        env = Environment(seed=99)
        obs = observe(env, 0, 300, 300)

        for step in range(60):
            legal = env.get_legal_moves(*env.pacman_pos)
            action = random.choice(legal)
            action_idx = 0
            env.step(action)
            next_obs = observe(env, 1, 299 - step, 300)
            buf.push(obs, action_idx, 1.0, next_obs, False)
            obs = next_obs

        self.assertEqual(buf.size, 60)
        self.assertTrue(buf.is_compact)

        # Sample batch
        batch = buf.sample(16, beta=0.4)
        self.assertEqual(batch["grids"].shape, (16, NUM_CHANNELS, GRID_HEIGHT, GRID_WIDTH))
        self.assertEqual(batch["scalars"].shape, (16, NUM_SCALARS))
        self.assertEqual(batch["actions"].shape, (16, 1))
        self.assertEqual(batch["rewards"].shape, (16, 1))

        # Checkpoint round trip
        state = buf.state_dict()
        self.assertTrue(state["is_compact"])
        self.assertIn("pellet_bits", state)

        restored = ReplayBuffer(capacity=100, rng=random.Random(42))
        restored.load_state_dict(state)
        self.assertEqual(restored.size, buf.size)
        self.assertEqual(restored.position, buf.position)
        self.assertAlmostEqual(restored.max_priority, buf.max_priority)
        self.assertTrue(restored.is_compact)
        np.testing.assert_array_equal(restored.pellet_bits[:60], buf.pellet_bits[:60])
        np.testing.assert_array_equal(restored.priorities[:60], buf.priorities[:60])


if __name__ == "__main__":
    unittest.main()
