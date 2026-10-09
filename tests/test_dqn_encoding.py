"""DQN observation encoding: layout, history-freedom and Markov-relevant distinctions."""

import importlib.util
import unittest

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    import numpy as np

    from core.environment import MODE_CYCLE, SCATTER_STEPS, Environment
    from rl.dqn_model import (
        ACTION_TO_IDX, CH_CYCLE_PHASE, CH_GHOST_HEADINGS, CH_GHOSTS, CH_PACMAN, CH_PACMAN_HEADING,
        CH_SCATTER, NUM_CHANNELS, NUM_SCALARS, SCALAR_HORIZON, SCALAR_STALL, STALL_STEPS, encode_state,
    )


def _encode(env, **overrides):
    args = dict(
        pacman_pos=tuple(env.pacman_pos), ghost_positions=[tuple(g) for g in env.ghost_positions],
        pellets=env.pellets, last_move=env.last_move, ghost_dirs=env.ghost_dirs, mode_step=env.mode_step,
        steps_without_pellet=env.steps_without_pellet, steps_remaining=300, horizon=300,
    )
    args.update(overrides)
    return encode_state(**args)


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional")
class TestDQNEncoding(unittest.TestCase):
    def test_shapes(self):
        grid, scalars = _encode(Environment(seed=1))
        self.assertEqual(grid.shape[0], NUM_CHANNELS)
        self.assertEqual(scalars.shape, (NUM_SCALARS,))

    def test_headings_are_set_only_at_their_actor(self):
        env = Environment(seed=1)
        grid, _ = _encode(env)
        px, py = env.pacman_pos
        pac_heading = grid[CH_PACMAN_HEADING + ACTION_TO_IDX[env.last_move]]
        self.assertEqual(pac_heading.sum(), 1.0)
        self.assertEqual(pac_heading[py, px], 1.0)
        # Every actor's position plane equals the sum of its heading planes
        np.testing.assert_array_equal(grid[CH_PACMAN], grid[CH_PACMAN_HEADING:CH_PACMAN_HEADING + 4].sum(0))
        for i in range(3):
            heading = grid[CH_GHOST_HEADINGS + 4 * i:CH_GHOST_HEADINGS + 4 * i + 4].sum(0)
            np.testing.assert_array_equal(grid[CH_GHOSTS + i], heading)

    def test_encoding_is_a_function_of_the_current_state(self):
        env = Environment(seed=2)
        for _ in range(5):
            env.step(env.get_legal_moves(*env.pacman_pos)[0])
        a_grid, a_scalars = _encode(env)
        b_grid, b_scalars = _encode(env)
        np.testing.assert_array_equal(a_grid, b_grid)
        np.testing.assert_array_equal(a_scalars, b_scalars)

    def test_distinguishes_headings_identity_and_phase(self):
        env = Environment(seed=1)
        base, _ = _encode(env)
        other_heading, _ = _encode(env, ghost_dirs=["left"] + list(env.ghost_dirs[1:]))
        self.assertFalse(np.array_equal(base, other_heading))
        swapped, _ = _encode(env, ghost_positions=[tuple(env.ghost_positions[1]), tuple(env.ghost_positions[0]),
                                                   tuple(env.ghost_positions[2])])
        self.assertFalse(np.array_equal(base, swapped))
        # The scatter plane describes the ghosts' *next* move (clock mode_step + 1)
        for mode_step in (0, SCATTER_STEPS - 1, SCATTER_STEPS, MODE_CYCLE - 1, MODE_CYCLE):
            grid, _ = _encode(env, mode_step=mode_step)
            env.mode_step = mode_step + 1
            self.assertEqual(bool(grid[CH_SCATTER, 0, 0]), env.in_scatter, f"mode_step={mode_step}")
            self.assertTrue(0.0 <= grid[CH_CYCLE_PHASE, 0, 0] <= 1.0)

    def test_scalars(self):
        env = Environment(seed=1)
        _, s = _encode(env, steps_without_pellet=STALL_STEPS * 2, steps_remaining=150, horizon=300)
        self.assertEqual(s[SCALAR_STALL], 1.0)  # saturates at the stall threshold
        self.assertAlmostEqual(float(s[SCALAR_HORIZON]), 0.5)


if __name__ == "__main__":
    unittest.main()
