"""Regression tests for core.environment game dynamics (no optional dependencies)."""

import unittest

from core.environment import MODE_CYCLE, SCATTER_STEPS, Environment, pacman_collides
from core.maze_data import GRID_WIDTH


class TestScatterChaseClock(unittest.TestCase):
    def test_phase_boundaries(self):
        env = Environment(seed=1)
        expected = {
            0: False,                        # before the first step
            1: True,                         # cycle opens in Scatter
            SCATTER_STEPS: True,             # last Scatter step
            SCATTER_STEPS + 1: False,        # first Chase step
            MODE_CYCLE: False,               # last Chase step
            MODE_CYCLE + 1: True,            # next cycle back in Scatter
        }
        for mode_step, in_scatter in expected.items():
            env.mode_step = mode_step
            self.assertEqual(env.in_scatter, in_scatter, f"mode_step={mode_step}")

    def test_step_advances_clock(self):
        env = Environment(seed=1)
        env.step(env.get_legal_moves(*env.pacman_pos)[0])
        self.assertEqual(env.mode_step, 1)
        self.assertTrue(env.in_scatter)


class TestCollisionRule(unittest.TestCase):
    def test_same_tile(self):
        self.assertTrue(pacman_collides([5, 5], [6, 5], [[7, 5]], [[6, 5]]))

    def test_head_on_swap(self):
        self.assertTrue(pacman_collides([5, 5], [6, 5], [[6, 5]], [[5, 5]]))

    def test_walking_onto_ghost_that_moves_away(self):
        # Pac-Man enters the ghost's tile while the ghost steps perpendicular
        self.assertTrue(pacman_collides([5, 5], [6, 5], [[6, 5]], [[6, 4]]))

    def test_ghost_entering_vacated_tile_is_a_miss(self):
        self.assertFalse(pacman_collides([5, 5], [6, 5], [[4, 5]], [[5, 5]]))

    def test_ghost_reaching_stationary_pacman(self):
        # Illegal move (no-op): only the same-tile rule applies
        self.assertTrue(pacman_collides([5, 5], [5, 5], [[5, 4]], [[5, 5]]))
        self.assertFalse(pacman_collides([5, 5], [5, 5], [[5, 5]], [[5, 4]]))

    def test_environment_uses_rule(self):
        env = Environment(seed=3)
        ghost_start = list(env.pacman_pos)
        ghost_start[0] -= 1  # tile left of Pac-Man; Pac-Man steps into it
        env.ghost_positions = [ghost_start]
        env.ghost_dirs = ["up"]
        collided, _, _ = env.step("left")
        self.assertTrue(collided)


class TestTunnel(unittest.TestCase):
    def test_pacman_wraps_through_tunnel(self):
        env = Environment(seed=1)
        env.pacman_pos = [0, 9]
        env.ghost_positions = [[1, 1]]
        env.ghost_dirs = ["up"]
        env.step("left")
        self.assertEqual(env.pacman_pos, [GRID_WIDTH - 1, 9])

    def test_ghost_wraps_through_tunnel(self):
        env = Environment(seed=1)
        env.pacman_pos = [1, 1]
        env.ghost_positions = [[0, 9]]
        env.ghost_dirs = ["left"]  # reversing is forbidden, so "left" is the only option
        env.step("right")
        self.assertEqual(env.ghost_positions[0], [GRID_WIDTH - 1, 9])


class TestIllegalMoves(unittest.TestCase):
    def test_blocked_move_is_counted_noop(self):
        env = Environment(seed=1)
        start = list(env.pacman_pos)
        self.assertNotIn("up", env.get_legal_moves(*start))
        env.step("up")
        self.assertEqual(env.pacman_pos, start)
        self.assertEqual(env.illegal_moves, 1)

    def test_unknown_move_raises(self):
        with self.assertRaises(ValueError):
            Environment(seed=1).step("north")


if __name__ == "__main__":
    unittest.main()
