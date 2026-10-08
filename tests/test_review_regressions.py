"""Regression tests for fixes recorded in CODE_REVIEW.md."""

import unittest
import threading
import time
import os

from agents.base import DecisionResult
from compare_baselines import run_episode
from rl.train_dqn import ReplayBuffer, STALL_STEPS, compute_reward, masked_next_actions
from pacman_game import PacmanGame
from core.environment import Environment

try:
    import numpy as np
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False


class InvalidMoveAgent:
    name = "Invalid move test agent"

    def reset(self):
        pass

    def decide(self, *args):
        return DecisionResult("unknown", error_msg="invalid action")


class BlockingAgent:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def decide(self, *args):
        self.started.set()
        self.release.wait(timeout=2)
        return DecisionResult("left")


class ImmediateAgent:
    def decide(self, *args):
        return DecisionResult("right")


class TestReviewRegressions(unittest.TestCase):
    def test_stall_cutoff_applies_penalty(self):
        reward, stalled = compute_reward(None, "left", False, False, False, 2, STALL_STEPS - 1)
        self.assertFalse(stalled)
        self.assertEqual(reward, -0.5)

        reward, stalled = compute_reward(None, "left", False, False, False, 2, STALL_STEPS)
        self.assertTrue(stalled)
        self.assertEqual(reward, -50.5)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional")
    def test_replay_preserves_next_action_mask_and_masks_argmax(self):
        buffer = ReplayBuffer(capacity=4)
        state = np.zeros((1,), dtype=np.float32)
        mask = np.array([True, False, True, False])
        buffer.push(state, 0, 0.0, state, 0.0, mask)
        sample = buffer.sample(1)
        sampled_mask = sample[5]
        self.assertTrue(torch.equal(sampled_mask[0], torch.tensor(mask)))

        q_values = torch.tensor([[1.0, 100.0, 2.0, 50.0]])
        action = masked_next_actions(q_values, sampled_mask)
        self.assertEqual(action.item(), 2)

    def test_tournament_records_invalid_choice_and_continues(self):
        result = run_episode(InvalidMoveAgent(), seed=1000, max_moves=1)
        self.assertEqual(result["invalid_actions"], 1)
        self.assertEqual(result["illegal_moves"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["moves"], 1)

    def test_respawn_resets_shared_environment_clock(self):
        env = Environment(seed=1)
        env.mode_step = 34
        env.respawn()
        self.assertEqual(env.mode_step, 0)

    def test_pygame_collision_path_uses_environment_respawn(self):
        import pygame

        old_driver = os.environ.get("SDL_VIDEODRIVER")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        game = None
        try:
            game = PacmanGame(force_mock=True)
            game.env.step = lambda move: (True, False, False)
            legal = game.env.get_legal_moves(*game.pacman_pos)
            game.latest_decision_result = DecisionResult(legal[0])
            game.update(0.0)
            self.assertEqual(game.lives, 2)
            self.assertEqual(game.env.mode_step, 0)
            self.assertEqual(game.pacman_pos, game.env.pacman_pos)
        finally:
            pygame.quit()
            if old_driver is None:
                os.environ.pop("SDL_VIDEODRIVER", None)
            else:
                os.environ["SDL_VIDEODRIVER"] = old_driver

    def test_stale_async_decision_is_discarded_after_invalidation(self):
        game = PacmanGame.__new__(PacmanGame)
        game.decision_lock = threading.RLock()
        game.decision_epoch = 0
        game.pending_decision = False
        game.latest_decision_result = None
        game.active_decision = None
        game.pacman_pos = [9, 15]
        game.ghost_positions = [[9, 7]]
        game.pellets = {(1, 1)}
        game.last_move = "left"
        old_agent = BlockingAgent()
        new_agent = ImmediateAgent()
        game.controllers = [{"agent": old_agent}, {"agent": new_agent}]
        game.active_idx = 0

        game.start_decision_query(["left", "right"])
        self.assertTrue(old_agent.started.wait(timeout=1))
        game._invalidate_pending_decision()
        game.active_idx = 1
        game.start_decision_query(["left", "right"])
        deadline = time.monotonic() + 1
        while game.pending_decision and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(game.latest_decision_result)
        self.assertEqual(game.latest_decision_result.choice, "right")

        old_agent.release.set()
        time.sleep(0.02)
        self.assertEqual(game.latest_decision_result.choice, "right")


if __name__ == "__main__":
    unittest.main()
