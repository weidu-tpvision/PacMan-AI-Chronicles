"""Regression tests for fixes recorded in CODE_REVIEW.md."""

import importlib.util
import os
import threading
import time
import unittest

from agents.base import DecisionResult
from compare_baselines import run_episode
from core.environment import Environment
from pacman_game import PacmanGame

# Optional dependencies: skip the tests that need them instead of failing the whole run.
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = NUMPY_AVAILABLE and importlib.util.find_spec("torch") is not None
PYGAME_AVAILABLE = importlib.util.find_spec("pygame") is not None

if NUMPY_AVAILABLE:
    import numpy as np
    from rl.train_dqn import ReplayBuffer, STALL_STEPS, compute_reward, masked_next_actions
if TORCH_AVAILABLE:
    import torch


class InvalidMoveAgent:
    name = "Invalid move test agent"

    def reset(self):
        pass

    def decide(self, *args):
        return DecisionResult("unknown", error_msg="invalid action")


class WallMoveAgent:
    """Always answers "up", which is a wall at Pac-Man's spawn tile (9, 15)."""
    name = "Wall move test agent"

    def reset(self):
        pass

    def decide(self, *args):
        return DecisionResult("up")


class BlockingAgent:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def decide(self, *args):
        self.started.set()
        self.release.wait(timeout=2)
        return DecisionResult("left")


class StatefulBlockingAgent(BlockingAgent):
    """Records history in decide() like DQNAgent does; reset() clears it."""

    def __init__(self):
        super().__init__()
        self.history = []

    def decide(self, *args):
        self.started.set()
        self.release.wait(timeout=2)
        self.history.append(args[0])
        return DecisionResult("left")

    def reset(self):
        self.history.clear()


class ImmediateAgent:
    def decide(self, *args):
        return DecisionResult("right")


class TestReviewRegressions(unittest.TestCase):
    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional")
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

    @unittest.skipUnless(PYGAME_AVAILABLE, "pygame is optional")
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

    def test_reset_during_inflight_decision_leaves_agent_clean(self):
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
        agent = StatefulBlockingAgent()
        game.controllers = [{"agent": agent}]
        game.active_idx = 0

        game.start_decision_query(["left", "right"])
        self.assertTrue(agent.started.wait(timeout=1))
        # [R] / agent switch while the worker is still inside decide()
        game._invalidate_pending_decision()
        game._reset_agent(agent)  # must not block the UI thread
        agent.release.set()
        deadline = time.monotonic() + 2
        while game._agent_lock(agent).locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(agent.history, [], "stale decision left temporal state behind")
        self.assertIsNone(game.latest_decision_result)

    def test_tournament_counts_blocked_direction_once(self):
        result = run_episode(WallMoveAgent(), seed=1000, max_moves=1)
        self.assertEqual(result["invalid_actions"], 1)
        self.assertEqual(result["illegal_moves"], 1)

    def test_live_probe_rejects_reachable_host_without_systemone_endpoint(self):
        from compare_baselines import probe_live_endpoint
        from llm.decision_client import SystemOneAgent

        # Nothing listens on port 9; pretend /api/version answered so only the probe can catch it.
        backend = SystemOneAgent(host="http://127.0.0.1:9", timeout_sec=0.5)
        backend.is_ollama_online = lambda force_refresh=False: True
        self.assertFalse(probe_live_endpoint(backend))
        self.assertEqual(len(backend.latencies), 0)

    def test_registry_drops_score_badge_for_untrained_dqn(self):
        from unittest import mock
        import agents.registry as registry
        from agents.dqn_agent import DQNAgent

        untrained = lambda heuristics=True: DQNAgent(model_path="does_not_exist.pt", heuristics=heuristics)
        with mock.patch.object(registry, "DQNAgent", untrained),              mock.patch.object(registry, "load_tournament_scores", lambda: {"DQN (+inference heuristics)": 586.7}):
            controllers = registry.build_controllers(object(), "nimble")
        dqn = next(c for c in controllers if c["id"] == "dqn")
        self.assertIn("[UNTRAINED]", dqn["name"])
        self.assertNotIn("pts", dqn["badge"])


class TestWebArena(unittest.TestCase):
    def setUp(self):
        from web_arena import GameSession

        self.session = GameSession()

    def test_step_omits_static_walls_but_state_includes_them(self):
        self.assertNotIn("walls", self.session.step())
        self.assertTrue(self.session.get_state()["walls"])

    def test_dqn_receives_full_horizon_in_open_ended_session(self):
        captured = {}
        agent = self.session.current_agent
        original = agent.decide

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        agent.decide = spy
        for _ in range(3):
            self.session.step()
        self.assertEqual(captured["steps_remaining"], captured["horizon"])

    def test_concurrent_steps_are_serialized(self):
        import json
        import urllib.request
        import web_arena

        old_session = web_arena.session
        web_arena.session = self.session
        server = web_arena.ArenaHTTPServer(("127.0.0.1", 0), web_arena.ArenaHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/api/step"
            n_requests = 8

            def post():
                urllib.request.urlopen(urllib.request.Request(url, data=b"", method="POST"), timeout=10).read()

            workers = [threading.Thread(target=post) for _ in range(n_requests)]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            state_url = url.replace("/api/step", "/api/state")
            state = json.loads(urllib.request.urlopen(state_url, timeout=10).read())
            # Every step is applied exactly once (a game over would reset the counter).
            self.assertEqual(state["moves"], n_requests)
        finally:
            server.shutdown()
            server.server_close()
            web_arena.session = old_session


if __name__ == "__main__":
    unittest.main()
