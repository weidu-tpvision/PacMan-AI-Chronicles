"""
Automated Unit Tests for PacMan-AI-Chronicles Agent Suite.
Validates instantiation, AgentProtocol compliance, decision legality,
probability distributions, lifecycle reset, and stateful temporal tracking.
"""

import importlib.util
import unittest

from agents import (
    DQNAgent,
    GreedyHeuristicAgent,
    PretrainedQLearningAgent,
    RandomAgent,
    SystemOneBaselineAgent,
    TrainedQLearningAgent,
)
from agents.base import DecisionResult
from core.environment import Environment
from core.maze_data import GRID_WIDTH
from llm.decision_client import SystemOneAgent


class TestAgentSuite(unittest.TestCase):
    """Verifies all agent paradigms conform to AgentProtocol and make valid decisions."""

    def setUp(self):
        self.random_agent = RandomAgent()
        self.greedy_agent = GreedyHeuristicAgent()
        self.textbook_q = PretrainedQLearningAgent()
        self.trained_q = TrainedQLearningAgent()
        self.dqn_agent = DQNAgent()
        self.sys1_backend = SystemOneAgent(prefer_live=False)
        self.sys1_agent = SystemOneBaselineAgent(self.sys1_backend)

        self.agents = [
            self.random_agent,
            self.greedy_agent,
            self.textbook_q,
            self.trained_q,
            self.dqn_agent,
            self.sys1_agent,
        ]

    def test_agent_protocol_attributes(self):
        """Verify each agent implements name, category, reset(), and decide()."""
        for agent in self.agents:
            self.assertTrue(hasattr(agent, "name"), f"{agent} missing name attribute")
            self.assertTrue(hasattr(agent, "category"), f"{agent} missing category attribute")
            self.assertTrue(callable(getattr(agent, "reset", None)), f"{agent} missing reset method")
            self.assertTrue(callable(getattr(agent, "decide", None)), f"{agent} missing decide method")
            # Verify reset execution without exception
            agent.reset()

    def test_decision_legality_and_bounds(self):
        """Verify agents always choose from legal moves and provide valid probability sums."""
        env = Environment()
        # Legal moves come from the real maze so scenarios cannot drift from the layout.
        test_scenarios = [
            # Horizontal corridor at the spawn tile
            {"pos": (9, 15), "ghosts": [(9, 7), (7, 7), (11, 7)], "pellets": {(1, 1), (2, 1)}},
            # 4-way junction
            {"pos": (4, 3), "ghosts": [(4, 1), (8, 3)], "pellets": {(4, 4)}},
            # Corner (two exits)
            {"pos": (1, 1), "ghosts": [(9, 7)], "pellets": {(1, 2)}},
            # Tunnel edge: wraps to the far side
            {"pos": (0, 9), "ghosts": [(9, 7)], "pellets": {(4, 9)}},
        ]
        for scenario in test_scenarios:
            scenario["legal"] = env.get_legal_moves(*scenario["pos"])
        self.assertEqual(sorted(test_scenarios[0]["legal"]), ["left", "right"])
        self.assertEqual(len(test_scenarios[1]["legal"]), 4)
        self.assertEqual(sorted(test_scenarios[2]["legal"]), ["down", "right"])
        self.assertEqual(sorted(test_scenarios[3]["legal"]), ["left", "right"])

        for agent in self.agents:
            for scenario in test_scenarios:
                res = agent.decide(
                    pacman_pos=scenario["pos"],
                    ghost_positions=scenario["ghosts"],
                    pellets=scenario["pellets"],
                    legal_moves=scenario["legal"],
                )
                self.assertIsInstance(res, DecisionResult, f"{agent.name} did not return DecisionResult")
                self.assertIn(
                    res.choice,
                    scenario["legal"],
                    f"{agent.name} selected illegal move '{res.choice}' not in {scenario['legal']}",
                )
                self.assertGreaterEqual(res.latency_ms, 0.0, f"{agent.name} reported negative latency")
                self.assertGreaterEqual(res.confidence, 0.0, f"{agent.name} reported negative confidence")
                self.assertLessEqual(res.confidence, 1.0, f"{agent.name} confidence exceeded 1.0")

                if res.probabilities:
                    prob_sum = sum(res.probabilities.values())
                    self.assertAlmostEqual(
                        prob_sum,
                        1.0,
                        delta=0.03,
                        msg=f"{agent.name} probabilities do not sum to 1.0 (got {prob_sum})",
                    )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_dqn_decision_depends_only_on_current_state(self):
        """With environment context passed, earlier calls must not influence a decision."""
        import os
        import tempfile

        import torch
        from rl.dqn_model import PacmanDQN

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "random_init.pt")
            torch.save(PacmanDQN().state_dict(), path)
            dqn = DQNAgent(model_path=path, heuristics=False, require_weights=True)

        env = Environment(seed=7)
        ctx = dict(mode_step=3, ghost_dirs=["up", "left", "right"], steps_without_pellet=2, steps_remaining=250)
        state = ((9, 15), [(9, 7), (7, 7), (11, 7)], env.pellets, ["left", "right"], "left")
        first = dqn.decide(*state, **ctx)
        # Unrelated states in between, including a tunnel wrap
        dqn.decide((0, 9), [(5, 7)], set(), ["left", "right"], "left", **ctx)
        dqn.decide((GRID_WIDTH - 1, 9), [(5, 7)], set(), ["left", "right"], "left", **ctx)
        again = dqn.decide(*state, **ctx)
        self.assertEqual(first.probabilities, again.probabilities)


if __name__ == "__main__":
    unittest.main()
