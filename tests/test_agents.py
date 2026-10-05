"""
Automated Unit Tests for PacMan-AI-Chronicles Agent Suite.
Validates instantiation, AgentProtocol compliance, decision legality,
probability distributions, lifecycle reset, and stateful temporal tracking.
"""

import unittest
from typing import List, Tuple

from agents import (
    DQNAgent,
    GreedyHeuristicAgent,
    PretrainedQLearningAgent,
    RandomAgent,
    SystemOneBaselineAgent,
    TrainedQLearningAgent,
)
from agents.base import AgentProtocol, DecisionResult
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
        test_scenarios = [
            # Standard corridor
            {"pos": (9, 15), "ghosts": [(9, 7), (7, 7), (11, 7)], "pellets": {(1, 1), (2, 1)}, "legal": ["left", "right"]},
            # 4-way intersection
            {"pos": (6, 5), "ghosts": [(6, 1), (12, 5)], "pellets": {(6, 6)}, "legal": ["up", "down", "left", "right"]},
            # Single legal move (dead end)
            {"pos": (1, 1), "ghosts": [(9, 7)], "pellets": {(1, 2)}, "legal": ["right"]},
        ]

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

    def test_dqn_temporal_velocity_tracking_and_toroidal_preservation(self):
        """Verify DQNAgent retains velocity state across toroidal warp tunnel moves."""
        dqn = self.dqn_agent
        dqn.reset()
        self.assertIsNone(dqn.prev_pacman)
        self.assertIsNone(dqn.prev_ghosts)

        # Step 1: Normal move
        dqn.decide((1, 9), [(5, 7)], set(), ["left", "right"])
        self.assertEqual(dqn.prev_pacman, (1, 9))

        # Step 2: Move into leftmost warp tunnel cell (x=0)
        dqn.decide((0, 9), [(5, 7)], set(), ["left", "right"])
        self.assertEqual(dqn.prev_pacman, (0, 9))

        # Step 3: Wrap through tunnel to right side (x=GRID_WIDTH-1)
        # Toroidal distance is 1 (wraps around), so velocity should NOT be wiped
        dqn.decide((GRID_WIDTH - 1, 9), [(5, 7)], set(), ["left", "right"])
        self.assertIsNotNone(
            dqn.prev_pacman,
            "DQNAgent incorrectly wiped velocity tracking on horizontal warp tunnel traverse",
        )
        self.assertEqual(dqn.prev_pacman, (GRID_WIDTH - 1, 9))

        # Step 4: True jump/respawn across maze (dist > 2) should wipe temporal tracking
        dqn.decide((5, 18), [(5, 7)], set(), ["up", "down"])
        # Next decision should record new position
        self.assertEqual(dqn.prev_pacman, (5, 18))


if __name__ == "__main__":
    unittest.main()
