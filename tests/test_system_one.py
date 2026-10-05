"""
Automated Unit Tests for System 1 Decision Client Subsystem.
Validates spatial prompt building, heuristic fallback simulation,
single-move shortcuts, and rolling telemetry tracking.
"""

import unittest

from agents.base import DecisionResult
from core.maze_data import DIRECTIONS, MAZE_LAYOUT, START_POSITIONS
from llm.decision_client import SystemOneAgent


class TestSystemOneDecisionEngine(unittest.TestCase):
    """Verifies System 1 decision engine and offline heuristic simulator."""

    def setUp(self):
        self.agent = SystemOneAgent(model="nimble", prefer_live=False)

        self.pellets = set()
        self.walls = set()
        for y, row in enumerate(MAZE_LAYOUT):
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.add((x, y))
                elif char == ".":
                    self.pellets.add((x, y))

    def test_spatial_description_generation(self):
        """Verify spatial natural language summary reflects positions and threats."""
        pacman_pos = (9, 15)
        ghost_positions = [(9, 7), (7, 7)]
        legal_moves = ["left", "right"]

        desc = self.agent.build_spatial_description(
            pacman_pos=pacman_pos,
            ghost_positions=ghost_positions,
            legal_moves=legal_moves,
            pellets=self.pellets,
        )

        self.assertIn("Pacman at position (9, 15)", desc)
        self.assertIn("Legal corridor moves: left, right", desc)
        self.assertIn("Ghost 1 at distance", desc)

    def test_heuristic_fallback_decision(self):
        """Verify offline heuristic decision generates valid legal move and probabilities."""
        test_pos = (8, 15)
        legal_moves = [
            d for d, (dx, dy) in DIRECTIONS.items()
            if (test_pos[0] + dx, test_pos[1] + dy) not in self.walls
        ]
        ghost_positions = [tuple(g) for g in START_POSITIONS["ghosts"]]

        result = self.agent.decide_move(
            pacman_pos=test_pos,
            ghost_positions=ghost_positions,
            pellets=self.pellets,
            legal_moves=legal_moves,
        )

        self.assertIsInstance(result, DecisionResult)
        self.assertIn(result.choice, legal_moves)
        self.assertFalse(result.is_live)
        self.assertGreater(result.latency_ms, 0.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, delta=0.03)

    def test_single_legal_move_fast_path(self):
        """Verify corridor with only 1 legal move returns immediately without inference."""
        result = self.agent.decide_move(
            pacman_pos=(1, 1),
            ghost_positions=[(9, 7)],
            pellets=self.pellets,
            legal_moves=["down"],
        )
        self.assertEqual(result.choice, "down")
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.probabilities, {"down": 1.0})

    def test_rolling_telemetry(self):
        """Verify latency tracking respects the rolling window."""
        self.agent.reset()
        self.assertEqual(len(self.agent.latencies), 0)

        for _ in range(5):
            self.agent.decide_move(
                pacman_pos=(9, 15),
                ghost_positions=[(9, 7)],
                pellets=self.pellets,
                legal_moves=["up", "down"],
            )

        self.assertEqual(len(self.agent.latencies), 5)
        self.assertGreater(self.agent.avg_latency, 0.0)


if __name__ == "__main__":
    unittest.main()
