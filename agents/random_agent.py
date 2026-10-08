"""
Random Baseline Agent: Uniform random choice among legal corridor moves.
Serves as the empirical lower bound and noise floor.
"""

import random
import time
from typing import List, Optional, Tuple

from agents.base import DecisionResult


class RandomAgent:
    """Agent that chooses uniformly at random among all legal corridor actions."""

    def __init__(self, name: str = "Random Agent (Baseline)"):
        self.name = name
        self.category = "Baseline (Lower Bound)"

    def reset(self) -> None:
        """Stateless agent - reset is a no-op."""
        pass

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        t0 = time.perf_counter()
        if not legal_moves:
            return DecisionResult("left", {}, 0.0, 0.0, False, error_msg="no legal moves")

        choice = random.choice(legal_moves)
        prob = 1.0 / len(legal_moves)
        dist = {m: round(prob, 4) for m in legal_moves}
        lat = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            choice=choice,
            probabilities=dist,
            confidence=0.0,
            latency_ms=lat,
            is_live=False,
        )
