"""
Greedy Heuristic Agent: Multi-objective rule-based policy.
Balances survival (BFS ghost evasion, dead-end trap detection) with food gathering.
"""

import time
from typing import List, Optional, Tuple

from agents.base import DecisionResult, softmax
from core.maze_data import (
    DIRECTIONS,
    GRID_WIDTH,
    MAZE_DEAD_ENDS,
    MAZE_DIST_MATRIX,
    OPPOSITE_DIRECTIONS,
)


class GreedyHeuristicAgent:
    """Expert rule-based planner balancing ghost evasion and food collection."""

    def __init__(self, name: str = "Greedy Heuristic"):
        self.name = name
        self.category = "Handcrafted Rules (0.01ms)"

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
            return DecisionResult("left", {}, 0.0, 0.0, False)

        px, py = pacman_pos
        scores = {}

        for m in legal_moves:
            dx, dy = DIRECTIONS[m]
            nx = (px + dx) % GRID_WIDTH
            ny = py + dy

            score = 0.0

            # 1. Ghost evasion via true BFS distance
            ghost_dists = [
                MAZE_DIST_MATRIX.get(((nx, ny), tuple(g)), 99) for g in ghost_positions
            ]
            min_ghost_dist = min(ghost_dists) if ghost_dists else 99

            if min_ghost_dist <= 1:
                score -= 1000.0  # Immediate lethal threat
            elif min_ghost_dist == 2:
                score -= 400.0   # Critical danger zone
            elif min_ghost_dist == 3:
                score -= 150.0   # Caution zone
            else:
                score += min_ghost_dist * 2.0  # Safe corridor preference

            # 2. Dead-end trap evasion
            if (nx, ny) in MAZE_DEAD_ENDS:
                trap_depth = MAZE_DEAD_ENDS[(nx, ny)]
                if min_ghost_dist <= trap_depth + 3:
                    score -= 500.0

            # 3. Pellet reward & proximity
            if (nx, ny) in pellets:
                score += 50.0

            if pellets:
                sample_pellets = list(pellets)[:30]
                min_pellet_dist = min(
                    MAZE_DIST_MATRIX.get(((nx, ny), p), 99) for p in sample_pellets
                )
                score -= min_pellet_dist * 2.0

            # 4. Anti-oscillation reversal penalty
            if last_move and m == OPPOSITE_DIRECTIONS.get(last_move) and min_ghost_dist > 3:
                score -= 15.0

            scores[m] = score

        best_move = max(legal_moves, key=lambda m: scores[m])
        lat = (time.perf_counter() - t0) * 1000.0

        dist = softmax(scores, temp=25.0)
        sorted_probs = sorted(dist.values(), reverse=True)
        conf = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        return DecisionResult(
            choice=best_move,
            probabilities=dist,
            confidence=round(conf, 3),
            latency_ms=lat,
            is_live=False,
        )
