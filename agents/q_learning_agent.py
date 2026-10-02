"""
Q-Learning agents with linear feature approximations:
1. PretrainedQLearningAgent: Textbook TD-learning baseline.
2. TrainedQLearningAgent: Policy-optimized RL agent achieving 920+ average score.
"""

import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

from agents.base import DecisionResult, softmax
from core.maze_data import (
    DIRECTIONS,
    GRID_WIDTH,
    MAZE_DEAD_ENDS,
    MAZE_DIST_MATRIX,
    MAZE_JUNCTIONS,
    OPPOSITE_DIRECTIONS,
)


def _resolve_weight_path(filename: str) -> str:
    """Find weights file in rl/weights/, current directory, or parent directory."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "rl", "weights", filename),
        os.path.join("rl", "weights", filename),
        os.path.join(os.path.dirname(__file__), "..", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return filename


class PretrainedQLearningAgent:
    """Textbook Q-Learning baseline agent with standard classic features."""

    def __init__(self, name: str = "Q-Learning (Textbook Baseline)"):
        self.name = name
        self.category = "Classic TD-Learning (0.01ms)"
        self.weights = {
            "ghost_1_step": -120.0,
            "ghost_2_step": -35.0,
            "eats_pellet": 25.0,
            "nearest_pellet_dist": -1.8,
            "reverse_penalty": -6.0,
        }

    def get_features(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        action: str,
        last_move: Optional[str] = None,
    ) -> Dict[str, float]:
        px, py = pacman_pos
        dx, dy = DIRECTIONS[action]
        nx, ny = (px + dx) % GRID_WIDTH, py + dy

        min_ghost_dist = min(abs(nx - gx) + abs(ny - gy) for gx, gy in ghost_positions)

        feats = {
            "ghost_1_step": 1.0 if min_ghost_dist <= 1 else 0.0,
            "ghost_2_step": 1.0 if min_ghost_dist == 2 else 0.0,
            "eats_pellet": 1.0 if (nx, ny) in pellets else 0.0,
            "nearest_pellet_dist": 0.0,
            "reverse_penalty": 1.0 if (last_move and action == OPPOSITE_DIRECTIONS.get(last_move)) else 0.0,
        }

        if pellets:
            sample_pellets = list(pellets)[:25]
            feats["nearest_pellet_dist"] = float(
                min(abs(nx - fx) + abs(ny - fy) for fx, fy in sample_pellets)
            )

        return feats

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

        q_values = {}
        for m in legal_moves:
            feats = self.get_features(pacman_pos, ghost_positions, pellets, m, last_move)
            q_values[m] = sum(self.weights.get(k, 0.0) * feats.get(k, 0.0) for k in self.weights)

        best_q = max(q_values.values())
        best_moves = [m for m in legal_moves if q_values[m] == best_q]
        choice = random.choice(best_moves)

        dist = softmax(q_values, temp=15.0)
        sorted_probs = sorted(dist.values(), reverse=True)
        conf = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        lat = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            choice=choice,
            probabilities=dist,
            confidence=round(conf, 3),
            latency_ms=lat,
            is_live=False,
        )


class TrainedQLearningAgent:
    """Policy-Optimized RL Agent: High-scoring policy (920+ pts) trained with topological features."""

    def __init__(
        self,
        weights_path: str = "learned_enhanced_weights.json",
        name: str = "RL (Policy Optimized)",
    ):
        self.name = name
        self.category = "Optimized RL Policy (920+ pts)"
        # Default high-performance weights in case file is absent
        self.weights = {
            "ghost_1_step": -455.74,
            "ghost_2_step": -139.63,
            "ghost_3_step": -19.17,
            "ghost_safe_dist": 2.70,
            "dead_end_trap": -275.63,
            "safe_junction": 20.93,
            "eats_pellet": 71.94,
            "poisoned_pellet": -241.15,
            "nearest_pellet_dist": -1.14,
            "reverse_penalty": -29.96,
        }

        resolved_path = _resolve_weight_path(weights_path)
        if not os.path.exists(resolved_path):
            resolved_path = _resolve_weight_path("learned_q_weights.json")

        if os.path.exists(resolved_path):
            try:
                with open(resolved_path, "r") as f:
                    self.weights = json.load(f)
            except Exception:
                pass

    def get_features(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        action: str,
        last_move: Optional[str] = None,
    ) -> Dict[str, float]:
        px, py = pacman_pos
        dx, dy = DIRECTIONS[action]
        nx, ny = (px + dx) % GRID_WIDTH, py + dy

        ghost_bfs_dists = [
            MAZE_DIST_MATRIX.get(((nx, ny), tuple(g)), 99) for g in ghost_positions
        ]
        min_ghost_bfs = min(ghost_bfs_dists) if ghost_bfs_dists else 99

        # Dead-end trap detection
        is_dead_end = (nx, ny) in MAZE_DEAD_ENDS
        depth = MAZE_DEAD_ENDS.get((nx, ny), 0)
        is_dead_end_trap = 1.0 if (is_dead_end and min_ghost_bfs <= depth + 3) else 0.0

        # Safe junction mobility
        is_safe_junction = 1.0 if ((nx, ny) in MAZE_JUNCTIONS and min_ghost_bfs > 2) else 0.0

        # Pellet evaluation: safe vs poisoned trap
        eats_pellet = 1.0 if (nx, ny) in pellets else 0.0
        poisoned_pellet = 1.0 if (eats_pellet and (min_ghost_bfs <= 2 or is_dead_end_trap)) else 0.0

        # Nearest pellet distance via graph BFS
        nearest_pellet_dist = 0.0
        if pellets:
            sample_p = list(pellets)[:35]
            nearest_pellet_dist = float(
                min(MAZE_DIST_MATRIX.get(((nx, ny), p), 99) for p in sample_p)
            )

        rev_pen = (
            1.0
            if (last_move and action == OPPOSITE_DIRECTIONS.get(last_move) and min_ghost_bfs > 3)
            else 0.0
        )

        return {
            "ghost_1_step": 1.0 if min_ghost_bfs <= 1 else 0.0,
            "ghost_2_step": 1.0 if min_ghost_bfs == 2 else 0.0,
            "ghost_3_step": 1.0 if min_ghost_bfs == 3 else 0.0,
            "ghost_safe_dist": float(min_ghost_bfs) if min_ghost_bfs > 3 else 0.0,
            "dead_end_trap": is_dead_end_trap,
            "safe_junction": is_safe_junction,
            "eats_pellet": eats_pellet,
            "poisoned_pellet": poisoned_pellet,
            "nearest_pellet_dist": nearest_pellet_dist,
            "reverse_penalty": rev_pen,
        }

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

        q_values = {}
        for m in legal_moves:
            feats = self.get_features(pacman_pos, ghost_positions, pellets, m, last_move)
            q_values[m] = sum(self.weights.get(k, 0.0) * feats.get(k, 0.0) for k in self.weights)

        best_q = max(q_values.values())
        best_moves = [m for m in legal_moves if q_values[m] == best_q]
        choice = random.choice(best_moves)

        dist = softmax(q_values, temp=35.0)
        sorted_probs = sorted(dist.values(), reverse=True)
        conf = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        lat = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            choice=choice,
            probabilities=dist,
            confidence=round(conf, 3),
            latency_ms=lat,
            is_live=False,
        )
