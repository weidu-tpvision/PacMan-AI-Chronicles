"""
Q-learning agents with linear feature approximations:
1. PretrainedQLearningAgent: Textbook TD-learning baseline (weights from rl/train_q_learning.py).
2. TrainedQLearningAgent: Policy-optimized agent (weights from rl/optimize_policy.py, CEM search).

Both agents share their feature extractors with their trainers via agents.features,
guaranteeing train / inference parity.
"""

import json
import logging
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

from agents.base import DecisionResult, margin_confidence, softmax
from agents.features import (
    ENHANCED_FEATURES,
    TEXTBOOK_FEATURES,
    enhanced_features,
    linear_q,
    textbook_features,
)
from agents.paths import resolve_weights_path

logger = logging.getLogger(__name__)

TEXTBOOK_DEFAULT_WEIGHTS: Dict[str, float] = {
    "ghost_1_step": -180.0,
    "ghost_2_step": -50.0,
    "ghost_3_step": -15.0,
    "ghost_safe_dist": 20.0,
    "eats_pellet": 35.0,
    "nearest_pellet_dist": -15.0,
    "reverse_penalty": -8.0,
}

ENHANCED_DEFAULT_WEIGHTS: Dict[str, float] = {
    "ghost_1_step": -450.0,
    "ghost_2_step": -140.0,
    "ghost_3_step": -20.0,
    "ghost_safe_dist": 3.0,
    "dead_end_trap": -280.0,
    "safe_junction": 20.0,
    "eats_pellet": 70.0,
    "poisoned_pellet": -240.0,
    "nearest_pellet_dist": -1.2,
    "reverse_penalty": -30.0,
}


def _load_weights(filename: str, expected: List[str], defaults: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
    """Load weights JSON; validates the feature schema. Returns (weights, loaded_from_file)."""
    path = resolve_weights_path(filename)
    if not os.path.exists(path):
        logger.warning("Weights file %s not found - using built-in defaults.", path)
        return dict(defaults), False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load weights from %s (%s) - using defaults.", path, exc)
        return dict(defaults), False
    if set(data) != set(expected):
        logger.warning(
            "Weights in %s have keys %s but expected %s - using defaults.", path, sorted(data), sorted(expected)
        )
        return dict(defaults), False
    return {k: float(v) for k, v in data.items()}, True


class _LinearQAgent:
    """Shared decision logic for linear Q-value agents."""

    temp: float = 15.0
    feature_fn: Callable = staticmethod(textbook_features)

    def __init__(self, name: str, category: str, weights: Dict[str, float], weights_loaded: bool):
        self.name = name
        self.category = category
        self.weights = weights
        self.weights_loaded = weights_loaded

    def reset(self) -> None:
        """Stateless agent - reset is a no-op."""

    def get_features(self, pacman_pos, ghost_positions, pellets, action, last_move=None) -> Dict[str, float]:
        return type(self).feature_fn(pacman_pos, ghost_positions, pellets, action, last_move)

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

        q_values = {
            m: linear_q(self.weights, self.get_features(pacman_pos, ghost_positions, pellets, m, last_move))
            for m in legal_moves
        }
        best_q = max(q_values.values())
        choice = random.choice([m for m in legal_moves if q_values[m] == best_q])

        dist = softmax(q_values, temp=self.temp)
        return DecisionResult(
            choice=choice,
            probabilities=dist,
            confidence=margin_confidence(dist),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            is_live=False,
        )


class PretrainedQLearningAgent(_LinearQAgent):
    """Textbook approximate Q-learning agent (classic features, TD-trained weights)."""

    temp = 15.0
    feature_fn = staticmethod(textbook_features)

    def __init__(
        self,
        name: str = "Q-Learning (Textbook Baseline)",
        weights_path: str = "learned_q_weights.json",
    ):
        weights, loaded = _load_weights(weights_path, TEXTBOOK_FEATURES, TEXTBOOK_DEFAULT_WEIGHTS)
        super().__init__(name, "Classic TD-Learning", weights, loaded)


class TrainedQLearningAgent(_LinearQAgent):
    """Policy-optimized agent: CEM-searched weights over topological graph features."""

    temp = 35.0
    feature_fn = staticmethod(enhanced_features)

    def __init__(
        self,
        name: str = "RL (Policy Optimized)",
        weights_path: str = "learned_enhanced_weights.json",
    ):
        weights, loaded = _load_weights(weights_path, ENHANCED_FEATURES, ENHANCED_DEFAULT_WEIGHTS)
        super().__init__(name, "Optimized RL Policy (CEM)", weights, loaded)
