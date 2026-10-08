"""
Base classes, data models, and probability utilities for Pac-Man agents.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple


@dataclass
class DecisionResult:
    """Standardized decision payload for all agents.

    latency_simulated: True when `latency_ms` is a synthetic value (e.g. the offline
    System 1 simulator) rather than a measured wall-clock latency.
    """
    choice: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    latency_ms: float = 0.0
    is_live: bool = False
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None
    latency_simulated: bool = False


class AgentProtocol(Protocol):
    """Protocol defining the decision and lifecycle interface for all Pac-Man agents."""
    name: str
    category: str

    def reset(self) -> None:
        """Reset internal temporal / episodic state if applicable."""
        ...

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: Set[Tuple[int, int]],
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        ...


def softmax(scores: Dict[str, float], temp: float = 20.0) -> Dict[str, float]:
    """Compute temperature-scaled softmax probabilities over candidate actions (sums to 1.0)."""
    if not scores:
        return {}
    if temp <= 0:
        best_k = max(scores, key=scores.get)
        return {k: (1.0 if k == best_k else 0.0) for k in scores}

    max_s = max(scores.values())
    exp_s = {k: math.exp(max((v - max_s) / temp, -50.0)) for k, v in scores.items()}
    total = sum(exp_s.values())
    return {k: v / total for k, v in exp_s.items()}


def margin_confidence(probabilities: Dict[str, float]) -> float:
    """Top-1 minus top-2 probability margin in [0, 1]."""
    p = sorted(probabilities.values(), reverse=True)
    if not p:
        return 0.0
    conf = p[0] - p[1] if len(p) > 1 else 1.0
    return round(max(0.0, min(1.0, conf)), 3)
