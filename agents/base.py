"""
Base classes, data models, and probability utilities for Pac-Man agents.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Protocol, Set, Tuple


@dataclass
class DecisionResult:
    """Standardized decision payload for all agents."""
    choice: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    latency_ms: float = 0.0
    is_live: bool = False
    raw_response: Optional[dict] = field(default_factory=dict)
    error_msg: Optional[str] = None


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
    """Compute temperature-scaled softmax probabilities over candidate actions."""
    if not scores:
        return {}
    if temp <= 0:
        best_k = max(scores, key=scores.get)
        return {k: (1.0 if k == best_k else 0.0) for k in scores}

    max_s = max(scores.values())
    exp_s = {k: math.exp(min(max((v - max_s) / temp, -50.0), 50.0)) for k, v in scores.items()}
    total = sum(exp_s.values())
    if total == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: round(v / total, 4) for k, v in exp_s.items()}
