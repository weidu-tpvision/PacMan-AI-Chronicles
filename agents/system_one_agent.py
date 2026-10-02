"""
System 1 Baseline Agent: Wrapper adapting SystemOneAgent (Ollama / Jev)
to the standard agent interface.
"""

from typing import List, Optional, Tuple

from agents.base import DecisionResult


class SystemOneBaselineAgent:
    """Wrapper adapting the Ollama System 1 decision client into the agent protocol."""

    def __init__(self, agent, name: Optional[str] = None):
        self.agent = agent
        self.name = name or f"System 1 ({getattr(agent, 'model', 'default')})"
        self.category = "Neural Zero-Shot (~90ms)"

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        return self.agent.decide_move(
            pacman_pos, ghost_positions, pellets, legal_moves, last_move
        )
