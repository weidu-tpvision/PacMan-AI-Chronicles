"""
Human player agent for manual keyboard control.
Conforms to AgentProtocol.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from agents.base import AgentProtocol, DecisionResult


class HumanAgent:
    """Manual keyboard controller for human play."""

    name: str = "Human Player"
    category: str = "Manual Input"

    def __init__(self):
        self.desired_direction: Optional[str] = None
        self.current_heading: Optional[str] = None

    def set_desired_direction(self, direction: Optional[str]) -> None:
        """Buffer a player directional input ('up', 'down', 'left', 'right')."""
        if direction in ("up", "down", "left", "right", None):
            self.desired_direction = direction

    def get_intended_move(
        self, legal_moves: List[str], current_heading: Optional[str] = None
    ) -> Optional[str]:
        """Return the player's intended move if legal, else None."""
        if self.desired_direction and self.desired_direction in legal_moves:
            chosen = self.desired_direction
            self.desired_direction = None  # Turn consumed
            self.current_heading = chosen
            return chosen
        heading = current_heading or self.current_heading
        if heading and heading in legal_moves:
            self.current_heading = heading
            return heading
        return None

    def reset(self) -> None:
        """Reset internal directional buffers between episodes or on respawn."""
        self.desired_direction = None
        self.current_heading = None

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: Set[Tuple[int, int]],
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        """Select move based on buffered human input or current corridor heading."""
        choice = self.get_intended_move(legal_moves, last_move)
        is_legal = True
        if choice is None:
            # Blocked against wall: keep attempting current heading so ghosts advance
            choice = last_move or self.current_heading or (legal_moves[0] if legal_moves else "left")
            is_legal = choice in legal_moves
        else:
            self.current_heading = choice

        probs = {d: (1.0 if d == choice else 0.0) for d in ("up", "down", "left", "right")}
        return DecisionResult(
            choice=choice,
            probabilities=probs,
            confidence=1.0 if is_legal else 0.0,
            latency_ms=0.1,
            is_live=True,
            raw_response={"source": "manual_keyboard", "is_legal": is_legal},
        )
