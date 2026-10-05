"""
Deep Q-Network Agent: Executes forward passes on the trained PyTorch CNN model.
Uses unentangled 6-channel state encoding with velocity/momentum awareness.
"""

import collections
import logging
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
from agents.base import DecisionResult, softmax

try:
    import torch
    from rl.dqn_model import (
        ACTION_TO_IDX,
        NUM_CHANNELS,
        FRAME_STACK_SIZE,
        PacmanDQN,
        encode_frame,
        encode_state,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.maze_data import GRID_WIDTH, OPPOSITE_DIRECTIONS


def _resolve_model_path(filename: str) -> str:
    """Find checkpoint in rl/weights/, current directory, or parent directory."""
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


class DQNAgent:
    """Deep Q-Network Agent running a convolutional neural network with unentangled multi-channel velocity tracking."""

    def __init__(
        self,
        model_path: str = "dqn_pacman.pt",
        name: str = "Deep Q-Network (PyTorch DQN)",
        k: int = NUM_CHANNELS,
    ):
        self.name = name
        self.category = "Deep Neural RL (PyTorch CNN)"
        self.k = k
        self.prev_pacman: Optional[Tuple[int, int]] = None
        self.prev_ghosts: Optional[List[Tuple[int, int]]] = None
        self.model_loaded = False
        self.model = None

        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.model = PacmanDQN(in_channels=NUM_CHANNELS).to(self.device)
            resolved = _resolve_model_path(model_path)
            if os.path.exists(resolved):
                try:
                    self.model.load_state_dict(
                        torch.load(resolved, map_location=self.device, weights_only=True)
                    )
                    self.model_loaded = True
                except Exception as exc:
                    logging.warning("Failed to load DQN model weights from %s: %s", resolved, exc)
            self.model.eval()

    def reset(self):
        """Reset temporal state tracking for a new game episode."""
        self.prev_pacman = None
        self.prev_ghosts = None

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        t0 = time.perf_counter()

        if not TORCH_AVAILABLE or self.model is None or not legal_moves:
            m = random.choice(legal_moves) if legal_moves else "left"
            return DecisionResult(m, {m: 1.0}, 1.0, 0.05, False)

        # Detect new episode or respawn if position jumped significantly (toroidal aware)
        if self.prev_pacman is not None:
            dx = abs(pacman_pos[0] - self.prev_pacman[0])
            dx = min(dx, GRID_WIDTH - dx)
            dy = abs(pacman_pos[1] - self.prev_pacman[1])
            if dx + dy > 2:
                self.prev_pacman = None
                self.prev_ghosts = None

        state_arr = encode_state(
            pacman_pos=pacman_pos,
            ghost_positions=ghost_positions,
            pellets=pellets,
            prev_pacman_pos=self.prev_pacman,
            prev_ghost_positions=self.prev_ghosts,
        )

        # Update previous positions for next step
        self.prev_pacman = pacman_pos
        self.prev_ghosts = [tuple(g) for g in ghost_positions]

        with torch.no_grad():
            s_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
            raw_q = self.model(s_tensor).squeeze(0)

        legal_q = {m: raw_q[ACTION_TO_IDX[m]].item() for m in legal_moves}
        if last_move and len(legal_moves) > 1:
            opp = OPPOSITE_DIRECTIONS.get(last_move)
            if opp in legal_q:
                legal_q[opp] -= 1.0

        best_q = max(legal_q.values())
        best_moves = [m for m in legal_moves if legal_q[m] == best_q]
        choice = random.choice(best_moves)

        dist = softmax(legal_q, temp=15.0)
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
