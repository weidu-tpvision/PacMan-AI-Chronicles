"""
Deep Q-Network Agent: Executes forward passes on the trained PyTorch CNN model.
The observation is a function of the current game state only (see rl/dqn_model.py).

Two evaluation modes:
  heuristics=False  -> pure greedy argmax over the network's Q-values (the policy that
                       training validates and checkpoints; reported as "DQN (raw)").
  heuristics=True   -> adds inference-time reversal and anti-orbit penalties on top of
                       the network (reported separately as "DQN (+heuristics)").
"""

import collections
import logging
import os
import random
import time
from typing import List, Optional, Tuple

from agents.base import DecisionResult, margin_confidence, softmax
from agents.paths import resolve_weights_path
from core.environment import DEFAULT_MAX_STEPS
from core.maze_data import DIRECTIONS, GRID_WIDTH, OPPOSITE_DIRECTIONS

try:
    import torch
    from rl.dqn_model import ACTION_TO_IDX, PacmanDQN, encode_state
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on torch-less installs
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Q-values are in the trainer's reward units, where a pellet is worth 1
# (SCORE_PELLET * REWARD_SCALE); the optional inference heuristics use the same units.
REVERSAL_PENALTY = 0.1
ORBIT_PENALTY_PER_VISIT = 1.5
PROBABILITY_TEMPERATURE = 1.0  # softmax temperature for the displayed distribution


class DQNAgent:
    """Deep Q-Network agent over the full-resolution dueling CNN.

    The observation needs no agent history when callers pass the environment context
    (`ghost_dirs`, `mode_step`, `steps_without_pellet`, `steps_remaining`; every in-repo
    caller does). Small per-episode state remains only for the fallbacks used when that
    context is omitted and for the optional anti-orbit heuristic; call reset() on a new
    episode or respawn to clear it.
    """

    def __init__(
        self,
        model_path: str = "dqn_pacman.pt",
        name: str = "Deep Q-Network (PyTorch DQN)",
        heuristics: bool = True,
        require_weights: bool = False,
    ):
        """
        Args:
            model_path: checkpoint filename / path (resolved via rl/weights/).
            heuristics: apply inference-time reversal / anti-orbit penalties.
            require_weights: raise RuntimeError if torch or the checkpoint is unavailable
                (use for benchmarks). Otherwise the agent degrades to a uniform random
                policy and tags its name with "[UNTRAINED]" so results cannot be mistaken.
        """
        self.name = name
        self.category = "Deep Neural RL (PyTorch CNN)"
        self.heuristics = heuristics
        self.prev_ghosts: Optional[List[Tuple[int, int]]] = None  # only for heading inference
        self.recent_positions: collections.deque = collections.deque(maxlen=16)
        self.last_pellet_count: Optional[int] = None
        self._mode_step = 0
        self._decision_count = 0
        self._steps_without_pellet = 0
        self.model_loaded = False
        self.model = None
        self.model_path = resolve_weights_path(model_path)

        problem: Optional[str] = None
        if not TORCH_AVAILABLE:
            problem = "PyTorch is not installed"
        elif not os.path.exists(self.model_path):
            problem = f"checkpoint not found at {self.model_path}"
        else:
            self.device = torch.device("cpu")
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                model = PacmanDQN().to(self.device)
                model.load_state_dict(state_dict)
                model.eval()
                self.model = model
                self.model_loaded = True
            except RuntimeError as exc:  # parameter names / shapes differ
                problem = (f"checkpoint {self.model_path} does not match the current network / observation "
                           f"encoding (trained by an older version?) - retrain with rl/train_dqn.py ({exc.__class__.__name__})")
            except Exception as exc:  # corrupted / unreadable checkpoint
                problem = f"failed to load checkpoint {self.model_path}: {exc}"

        if problem:
            if require_weights:
                raise RuntimeError(f"DQNAgent unavailable: {problem}")
            logger.warning("DQNAgent running UNTRAINED random fallback: %s", problem)
            self.name = f"{name} [UNTRAINED]"

    def reset(self):
        """Reset temporal state tracking for a new game episode."""
        self.prev_ghosts = None
        self.recent_positions.clear()
        self.last_pellet_count = None
        self._mode_step = 0
        self._decision_count = 0
        self._steps_without_pellet = 0

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
        *,
        mode_step: Optional[int] = None,
        ghost_dirs: Optional[List[str]] = None,
        steps_without_pellet: Optional[int] = None,
        steps_remaining: Optional[int] = None,
        horizon: int = DEFAULT_MAX_STEPS,
    ) -> DecisionResult:
        t0 = time.perf_counter()

        if not legal_moves:
            return DecisionResult("left", {}, 0.0, 0.0, False, error_msg="no legal moves")
        if not self.model_loaded:
            m = random.choice(legal_moves)
            p = 1.0 / len(legal_moves)
            return DecisionResult(
                m, {mv: p for mv in legal_moves}, 0.0, (time.perf_counter() - t0) * 1000.0, False,
                error_msg="untrained fallback",
            )

        # Update visit history to detect and break local corridor loops
        curr_pellets = len(pellets)
        if self.last_pellet_count is None or curr_pellets != self.last_pellet_count:
            self.recent_positions.clear()
            self._steps_without_pellet = 0
            self.last_pellet_count = curr_pellets
        else:
            self._steps_without_pellet += 1
        self.recent_positions.append(tuple(pacman_pos))

        if ghost_dirs is None:
            inferred_dirs = []
            for index, pos in enumerate(ghost_positions):
                previous = self.prev_ghosts[index] if self.prev_ghosts and index < len(self.prev_ghosts) else pos
                dx = (pos[0] - previous[0]) % GRID_WIDTH
                if dx == GRID_WIDTH - 1:
                    dx = -1
                delta = (dx, pos[1] - previous[1])
                inferred_dirs.append({(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}.get(delta, "up"))
            ghost_dirs = inferred_dirs

        effective_mode_step = self._mode_step if mode_step is None else mode_step
        effective_stall_steps = self._steps_without_pellet if steps_without_pellet is None else steps_without_pellet
        effective_remaining = max(0, horizon - self._decision_count) if steps_remaining is None else steps_remaining

        grid, scalars = encode_state(
            pacman_pos=pacman_pos,
            ghost_positions=ghost_positions,
            pellets=pellets,
            last_move=last_move,
            ghost_dirs=ghost_dirs,
            mode_step=effective_mode_step,
            steps_without_pellet=effective_stall_steps,
            steps_remaining=effective_remaining,
            horizon=horizon,
        )

        self.prev_ghosts = [tuple(g) for g in ghost_positions]
        self._mode_step = effective_mode_step + 1
        self._decision_count += 1

        with torch.no_grad():
            raw_q = self.model(
                torch.from_numpy(grid).unsqueeze(0).to(self.device),
                torch.from_numpy(scalars).unsqueeze(0).to(self.device),
            ).squeeze(0)

        legal_q = {m: raw_q[ACTION_TO_IDX[m]].item() for m in legal_moves}

        if self.heuristics and len(legal_moves) > 1:
            if last_move:
                opp = OPPOSITE_DIRECTIONS.get(last_move)
                if opp in legal_q:
                    legal_q[opp] -= REVERSAL_PENALTY

            # Anti-orbit penalty: stepping into a tile visited repeatedly without eating
            if len(self.recent_positions) >= 4:
                for m in legal_moves:
                    dx, dy = DIRECTIONS[m]
                    nxt = ((pacman_pos[0] + dx) % GRID_WIDTH, pacman_pos[1] + dy)
                    visit_count = self.recent_positions.count(nxt)
                    if visit_count >= 2:
                        legal_q[m] -= visit_count * ORBIT_PENALTY_PER_VISIT

        best_q = max(legal_q.values())
        choice = random.choice([m for m in legal_moves if legal_q[m] == best_q])

        dist = softmax(legal_q, temp=PROBABILITY_TEMPERATURE)
        return DecisionResult(
            choice=choice,
            probabilities=dist,
            confidence=margin_confidence(dist),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            is_live=False,
        )
