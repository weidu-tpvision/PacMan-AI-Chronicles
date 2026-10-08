"""
Deep Q-Network Agent: Executes forward passes on the trained PyTorch CNN model.
Uses identity-preserving spatial channels with motion and environment-phase features.

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
    from rl.dqn_model import ACTION_TO_IDX, NUM_CHANNELS, PacmanDQN, detect_architecture, encode_state
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on torch-less installs
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

REVERSAL_PENALTY = 1.0
ORBIT_PENALTY_PER_VISIT = 20.0


class DQNAgent:
    """Deep Q-Network Agent running a convolutional neural network with unentangled multi-channel velocity tracking.

    Lifecycle contract: callers MUST call reset() on every new episode and on every
    life-loss respawn (both arenas, the tournament runner, and the trainers already do).
    Between resets, temporal history (previous positions, orbit memory, mode clock) is
    assumed continuous; there is intentionally no in-agent teleport heuristic, which
    could silently leak stale pre-death state for deaths near the spawn point.
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
        self.prev_pacman: Optional[Tuple[int, int]] = None
        self.prev_ghosts: Optional[List[Tuple[int, int]]] = None
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
                # Architecture (full-resolution "deep" vs legacy "pool") and dueling head are
                # read from the parameter names, so every shipped checkpoint generation loads.
                arch, dueling = detect_architecture(state_dict)
                model = PacmanDQN(in_channels=NUM_CHANNELS, dueling=dueling, arch=arch).to(self.device)
                self.arch = arch
                first_weight = state_dict.get("conv.0.weight")
                if first_weight is not None and first_weight.shape[1] == 6:
                    # Expand legacy merged-ghost channels into the new per-ghost planes.
                    migrated = model.state_dict()
                    for key, value in state_dict.items():
                        if key != "conv.0.weight" and key in migrated and migrated[key].shape == value.shape:
                            migrated[key] = value
                    expanded = torch.zeros_like(migrated["conv.0.weight"])
                    mapping = {0: [0], 1: [1], 2: [2], 4: [3]}
                    mapping[3] = [4, 5, 6]
                    mapping[5] = [7, 8, 9]
                    for old_channel, new_channels in mapping.items():
                        for new_channel in new_channels:
                            expanded[:, new_channel] = first_weight[:, old_channel]
                    migrated["conv.0.weight"] = expanded
                    state_dict = migrated
                model.load_state_dict(state_dict)
                model.eval()
                self.model = model
                self.model_loaded = True
            except Exception as exc:  # corrupted / incompatible checkpoint
                problem = f"failed to load checkpoint {self.model_path}: {exc}"

        if problem:
            if require_weights:
                raise RuntimeError(f"DQNAgent unavailable: {problem}")
            logger.warning("DQNAgent running UNTRAINED random fallback: %s", problem)
            self.name = f"{name} [UNTRAINED]"

    def reset(self):
        """Reset temporal state tracking for a new game episode."""
        self.prev_pacman = None
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

        state_arr = encode_state(
            pacman_pos=pacman_pos,
            ghost_positions=ghost_positions,
            pellets=pellets,
            prev_pacman_pos=self.prev_pacman,
            prev_ghost_positions=self.prev_ghosts,
            last_move=last_move,
            ghost_dirs=ghost_dirs,
            mode_step=effective_mode_step,
            steps_without_pellet=effective_stall_steps,
            steps_remaining=effective_remaining,
            horizon=horizon,
        )

        # Update previous positions for next step
        self.prev_pacman = tuple(pacman_pos)
        self.prev_ghosts = [tuple(g) for g in ghost_positions]
        self._mode_step = effective_mode_step + 1
        self._decision_count += 1

        with torch.no_grad():
            s_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
            raw_q = self.model(s_tensor).squeeze(0)

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

        dist = softmax(legal_q, temp=15.0)
        return DecisionResult(
            choice=choice,
            probabilities=dist,
            confidence=margin_confidence(dist),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            is_live=False,
        )
