"""
Training pipeline for Deep Q-Network (DQN) and Double-DQN on Pac-Man.
Implements:
- History-free observation: identity-preserving grid planes + stall/horizon scalars
- Full-resolution convolution stack (no pooling) with dueling heads (see dqn.md)
- Prioritized Experience Replay with importance-sampling correction
- Target Network (Double DQN to prevent Q-value overestimation), legal-action masks
- Smooth L1 (Huber) Loss and Adam optimization
- Epsilon-greedy exploration decayed over a fraction of the planned episodes
- Final re-evaluation of the best and final weights on VAL seeds not used for selection
- Reward = tournament score delta x REWARD_SCALE, plus small reversal and per-step
  stall penalties; stalls are penalized, not truncated (same episodes as evaluation)
- Fully seeded runs with disjoint TRAIN / VAL seed ranges (core.seeds)
- Model checkpointing to rl/weights/dqn_pacman.pt (selected on the *pure greedy* policy)
- Resumable training: a full training-state checkpoint (networks, optimizer, LR schedule,
  replay buffer, exploration, counters, metrics, RNG streams) is written every
  `checkpoint_every` episodes, on --stop-after and on Ctrl+C; continue with --resume.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.environment import DEFAULT_MAX_STEPS, SCORE_DEATH, SCORE_PELLET, SCORE_WIN, Environment
from core.maze_data import OPPOSITE_DIRECTIONS
from core.seeds import seed_everything, train_seed, val_seeds
from rl.dqn_model import (
    ACTION_TO_IDX,
    GHOST_SLOTS,
    MODE_CYCLE,
    N_PELLET_BYTES,
    NUM_CHANNELS,
    NUM_SCALARS,
    SCATTER_STEPS,
    STALL_STEPS,
    CompactObservation,
    PacmanDQN,
    decode_grids_batch,
    encode_state,
    pack_pellets,
)

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "dqn_pacman.pt")
TRAINING_STATE_NAME = "dqn_training_state.pt"
TRAINING_STATE_VERSION = 8  # v8: compact replay buffer (41B/trans), SumTree PER O(log N), 200k default capacity


def default_num_threads() -> int:
    """Intra-op torch threads for training. All logical CPUs oversubscribes this small
    network (measured: 16 threads was 2.5-4x slower per update than 12 on a 16-CPU
    machine), so default to three quarters of them."""
    return max(1, (os.cpu_count() or 1) * 3 // 4)


# Reward: the evaluation score (SCORE_* in core.environment) scaled to a small range,
# so the agent optimizes exactly what the tournament measures, plus two small shaping
# terms (in the same scaled units, where a pellet is worth SCORE_PELLET * REWARD_SCALE).
REWARD_SCALE = 0.1
R_REVERSAL = -0.1        # reversing in a corridor while other moves exist
R_STALL_PER_STEP = -0.05  # every step once STALL_STEPS steps passed without a pellet


def compute_reward(
    prev_move: Optional[str],
    move: str,
    ate: bool,
    collided: bool,
    won: bool,
    n_legal: int,
    steps_without_pellet: int,
    pellets_remaining: int = 150,
) -> Tuple[float, bool]:
    """
    Reward for one transition: scaled score delta plus shaping.

    `prev_move` MUST be Pac-Man's heading *before* this step (read it before env.step()),
    and `steps_without_pellet` the counter *after* it. Returns (reward, stalled), where
    `stalled` means the stall penalty applied. Stalling never ends the episode: the
    trainer ends episodes exactly where evaluation does (collision, board cleared, time
    limit), and the stall plane in the observation keeps the penalty Markov.
    """
    score_delta = (
        (SCORE_PELLET if ate else 0)
        + (SCORE_DEATH if collided else 0)
        + (SCORE_WIN if won and not collided else 0)
    )
    r = REWARD_SCALE * score_delta
    if prev_move and n_legal > 1 and move == OPPOSITE_DIRECTIONS.get(prev_move):
        r += R_REVERSAL
    stalled = steps_without_pellet >= STALL_STEPS and not collided and not won
    if stalled:
        r += R_STALL_PER_STEP
    return r, stalled


def observe(
    env: Environment,
    steps_without_pellet: int,
    steps_remaining: int,
    horizon: int,
    egocentric: bool = False,
) -> CompactObservation:
    """(grid, scalars) observation of the environment's current state with compact metadata."""
    px, py = env.pacman_pos
    gx = tuple(int(g[0]) for g in env.ghost_positions[:GHOST_SLOTS])
    gy = tuple(int(g[1]) for g in env.ghost_positions[:GHOST_SLOTS])
    p_head = ACTION_TO_IDX.get(env.last_move, -1)
    g_head = tuple(ACTION_TO_IDX.get(d, -1) for d in (env.ghost_dirs or ["up"] * GHOST_SLOTS)[:GHOST_SLOTS])
    mode_step = int(env.mode_step)
    pellet_bits = pack_pellets(env.pellets)

    grid, scalars = encode_state(
        tuple(env.pacman_pos), [tuple(g) for g in env.ghost_positions], env.pellets,
        last_move=env.last_move, ghost_dirs=env.ghost_dirs, mode_step=env.mode_step,
        steps_without_pellet=steps_without_pellet, steps_remaining=steps_remaining, horizon=horizon,
        egocentric=egocentric,
    )
    compact = (px, py, gx, gy, p_head, g_head, mode_step, pellet_bits, scalars)
    return CompactObservation(grid, scalars, compact)


def greedy_action(policy_net, obs, legal) -> str:
    """Pure greedy argmax over legal actions (ties broken by the global RNG)."""
    grid, scalars = obs
    device = next(policy_net.parameters()).device
    with torch.no_grad():
        g = torch.from_numpy(grid).unsqueeze(0).to(device)
        s = torch.from_numpy(scalars).unsqueeze(0).to(device)
        q = policy_net(g, s).squeeze(0)
    legal_q = {m: q[ACTION_TO_IDX[m]].item() for m in legal}
    best = max(legal_q.values())
    return random.choice([m for m in legal if legal_q[m] == best])


def masked_next_actions(q_values: torch.Tensor, legal_action_masks: torch.Tensor) -> torch.Tensor:
    """Select the highest-valued legal action for each next state."""
    masked_q = q_values.masked_fill(~legal_action_masks, torch.finfo(q_values.dtype).min)
    return masked_q.argmax(dim=1, keepdim=True)


class SumTree:
    """Complete binary sum and max tree for Prioritized Experience Replay.

    Maintains priorities in a binary tree of size 2 * tree_capacity (power-of-2):
    - Leaves are stored at indices [tree_capacity, tree_capacity + capacity - 1].
    - Internal node k has children 2k and 2k + 1, storing tree_sum[k] = left + right
      and tree_max[k] = max(left, right).
    - O(log N) point priority updates and prefix-sum inverse-CDF sampling.
    - O(1) live max priority tracking via root tree_max[1].
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree_capacity = 1 << (capacity - 1).bit_length() if capacity > 1 else 1
        self.tree_sum = np.zeros(2 * self.tree_capacity, dtype=np.float64)
        self.tree_max = np.zeros(2 * self.tree_capacity, dtype=np.float32)

    @property
    def total_sum(self) -> float:
        return float(self.tree_sum[1])

    @property
    def max_value(self) -> float:
        return float(self.tree_max[1])

    def update(self, idx: int, priority: float, alpha: float = 0.6) -> None:
        """Update leaf idx with priority and propagate up to root."""
        k = self.tree_capacity + idx
        val = float(priority) ** alpha
        diff = val - self.tree_sum[k]
        self.tree_sum[k] = val
        self.tree_max[k] = priority
        k //= 2
        while k >= 1:
            self.tree_sum[k] += diff
            self.tree_max[k] = max(self.tree_max[2 * k], self.tree_max[2 * k + 1])
            k //= 2

    def sample(self, uniform_draws: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized batch sampling of indices given uniform draws in [0, total_sum).

        Returns (indices, priorities**alpha).
        """
        v = np.array(uniform_draws, dtype=np.float64, copy=True)
        k = np.ones(len(v), dtype=np.int64)
        depth = (self.tree_capacity - 1).bit_length()
        for _ in range(depth):
            left = 2 * k
            left_sum = self.tree_sum[left]
            go_right = v > left_sum
            v = np.where(go_right, v - left_sum, v)
            k = left + go_right

        indices = np.clip(k - self.tree_capacity, 0, max(0, size - 1))
        scaled_priorities = self.tree_sum[self.tree_capacity + indices]
        return indices, scaled_priorities

    def rebuild(self, priorities: np.ndarray, alpha: float = 0.6) -> None:
        """Rebuild entire tree from priorities array."""
        self.tree_sum.fill(0.0)
        self.tree_max.fill(0.0)
        n = min(len(priorities), self.capacity)
        vals = np.asarray(priorities[:n], dtype=np.float32)
        leaf_start = self.tree_capacity
        self.tree_sum[leaf_start : leaf_start + n] = (vals ** alpha).astype(np.float64)
        self.tree_max[leaf_start : leaf_start + n] = vals
        for k in range(self.tree_capacity - 1, 0, -1):
            self.tree_sum[k] = self.tree_sum[2 * k] + self.tree_sum[2 * k + 1]
            self.tree_max[k] = max(self.tree_max[2 * k], self.tree_max[2 * k + 1])


class Priorities(np.ndarray):
    """Numpy ndarray subclass that notifies ReplayBuffer when priorities are updated."""
    def __new__(cls, input_array, on_set=None):
        obj = np.asarray(input_array).view(cls)
        obj.on_set = on_set
        return obj

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if hasattr(self, "on_set") and self.on_set is not None:
            self.on_set(index, value)


class ReplayBuffer:
    """Proportional prioritized experience replay with compact storage and SumTree.

    Transitions are stored in raw compact state representation (41 bytes per state),
    enabling 200,000+ transitions in under ~25 MB of RAM (a ~360x reduction from full fp16 grids).
    Batch decoding into full spatial tensors is vectorized via decode_grids_batch() in ~1 ms.
    Priority sampling and point updates are O(log N) via a power-of-2 SumTree, with O(1)
    live max-priority tracking. Non-compact observations are seamlessly supported via a fallback path.
    """
    _COMPACT_ARRAYS = (
        "px", "py", "gx", "gy", "p_head", "g_head", "mode_steps", "pellet_bits", "scalars",
        "next_px", "next_py", "next_gx", "next_gy", "next_p_head", "next_g_head",
        "next_mode_steps", "next_pellet_bits", "next_scalars",
    )
    _OBS_ARRAYS = ("grids", "scalars", "next_grids", "next_scalars")
    _TRANSITION_ARRAYS = ("priorities", "actions", "rewards", "dones", "next_action_masks")

    def __init__(
        self,
        capacity: int = 200_000,
        rng: Optional[random.Random] = None,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-5,
        egocentric: bool = False,
    ):
        self.capacity = capacity
        self.rng = rng or random.Random()
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.egocentric = egocentric
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

        self.tree = SumTree(capacity)
        self.priorities = Priorities(np.zeros(capacity, dtype=np.float32), on_set=self._on_priority_set)

        self.is_compact = True
        self.px = np.zeros(capacity, dtype=np.uint8)
        self.py = np.zeros(capacity, dtype=np.uint8)
        self.gx = np.zeros((capacity, GHOST_SLOTS), dtype=np.uint8)
        self.gy = np.zeros((capacity, GHOST_SLOTS), dtype=np.uint8)
        self.p_head = np.full(capacity, -1, dtype=np.int8)
        self.g_head = np.full((capacity, GHOST_SLOTS), -1, dtype=np.int8)
        self.mode_steps = np.zeros(capacity, dtype=np.uint16)
        self.pellet_bits = np.zeros((capacity, N_PELLET_BYTES), dtype=np.uint8)
        self.scalars = np.zeros((capacity, NUM_SCALARS), dtype=np.float32)

        self.next_px = np.zeros(capacity, dtype=np.uint8)
        self.next_py = np.zeros(capacity, dtype=np.uint8)
        self.next_gx = np.zeros((capacity, GHOST_SLOTS), dtype=np.uint8)
        self.next_gy = np.zeros((capacity, GHOST_SLOTS), dtype=np.uint8)
        self.next_p_head = np.full(capacity, -1, dtype=np.int8)
        self.next_g_head = np.full((capacity, GHOST_SLOTS), -1, dtype=np.int8)
        self.next_mode_steps = np.zeros(capacity, dtype=np.uint16)
        self.next_pellet_bits = np.zeros((capacity, N_PELLET_BYTES), dtype=np.uint8)
        self.next_scalars = np.zeros((capacity, NUM_SCALARS), dtype=np.float32)

        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_action_masks = np.ones((capacity, len(ACTION_TO_IDX)), dtype=np.bool_)

        self.grids = self.next_grids = None

    def _on_priority_set(self, index, value):
        if isinstance(index, (int, np.integer)):
            self.tree.update(int(index), float(value), alpha=self.alpha)
            self.max_priority = max(float(self.tree.max_value), self.priority_epsilon)
        else:
            self.tree.rebuild(self.priorities[:self.size], alpha=self.alpha)
            self.max_priority = max(float(self.tree.max_value), self.priority_epsilon)

    def _allocate(self, grid_shape, num_scalars: int):
        self.grids = np.zeros((self.capacity, *grid_shape), dtype=np.float16)
        self.next_grids = np.zeros((self.capacity, *grid_shape), dtype=np.float16)
        if self.scalars.shape[1] != num_scalars:
            self.scalars = np.zeros((self.capacity, num_scalars), dtype=np.float32)
            self.next_scalars = np.zeros((self.capacity, num_scalars), dtype=np.float32)

    def push(self, obs, action_idx, reward, next_obs, done, next_action_mask=None):
        i = self.position
        if hasattr(obs, "compact") and hasattr(next_obs, "compact"):
            c, nc = obs.compact, next_obs.compact
            self.px[i], self.py[i] = c[0], c[1]
            self.gx[i] = c[2]
            self.gy[i] = c[3]
            self.p_head[i] = c[4]
            self.g_head[i] = c[5]
            self.mode_steps[i] = c[6]
            self.pellet_bits[i] = c[7]
            self.scalars[i] = c[8]

            self.next_px[i], self.next_py[i] = nc[0], nc[1]
            self.next_gx[i] = nc[2]
            self.next_gy[i] = nc[3]
            self.next_p_head[i] = nc[4]
            self.next_g_head[i] = nc[5]
            self.next_mode_steps[i] = nc[6]
            self.next_pellet_bits[i] = nc[7]
            self.next_scalars[i] = nc[8]
        else:
            self.is_compact = False
            grid, scalars = obs
            next_grid, next_scalars = next_obs
            if self.grids is None:
                self._allocate(np.shape(grid), len(scalars))
            self.grids[i] = grid
            self.scalars[i] = scalars
            self.next_grids[i] = next_grid
            self.next_scalars[i] = next_scalars

        self.actions[i] = action_idx
        self.rewards[i] = reward
        self.dones[i] = done
        self.next_action_masks[i] = True if next_action_mask is None else next_action_mask
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.priorities[i] = self.max_priority

    def sample(self, batch_size: int, beta: float = 0.4, stratified: bool = True) -> dict:
        total = self.tree.total_sum
        if stratified and batch_size > 1:
            seg = total / batch_size
            draws = np.array([(k + self.rng.random()) * seg for k in range(batch_size)], dtype=np.float64)
        else:
            draws = np.array([self.rng.random() for _ in range(batch_size)], dtype=np.float64) * total
        indices, scaled = self.tree.sample(draws, self.size)
        probs = scaled / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        if self.is_compact:
            grids = decode_grids_batch(
                self.px[indices], self.py[indices], self.gx[indices], self.gy[indices],
                self.p_head[indices], self.g_head[indices], self.mode_steps[indices],
                self.pellet_bits[indices], egocentric=self.egocentric,
            )
            next_grids = decode_grids_batch(
                self.next_px[indices], self.next_py[indices], self.next_gx[indices], self.next_gy[indices],
                self.next_p_head[indices], self.next_g_head[indices], self.next_mode_steps[indices],
                self.next_pellet_bits[indices], egocentric=self.egocentric,
            )
        else:
            grids = torch.from_numpy(self.grids[indices].astype(np.float32))
            next_grids = torch.from_numpy(self.next_grids[indices].astype(np.float32))

        return {
            "grids": grids,
            "scalars": torch.from_numpy(self.scalars[indices]),
            "actions": torch.from_numpy(self.actions[indices]).unsqueeze(1),
            "rewards": torch.from_numpy(self.rewards[indices]).unsqueeze(1),
            "next_grids": next_grids,
            "next_scalars": torch.from_numpy(self.next_scalars[indices]),
            "dones": torch.from_numpy(self.dones[indices]).unsqueeze(1),
            "next_action_masks": torch.from_numpy(self.next_action_masks[indices]),
            "indices": indices,
            "weights": torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
        }

    def update_priorities(self, indices, td_errors):
        raw_priorities = np.asarray(self.priorities)
        for index, error in zip(indices, td_errors):
            idx = int(index)
            priority = abs(float(error)) + self.priority_epsilon
            raw_priorities[idx] = priority
            self.tree.update(idx, priority, alpha=self.alpha)
        self.max_priority = max(float(self.tree.max_value), self.priority_epsilon)

    def state_dict(self) -> dict:
        """Tensor snapshot of the filled part of the buffer (shares memory, no copy)."""
        n = self.size
        state = {
            "capacity": self.capacity, "alpha": self.alpha, "priority_epsilon": self.priority_epsilon,
            "position": self.position, "size": n, "max_priority": self.max_priority,
            "is_compact": self.is_compact, "egocentric": self.egocentric,
        }
        for name in self._TRANSITION_ARRAYS:
            state[name] = torch.from_numpy(np.asarray(getattr(self, name)[:n]))
        if self.is_compact:
            for name in self._COMPACT_ARRAYS:
                state[name] = torch.from_numpy(getattr(self, name)[:n])
        elif self.grids is not None:
            for name in self._OBS_ARRAYS:
                state[name] = torch.from_numpy(getattr(self, name)[:n])
        return state

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError(f"replay capacity mismatch: checkpoint {state['capacity']} vs buffer {self.capacity}")
        n = state["size"]
        self.alpha = state["alpha"]
        self.priority_epsilon = state["priority_epsilon"]
        self.egocentric = state.get("egocentric", False)
        self.position = state["position"]
        self.size = n
        for name in self._TRANSITION_ARRAYS:
            arr = np.asarray(getattr(self, name))
            arr[:n] = state[name].numpy()
            if name == "priorities":
                arr[n:] = 0.0
        self.tree.rebuild(np.asarray(self.priorities)[:n], alpha=self.alpha)
        self.max_priority = state["max_priority"]
        self.is_compact = state.get("is_compact", False)
        if self.is_compact:
            for name in self._COMPACT_ARRAYS:
                getattr(self, name)[:n] = state[name].numpy()
        elif "grids" in state:
            self._allocate(tuple(state["grids"].shape[1:]), state["scalars"].shape[1])
            for name in self._OBS_ARRAYS:
                getattr(self, name)[:n] = state[name].numpy()

    def __len__(self):
        return self.size


def _atomic_torch_save(obj, path: str) -> None:
    """Write via a temp file + rename so an interrupted save never corrupts `path`."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _rng_snapshot(rng: random.Random) -> dict:
    """All RNG streams the trainer consumes, as weights_only-loadable primitives."""
    np_name, np_keys, np_pos, np_has_gauss, np_gauss = np.random.get_state()
    snap = {
        "trainer": rng.getstate(),
        "python": random.getstate(),
        "numpy": (np_name, np_keys.tolist(), np_pos, np_has_gauss, np_gauss),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state()
    return snap


def _rng_restore(rng: random.Random, snap: dict) -> None:
    rng.setstate(snap["trainer"])
    random.setstate(snap["python"])
    np_name, np_keys, np_pos, np_has_gauss, np_gauss = snap["numpy"]
    np.random.set_state((np_name, np.array(np_keys, dtype=np.uint32), np_pos, np_has_gauss, np_gauss))
    torch.set_rng_state(snap["torch"])
    if "torch_cuda" in snap and torch.cuda.is_available():
        torch.cuda.set_rng_state(snap["torch_cuda"])


def default_training_state_path(save_path: str = DEFAULT_MODEL_PATH) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(save_path)), TRAINING_STATE_NAME)


def epsilon_schedule(steps_done: int, total_steps: int, start: float, minimum: float, decay_fraction: float) -> float:
    """Exploration rate at environment step `steps_done`: exponential decay from `start`
    that reaches `minimum` after `decay_fraction` of the step budget. Step-based (not
    episode-based) because episode length grows a lot as the agent improves; a pure
    function of the step count, so resuming stays exact."""
    decay_steps = max(1.0, decay_fraction * total_steps)
    progress = min(1.0, steps_done / decay_steps)
    return max(minimum, start * (minimum / start) ** progress)


def beta_schedule(steps_done: int, total_steps: int, start: float) -> float:
    """PER importance-sampling exponent, annealed linearly from `start` to 1 over the budget."""
    return min(1.0, start + (1.0 - start) * steps_done / max(1, total_steps))


def cosine_lr(steps_done: int, total_steps: int, lr: float, final_ratio: float = 0.1) -> float:
    """Cosine learning-rate annealing from `lr` to `lr * final_ratio` over the step budget."""
    progress = min(1.0, steps_done / max(1, total_steps))
    return lr * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float) -> None:
    """Polyak averaging: target <- (1 - tau) * target + tau * policy."""
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.lerp_(param, tau)


class NStepBuffer:
    """Accumulates n-step discounted returns: R = sum_{i=0}^{m-1} gamma^i * r_i.

    Maintains a rolling deque of transitions within an episode. When len == n, yields
    the oldest transition discounted over n steps to next_obs. When an episode ends,
    flush() drains the remaining m < n transitions, which are all terminal (d=1.0).
    """
    def __init__(self, n: int, gamma: float):
        if n < 1:
            raise ValueError(f"n_step must be >= 1, got {n}")
        self.n = n
        self.gamma = gamma
        self.buffer = collections.deque()

    def add(self, obs, action_idx, reward, next_obs, done, next_mask):
        self.buffer.append((obs, action_idx, reward, next_obs, done, next_mask))
        if len(self.buffer) == self.n:
            return self._pop_n_step_transition()
        return None

    def _pop_n_step_transition(self):
        obs, action_idx = self.buffer[0][0], self.buffer[0][1]
        reward = 0.0
        for i, transition in enumerate(self.buffer):
            reward += (self.gamma ** i) * transition[2]
        last = self.buffer[-1]
        self.buffer.popleft()
        return obs, action_idx, reward, last[3], float(last[4]), last[5]

    def flush(self):
        transitions = []
        while self.buffer:
            obs, action_idx = self.buffer[0][0], self.buffer[0][1]
            reward = 0.0
            for i, transition in enumerate(self.buffer):
                reward += (self.gamma ** i) * transition[2]
            last = self.buffer[-1]
            transitions.append((obs, action_idx, reward, last[3], float(last[4]), last[5]))
            self.buffer.popleft()
        return transitions

    def clear(self):
        self.buffer.clear()


def evaluate_dqn(policy_net: nn.Module, episodes: int = 15, max_steps: int = DEFAULT_MAX_STEPS) -> Tuple[float, float]:
    """Mean score and pellets of the pure greedy policy (no inference heuristics) on the
    first `episodes` VAL seeds (the checkpoint-selection seeds)."""
    scores, pellets = evaluate_dqn_episodes(policy_net, val_seeds(episodes), max_steps)
    return statistics.mean(scores), statistics.mean(pellets)


def evaluate_dqn_episodes(policy_net: nn.Module, seeds, max_steps: int = DEFAULT_MAX_STEPS) -> Tuple[list, list]:
    """Per-episode scores and pellets of the pure greedy policy on `seeds`."""
    policy_net.eval()
    egocentric = getattr(policy_net, "egocentric", False)
    scores, pellets_cleared = [], []

    for seed in seeds:
        seed_everything(seed)
        env = Environment(seed=seed)
        score = pellets_count = 0

        for step_index in range(max_steps):
            legal = env.get_legal_moves(*env.pacman_pos)
            obs = observe(env, env.steps_without_pellet, max_steps - step_index, max_steps, egocentric=egocentric)
            move = greedy_action(policy_net, obs, legal)

            collided, ate_pellet, won = env.step(move)
            if ate_pellet:
                score += SCORE_PELLET
                pellets_count += 1
            if collided:
                score += SCORE_DEATH
                break
            if won:
                score += SCORE_WIN
                break

        scores.append(score)
        pellets_cleared.append(pellets_count)

    policy_net.train()
    return scores, pellets_cleared


def _summarize(scores: list, pellets: list) -> dict:
    n = len(scores)
    ci95 = 1.96 * statistics.stdev(scores) / n ** 0.5 if n > 1 else 0.0
    return {
        "mean_score": round(statistics.mean(scores), 2), "score_ci95": round(ci95, 2),
        "mean_pellets": round(statistics.mean(pellets), 2),
    }


def train_dqn(
    total_steps: Optional[int] = None,
    episodes: Optional[int] = None,
    lr: float = 5e-4,
    batch_size: int = 64,
    gamma: float = 0.995,
    n_step: int = 1,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay_fraction: float = 0.6,
    per_beta_start: float = 0.4,
    target_update_steps: int = 400,
    target_tau: float = 0.0,
    train_every: int = 1,
    warmup_steps: int = 1000,
    max_steps: int = DEFAULT_MAX_STEPS,
    val_every_steps: Optional[int] = 25_000,
    val_every: Optional[int] = None,
    val_episodes: int = 20,
    seed: int = 0,
    save_path: str = DEFAULT_MODEL_PATH,
    make_plots: bool = True,
    buffer_capacity: int = 200_000,
    checkpoint_every: int = 100,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    stop_after: Optional[int] = None,
    final_val_episodes: int = 200,
    num_threads: Optional[int] = None,
    egocentric: bool = False,
    device: Optional[str] = None,
):
    """Train (or resume training) the DQN for a budget of `total_steps` environment steps
    (warm-up excluded; the last episode is cut at the budget). Alternatively, an episode budget
    `episodes` can be provided for backward compatibility.

    Resumable runs: a training-state checkpoint (`checkpoint_path`, default
    rl/weights/dqn_training_state.pt) is written every `checkpoint_every` episodes, after
    `stop_after` episodes of this session, and on Ctrl+C. With `resume=True` the run
    continues from it using the *checkpointed* hyperparameters (the arguments above are
    ignored except `checkpoint_path`, `stop_after`, `checkpoint_every`, `make_plots`,
    `final_val_episodes` and `num_threads`).

    One gradient update (on `batch_size` replayed transitions) runs every `train_every`
    environment steps, so the replay ratio is batch_size / train_every samples per step.
    `num_threads` sets torch's intra-op threads (default: default_num_threads()); it is a
    runtime setting, not a hyperparameter.

    Target network: Polyak-averaged after every gradient update when `target_tau` > 0,
    otherwise hard-copied every `target_update_steps` environment steps.

    Multi-step returns: transitions accumulate discounted returns over `n_step` environment
    steps (default 1; R = sum_{i=0}^{n-1} gamma^i * r_i) and bootstrap from s_{t+n} with discount gamma^n.
    At episode termination (collision, win, horizon cutoff), remaining transitions flush as
    terminal targets.

    All schedules are pure functions of the environment-step counter: exploration decays
    from `epsilon_start` to `epsilon_min` over `epsilon_decay_fraction` of the budget, the
    PER exponent anneals from `per_beta_start` to 1 and the learning rate follows a cosine
    from `lr` to lr / 10 over the budget. The greedy policy is validated on `val_episodes`
    VAL seeds whenever another `val_every_steps` steps have passed (checked at episode ends)
    and at the end of the run. When the run completes, the best checkpoint and the final
    weights are re-evaluated on `final_val_episodes` VAL seeds that were *not* used for
    checkpoint selection (selection picks the best of many noisy validations, so its own
    score is optimistic); results go to the log, the return value and dqn_final_eval.json.
    Resuming from a periodic or --stop-after checkpoint reproduces the uninterrupted
    run exactly; a Ctrl+C checkpoint also keeps the interrupted episode's partial
    experience and updates. The state file is deleted once the run completes.
    """
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required to train DQN.")
        return None

    checkpoint_path = checkpoint_path or default_training_state_path(save_path)
    ckpt = None
    if resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No training state to resume at {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if ckpt.get("version") != TRAINING_STATE_VERSION:
            raise ValueError(f"Unsupported training state version {ckpt.get('version')!r} in {checkpoint_path}")
        cfg = ckpt["config"]
        total_steps = cfg.get("total_steps")
        episodes = cfg.get("episodes")
        lr, batch_size, gamma = cfg["lr"], cfg["batch_size"], cfg["gamma"]
        n_step = cfg.get("n_step", 1)
        epsilon_start, epsilon_min, epsilon_decay_fraction = cfg["epsilon_start"], cfg["epsilon_min"], cfg["epsilon_decay_fraction"]
        per_beta_start, target_update_steps, train_every = cfg["per_beta_start"], cfg["target_update_steps"], cfg["train_every"]
        target_tau = cfg.get("target_tau", 0.0)
        warmup_steps, max_steps = cfg["warmup_steps"], cfg["max_steps"]
        val_every_steps = cfg.get("val_every_steps")
        val_every = cfg.get("val_every")
        val_episodes = cfg["val_episodes"]
        seed, save_path, buffer_capacity = cfg["seed"], cfg["save_path"], cfg["buffer_capacity"]
        egocentric = cfg.get("egocentric", False)

    if total_steps is None and episodes is None:
        total_steps = 1_000_000
    if total_steps is not None and total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if episodes is not None and episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {n_step}")
    if train_every < 1:
        raise ValueError(f"train_every must be >= 1, got {train_every}")
    if val_every_steps is not None and val_every_steps < 1:
        raise ValueError(f"val_every_steps must be >= 1, got {val_every_steps}")
    if val_every is not None and val_every < 1:
        raise ValueError(f"val_every must be >= 1, got {val_every}")
    if not 0.0 <= target_tau <= 1.0:
        raise ValueError(f"target_tau must be in [0, 1], got {target_tau}")

    budget_steps = total_steps if total_steps is not None else (episodes * max_steps)
    gamma_n = gamma ** n_step

    config = {
        "total_steps": total_steps, "episodes": episodes, "lr": lr, "batch_size": batch_size,
        "gamma": gamma, "n_step": n_step,
        "epsilon_start": epsilon_start, "epsilon_min": epsilon_min, "epsilon_decay_fraction": epsilon_decay_fraction,
        "per_beta_start": per_beta_start, "target_update_steps": target_update_steps, "target_tau": target_tau,
        "train_every": train_every, "warmup_steps": warmup_steps, "max_steps": max_steps,
        "val_every_steps": val_every_steps, "val_every": val_every, "val_episodes": val_episodes,
        "seed": seed, "save_path": save_path, "buffer_capacity": buffer_capacity,
        "egocentric": egocentric,
    }

    num_threads = num_threads or default_num_threads()
    torch.set_num_threads(num_threads)

    seed_everything(seed)
    rng = random.Random(seed)

    weights_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(weights_dir, "dqn_training.log")
    json_path = os.path.join(weights_dir, "dqn_training_metrics.json")
    csv_path = os.path.join(weights_dir, "dqn_training_metrics.csv")
    final_eval_path = os.path.join(weights_dir, "dqn_final_eval.json")

    if device is None or device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    policy_net = PacmanDQN(egocentric=egocentric).to(device_obj)
    target_net = PacmanDQN(egocentric=egocentric).to(device_obj)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, rng=rng, egocentric=egocentric)

    metrics_history = []
    best_val_score = -float("inf")
    best_val_episode = best_val_steps = None
    epsilon = epsilon_start
    steps_done = 0       # environment steps after warm-up; drives every schedule
    validations_done = 0  # number of val_every_steps boundaries already validated
    start_episode = 0
    elapsed_before = 0.0
    recent_scores = collections.deque(maxlen=50)

    if ckpt is not None:
        policy_net.load_state_dict(ckpt["policy_net"])
        target_net.load_state_dict(ckpt["target_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_obj)
        replay_buffer.load_state_dict(ckpt["replay_buffer"])
        prog = ckpt["progress"]
        start_episode, steps_done, epsilon = prog["episodes_done"], prog["steps_done"], prog["epsilon"]
        validations_done = prog["validations_done"]
        best_val_score, best_val_episode, best_val_steps = prog["best_val_score"], prog["best_val_episode"], prog["best_val_steps"]
        elapsed_before = prog["elapsed_sec"]
        recent_scores.extend(prog["recent_scores"])
        metrics_history = list(ckpt["metrics_history"])
        _rng_restore(rng, ckpt["rng"])
        ckpt = None  # release the (large) loaded buffer tensors

    def export_metrics():
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(metrics_history, f, indent=2)
        if metrics_history:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()), lineterminator="\n")
                writer.writeheader()
                writer.writerows(metrics_history)

    def save_training_state(episodes_done: int, elapsed_sec: float):
        _atomic_torch_save({
            "version": TRAINING_STATE_VERSION,
            "config": config,
            "policy_net": {k: v.cpu() for k, v in policy_net.state_dict().items()},
            "target_net": {k: v.cpu() for k, v in target_net.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
            "progress": {
                "episodes_done": episodes_done, "steps_done": steps_done, "epsilon": epsilon,
                "validations_done": validations_done,
                "best_val_score": best_val_score, "best_val_episode": best_val_episode, "best_val_steps": best_val_steps,
                "elapsed_sec": elapsed_sec, "recent_scores": list(recent_scores),
            },
            "metrics_history": metrics_history,
            "rng": _rng_snapshot(rng),
        }, checkpoint_path)
        export_metrics()

    with open(log_path, "a" if start_episode else "w", encoding="utf-8", newline="\n") as log_file:

        def log_print(msg: str = ""):
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        budget_str = f"{budget_steps:,} env steps" if total_steps is not None else f"{episodes} episodes ({budget_steps:,} env steps)"
        val_str = f"every {val_every} episodes" if val_every is not None else f"every {val_every_steps:,} steps"
        framing_str = "Egocentric (Option 1: Pac-Man centered)" if egocentric else "Allocentric (Option 2: 25x25 dilated + circular X tunnel)"

        if start_episode:
            log_print(f"\n[RESUME] Continuing from step {steps_done:,}/{budget_steps:,} (episode {start_episode}) "
                      f"({os.path.relpath(checkpoint_path)}, {len(replay_buffer):,} replay transitions)\n")
        else:
            target_mode = f"soft (tau={target_tau})" if target_tau > 0 else f"hard every {target_update_steps} steps"
            log_print("======================================================================")
            log_print(f" DEEP Q-NETWORK (DQN) TRAINING: {NUM_CHANNELS} grid planes + {NUM_SCALARS} scalars")
            log_print("======================================================================")
            log_print(f"Device: {device_obj} ({num_threads} threads) | Framing: {framing_str}")
            log_print(f"Batch Size: {batch_size} | Learning Rate: {lr} | Gamma: {gamma} (n-step: {n_step}) | Seed: {seed}")
            log_print(f"Budget: {budget_str} | Horizon: {max_steps} | Target: {target_mode} | "
                      f"Update: every {train_every} step(s) (replay ratio {batch_size / train_every:g}) | Buffer: {buffer_capacity:,}")
            log_print(f"Validation: {val_episodes} VAL seeds {val_str} (pure greedy policy)")
            log_print(f"Training state: {os.path.relpath(checkpoint_path)} every {checkpoint_every} episodes\n")
            log_print(f"Warming up replay buffer with {warmup_steps:,} random-policy steps...")

            # ---- Warm-up phase (uniform random policy) ----
            env = Environment(seed=train_seed(rng))
            n_step_warmup = NStepBuffer(n_step, gamma)
            while len(replay_buffer) < warmup_steps:
                legal = env.get_legal_moves(*env.pacman_pos)
                s = observe(env, env.steps_without_pellet, max(0, max_steps - env.step_count), max_steps, egocentric=egocentric)

                m = rng.choice(legal)
                prev_move = env.last_move
                col, ate, won = env.step(m)
                r, _ = compute_reward(prev_move, m, ate, col, won, len(legal), env.steps_without_pellet)

                ns = observe(env, env.steps_without_pellet, max(0, max_steps - env.step_count), max_steps, egocentric=egocentric)
                next_legal = env.get_legal_moves(*env.pacman_pos)
                next_mask = np.array([a in next_legal for a in ACTION_TO_IDX], dtype=np.bool_)
                if not next_mask.any():
                    next_mask[:] = True
                done = col or won or env.step_count >= max_steps
                tr = n_step_warmup.add(s, ACTION_TO_IDX[m], r, ns, float(done), next_mask)
                if tr is not None:
                    replay_buffer.push(*tr)

                if done:
                    for tr in n_step_warmup.flush():
                        replay_buffer.push(*tr)
                    env = Environment(seed=train_seed(rng))

            for tr in n_step_warmup.flush():
                replay_buffer.push(*tr)

            log_print("Warm-up complete! Beginning DQN neural optimization...\n")

        t0 = time.time() - elapsed_before
        max_episodes = episodes if episodes is not None else float("inf")
        session_end = max_episodes if stop_after is None else min(max_episodes, start_episode + max(0, stop_after))

        ep = start_episode
        try:
            while ep < session_end and steps_done < budget_steps:
                env = Environment(seed=train_seed(rng))
                n_step_buffer = NStepBuffer(n_step, gamma)
                ep_reward = 0.0
                pellets_eaten = 0
                collided = won = False
                ep_losses = []
                steps = 0

                for steps in range(1, max_steps + 1):
                    steps_done += 1
                    epsilon = epsilon_schedule(steps_done, budget_steps, epsilon_start, epsilon_min, epsilon_decay_fraction)
                    beta = beta_schedule(steps_done, budget_steps, per_beta_start)
                    current_lr = cosine_lr(steps_done, budget_steps, lr)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                    legal = env.get_legal_moves(*env.pacman_pos)
                    s = observe(env, env.steps_without_pellet, max_steps - steps + 1, max_steps, egocentric=egocentric)

                    # Epsilon-greedy action selection over the raw network
                    if rng.random() < epsilon:
                        m = rng.choice(legal)
                    else:
                        m = greedy_action(policy_net, s, legal)

                    prev_move = env.last_move  # heading BEFORE the step (fixes dead reversal penalty)
                    col, ate, won = env.step(m)
                    if ate:
                        pellets_eaten += 1
                    collided = col

                    r, _ = compute_reward(prev_move, m, ate, col, won, len(legal), env.steps_without_pellet)
                    ep_reward += r

                    ns = observe(env, env.steps_without_pellet, max_steps - steps, max_steps, egocentric=egocentric)
                    terminal = col or won or steps >= max_steps or steps_done >= budget_steps
                    next_legal = env.get_legal_moves(*env.pacman_pos)
                    next_mask = np.array([a in next_legal for a in ACTION_TO_IDX], dtype=np.bool_)
                    if not next_mask.any():
                        next_mask[:] = True
                    tr = n_step_buffer.add(s, ACTION_TO_IDX[m], r, ns, float(terminal), next_mask)
                    if tr is not None:
                        replay_buffer.push(*tr)

                    # Sample batch & optimize (every train_every steps of the global,
                    # checkpointed step counter, so resumed runs update at the same steps)
                    if len(replay_buffer) >= batch_size and steps_done % train_every == 0:
                        batch = replay_buffer.sample(batch_size, beta)
                        b_grids = batch["grids"].to(device_obj)
                        b_scalars = batch["scalars"].to(device_obj)
                        b_actions = batch["actions"].to(device_obj)
                        b_next_grids = batch["next_grids"].to(device_obj)
                        b_next_scalars = batch["next_scalars"].to(device_obj)
                        b_next_masks = batch["next_action_masks"].to(device_obj)
                        b_rewards = batch["rewards"].to(device_obj)
                        b_dones = batch["dones"].to(device_obj)
                        b_weights = batch["weights"].to(device_obj)

                        curr_q = policy_net(b_grids, b_scalars).gather(1, b_actions)

                        # Double DQN target computation
                        with torch.no_grad():
                            next_q_online = policy_net(b_next_grids, b_next_scalars)
                            best_next_actions = masked_next_actions(next_q_online, b_next_masks)
                            next_target_q = target_net(b_next_grids, b_next_scalars).gather(1, best_next_actions)
                            expected_q = b_rewards + gamma_n * next_target_q * (1.0 - b_dones)

                        td_errors = expected_q - curr_q
                        per_item_loss = nn.functional.smooth_l1_loss(curr_q, expected_q, reduction="none")
                        loss = (b_weights * per_item_loss).mean()
                        replay_buffer.update_priorities(batch["indices"], td_errors.detach().squeeze(1).cpu().numpy())
                        ep_losses.append(loss.item())

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                        optimizer.step()

                        if target_tau > 0:
                            soft_update(target_net, policy_net, target_tau)

                    # Target network hard synchronization (if not using soft updates)
                    if target_tau <= 0 and steps_done % target_update_steps == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                    if terminal:
                        for tr in n_step_buffer.flush():
                            replay_buffer.push(*tr)
                        break

                current_lr = optimizer.param_groups[0]["lr"]

                score = pellets_eaten * SCORE_PELLET + (SCORE_DEATH if collided else 0) + (SCORE_WIN if won else 0)
                recent_scores.append(score)
                avg_rec = statistics.mean(recent_scores)
                avg_loss = statistics.mean(ep_losses) if ep_losses else 0.0

                is_budget_done = (steps_done >= budget_steps) or (episodes is not None and ep + 1 >= episodes)
                trigger_val = False
                if val_every is not None:
                    trigger_val = ((ep + 1) % val_every == 0) or is_budget_done
                elif val_every_steps is not None:
                    trigger_val = (steps_done // val_every_steps > validations_done) or is_budget_done

                val_sc = val_pel = None
                if trigger_val:
                    if val_every_steps is not None:
                        validations_done = max(validations_done + 1, steps_done // val_every_steps)
                    else:
                        validations_done += 1
                    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    val_sc, val_pel = evaluate_dqn(policy_net, episodes=val_episodes, max_steps=max_steps)
                    random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(torch_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state)

                    mark = ""
                    if val_sc > best_val_score:
                        best_val_score = val_sc
                        best_val_episode = ep + 1
                        best_val_steps = steps_done
                        _atomic_torch_save({k: v.cpu() for k, v in policy_net.state_dict().items()}, save_path)
                        mark = " [* SAVED BEST]"

                    ep_str = f"Ep {ep+1:04d}" if episodes is None else f"Ep {ep+1:04d}/{episodes}"
                    log_print(
                        f"Step {steps_done:07d}/{budget_steps:07d} ({ep_str}) | Train Avg(50): {avg_rec:6.1f} | "
                        f"Val Score: {val_sc:6.1f} (Pellets: {val_pel:4.1f}) | "
                        f"eps: {epsilon:4.2f} | lr: {current_lr:.2e} | Time: {time.time() - t0:4.0f}s{mark}"
                    )

                metrics_history.append({
                    "episode": ep + 1,
                    "step": steps_done,
                    "score": score,
                    "train_avg_score": round(avg_rec, 1),
                    "pellets": pellets_eaten,
                    "reward": round(ep_reward, 1),
                    "steps": steps,
                    "collided": collided,
                    "loss": round(avg_loss, 4),
                    "epsilon": round(epsilon, 4),
                    "learning_rate": current_lr,
                    "val_score": round(val_sc, 1) if val_sc is not None else None,
                    "val_pellets": round(val_pel, 1) if val_pel is not None else None,
                })

                ep += 1
                done = ep
                session_stopped = ep >= session_end and not is_budget_done
                if not is_budget_done and (done % checkpoint_every == 0 or session_stopped):
                    save_training_state(done, time.time() - t0)
                    if session_stopped:
                        log_print(f"[CHECKPOINT] Stopped after episode {done} ({steps_done:,}/{budget_steps:,} steps); "
                                  f"continue with --resume ({os.path.relpath(checkpoint_path)})")

        except KeyboardInterrupt:
            # Count only fully completed episodes (also correct if Ctrl+C lands inside a
            # periodic save). The interrupted episode's partial experience and updates are
            # kept, and a resume replays that episode index.
            done = len(metrics_history)
            log_print(f"\n[INTERRUPTED] Saving training state after episode {done} ({steps_done:,}/{budget_steps:,} steps)...")
            save_training_state(done, time.time() - t0)
            log_print(f"[CHECKPOINT] Saved to {os.path.relpath(checkpoint_path)}; continue with --resume")
            return {"completed": False, "episodes_done": done, "steps_done": steps_done, "best_val_score": best_val_score,
                    "best_val_episode": best_val_episode, "metrics": metrics_history}

        completed = (steps_done >= budget_steps) or (episodes is not None and ep >= episodes)
        if not completed:
            return {"completed": False, "episodes_done": len(metrics_history), "steps_done": steps_done,
                    "best_val_score": best_val_score, "best_val_episode": best_val_episode, "metrics": metrics_history}

        final_eval = None
        if final_val_episodes > 0:
            # Unbiased re-check: seeds right after the selection seeds, never used to pick a checkpoint.
            seeds = val_seeds(final_val_episodes, offset=val_episodes)
            log_print(f"\nFinal re-evaluation on {len(seeds)} unseen VAL seeds ({seeds[0]}..{seeds[-1]})...")
            best_net = PacmanDQN(egocentric=egocentric).to(device_obj)
            best_net.load_state_dict(torch.load(save_path, map_location=device_obj, weights_only=True))
            final_eval = {
                "seeds": [seeds[0], seeds[-1]],
                "selection_seeds": [val_seeds(val_episodes)[0], val_seeds(val_episodes)[-1]],
                "best_checkpoint": {"episode": best_val_episode, "step": best_val_steps, "selection_score": best_val_score,
                                    **_summarize(*evaluate_dqn_episodes(best_net, seeds, max_steps))},
                "final_weights": {"episode": ep, "step": steps_done, **_summarize(*evaluate_dqn_episodes(policy_net, seeds, max_steps))},
            }
            with open(final_eval_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(final_eval, f, indent=2)
            for label, key in (("Best checkpoint", "best_checkpoint"), ("Final weights  ", "final_weights")):
                r = final_eval[key]
                step_str = f", step {r['step']:,}" if r.get("step") is not None else ""
                log_print(f"  {label} (ep {r['episode']}{step_str}): {r['mean_score']:7.1f} +/- {r['score_ci95']:5.1f} "
                          f"(pellets {r['mean_pellets']:5.1f})")

        log_print("\n======================================================================")
        # Relative path: the log is a committed artifact and must not embed local directories
        log_print(f" TRAINING COMPLETE! Best Checkpoint Saved to {os.path.relpath(save_path)}")
        step_str = f", step {best_val_steps:,}" if best_val_steps is not None else ""
        log_print(f" Best Validation Score: {best_val_score:.1f} (episode {best_val_episode}{step_str}; selection score, optimistic)")
        log_print("======================================================================")

    export_metrics()
    # The resumable state only serves unfinished runs; drop it so a stale state can't be resumed.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if make_plots:
        from rl.plot_metrics import plot_metrics  # lazy: pulls in pygame
        plot_metrics(json_path, weights_dir)

    return {"completed": True, "episodes_done": ep, "steps_done": steps_done, "best_val_score": best_val_score,
            "best_val_episode": best_val_episode, "metrics": metrics_history, "final_eval": final_eval}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN on Pac-Man",
        epilog="Resume an interrupted run with --resume; hyperparameters then come from the checkpoint.",
    )
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Total environment steps budget (default 1,000,000)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (optional alternative to --total-steps)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default 5e-4)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default 64)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor (default 0.99)")
    parser.add_argument("--n-step", type=int, default=None, help="Multi-step return horizon (default 1)")
    parser.add_argument("--train-every", type=int, default=None,
                        help="Gradient update every N environment steps (default 1)")
    parser.add_argument("--target-tau", type=float, default=None,
                        help="Polyak target soft-update rate (0.0 = hard sync every target-update-steps)")
    parser.add_argument("--target-update-steps", type=int, default=None,
                        help="Hard target sync interval in environment steps (default 400, when target-tau=0)")
    parser.add_argument("--buffer-capacity", type=int, default=None,
                        help="Replay buffer capacity (default 200,000)")
    parser.add_argument("--val-every-steps", type=int, default=None,
                        help="Validate greedy policy every N environment steps (default 25,000)")
    parser.add_argument("--val-every", type=int, default=None,
                        help="Validate greedy policy every N episodes (optional alternative to --val-every-steps)")
    parser.add_argument("--seed", type=int, default=None, help="Global RNG seed (default 0)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save best model weights")
    parser.add_argument("--egocentric", action="store_true", default=None,
                        help="Use Option 1 (Pac-Man-centered egocentric view) instead of default Option 2 (fixed allocentric view)")
    parser.add_argument("--resume", action="store_true", help="Continue from the training-state checkpoint")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help=f"Training-state file (default: <save-path dir>/{TRAINING_STATE_NAME})")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save training state every N episodes (default 100)")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Run at most N episodes in this session, save training state, then exit")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation at the end")
    parser.add_argument("--final-val-episodes", type=int, default=200,
                        help="Unseen VAL seeds for the final re-evaluation of best/final weights (0 = skip)")
    parser.add_argument("--threads", type=int, default=None,
                        help=f"Torch intra-op threads (default {default_num_threads()} = 3/4 of logical CPUs)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device for training ('cuda', 'cpu', or 'auto'; default: auto detects CUDA)")
    args = parser.parse_args()

    hyper = {
        "total_steps": args.total_steps, "episodes": args.episodes, "lr": args.lr,
        "batch_size": args.batch_size, "gamma": args.gamma, "n_step": args.n_step,
        "train_every": args.train_every, "target_tau": args.target_tau,
        "target_update_steps": args.target_update_steps, "buffer_capacity": args.buffer_capacity,
        "val_every_steps": args.val_every_steps, "val_every": args.val_every,
        "seed": args.seed, "save_path": args.save_path, "egocentric": args.egocentric,
    }
    passed = {k: v for k, v in hyper.items() if v is not None}
    if args.resume and passed:
        parser.error(f"--resume uses the checkpoint's hyperparameters; drop {', '.join('--' + k.replace('_', '-') for k in passed)}")

    common = dict(checkpoint_every=args.checkpoint_every, stop_after=args.stop_after, make_plots=not args.no_plots,
                  final_val_episodes=args.final_val_episodes, num_threads=args.threads, device=args.device)
    if args.resume:
        train_dqn(resume=True, checkpoint_path=args.checkpoint_path
                  or default_training_state_path(), **common)
    else:
        train_dqn(checkpoint_path=args.checkpoint_path, **passed, **common)

