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
from rl.dqn_model import ACTION_TO_IDX, NUM_CHANNELS, NUM_SCALARS, STALL_STEPS, PacmanDQN, encode_state

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "dqn_pacman.pt")
TRAINING_STATE_NAME = "dqn_training_state.pt"
TRAINING_STATE_VERSION = 4  # v4: run-length-relative epsilon schedule

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


def observe(env: Environment, steps_without_pellet: int, steps_remaining: int, horizon: int):
    """(grid, scalars) observation of the environment's current state."""
    return encode_state(
        tuple(env.pacman_pos), [tuple(g) for g in env.ghost_positions], env.pellets,
        last_move=env.last_move, ghost_dirs=env.ghost_dirs, mode_step=env.mode_step,
        steps_without_pellet=steps_without_pellet, steps_remaining=steps_remaining, horizon=horizon,
    )


def greedy_action(policy_net, obs, legal) -> str:
    """Pure greedy argmax over legal actions (ties broken by the global RNG)."""
    grid, scalars = obs
    with torch.no_grad():
        q = policy_net(torch.from_numpy(grid).unsqueeze(0), torch.from_numpy(scalars).unsqueeze(0)).squeeze(0)
    legal_q = {m: q[ACTION_TO_IDX[m]].item() for m in legal}
    best = max(legal_q.values())
    return random.choice([m for m in legal if legal_q[m] == best])


def masked_next_actions(q_values: torch.Tensor, legal_action_masks: torch.Tensor) -> torch.Tensor:
    """Select the highest-valued legal action for each next state."""
    masked_q = q_values.masked_fill(~legal_action_masks, torch.finfo(q_values.dtype).min)
    return masked_q.argmax(dim=1, keepdim=True)


class ReplayBuffer:
    """Proportional prioritized replay with importance-sampling correction.

    Observations are (grid, scalars) pairs stored in preallocated ring arrays (allocated
    on the first push, once the shapes are known). Grids are stored as float16 (they are
    binary apart from the cycle-phase plane, which keeps ~3 decimal digits), halving
    RAM; scalars stay float32. sample() returns float32 tensors. The arrays also make
    state_dict() a zero-copy snapshot for resumable training checkpoints.
    """
    def __init__(self, capacity: int = 40000, rng: Optional[random.Random] = None, alpha: float = 0.6, priority_epsilon: float = 1e-5):
        self.capacity = capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.rng = rng or random.Random()
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.position = 0
        self.size = 0
        self.max_priority = 1.0  # new transitions get the highest priority seen so far
        self.grids = self.scalars = self.next_grids = self.next_scalars = None
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_action_masks = np.ones((capacity, len(ACTION_TO_IDX)), dtype=np.bool_)

    def _allocate(self, grid_shape, num_scalars: int):
        self.grids = np.zeros((self.capacity, *grid_shape), dtype=np.float16)
        self.next_grids = np.zeros((self.capacity, *grid_shape), dtype=np.float16)
        self.scalars = np.zeros((self.capacity, num_scalars), dtype=np.float32)
        self.next_scalars = np.zeros((self.capacity, num_scalars), dtype=np.float32)

    def push(self, obs, action_idx, reward, next_obs, done, next_action_mask=None):
        grid, scalars = obs
        next_grid, next_scalars = next_obs
        if self.grids is None:
            self._allocate(np.shape(grid), len(scalars))
        i = self.position
        self.grids[i] = grid
        self.scalars[i] = scalars
        self.next_grids[i] = next_grid
        self.next_scalars[i] = next_scalars
        self.actions[i] = action_idx
        self.rewards[i] = reward
        self.dones[i] = done
        self.next_action_masks[i] = True if next_action_mask is None else next_action_mask
        self.priorities[i] = self.max_priority
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> dict:
        priorities = self.priorities[:self.size]
        scaled = priorities ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = self.rng.choices(range(self.size), weights=probabilities, k=batch_size)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return {
            "grids": torch.from_numpy(self.grids[indices].astype(np.float32)),
            "scalars": torch.from_numpy(self.scalars[indices]),
            "actions": torch.from_numpy(self.actions[indices]).unsqueeze(1),
            "rewards": torch.from_numpy(self.rewards[indices]).unsqueeze(1),
            "next_grids": torch.from_numpy(self.next_grids[indices].astype(np.float32)),
            "next_scalars": torch.from_numpy(self.next_scalars[indices]),
            "dones": torch.from_numpy(self.dones[indices]).unsqueeze(1),
            "next_action_masks": torch.from_numpy(self.next_action_masks[indices]),
            "indices": indices,
            "weights": torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
        }

    def update_priorities(self, indices, td_errors):
        for index, error in zip(indices, td_errors):
            priority = abs(float(error)) + self.priority_epsilon
            self.priorities[index] = priority
            self.max_priority = max(self.max_priority, priority)

    _OBS_ARRAYS = ("grids", "scalars", "next_grids", "next_scalars")
    _TRANSITION_ARRAYS = ("priorities", "actions", "rewards", "dones", "next_action_masks")

    def state_dict(self) -> dict:
        """Tensor snapshot of the filled part of the buffer (shares memory, no copy)."""
        n = self.size
        state = {
            "capacity": self.capacity, "alpha": self.alpha, "priority_epsilon": self.priority_epsilon,
            "position": self.position, "size": n, "max_priority": self.max_priority,
        }
        for name in self._TRANSITION_ARRAYS:
            state[name] = torch.from_numpy(getattr(self, name)[:n])
        if self.grids is not None:
            for name in self._OBS_ARRAYS:
                state[name] = torch.from_numpy(getattr(self, name)[:n])
        return state

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError(f"replay capacity mismatch: checkpoint {state['capacity']} vs buffer {self.capacity}")
        n = state["size"]
        self.alpha = state["alpha"]
        self.priority_epsilon = state["priority_epsilon"]
        self.position = state["position"]
        self.size = n
        self.max_priority = state["max_priority"]
        for name in self._TRANSITION_ARRAYS:
            getattr(self, name)[:n] = state[name].numpy()
        if "grids" in state:
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
    return {
        "trainer": rng.getstate(),
        "python": random.getstate(),
        "numpy": (np_name, np_keys.tolist(), np_pos, np_has_gauss, np_gauss),
        "torch": torch.get_rng_state(),
    }


def _rng_restore(rng: random.Random, snap: dict) -> None:
    rng.setstate(snap["trainer"])
    random.setstate(snap["python"])
    np_name, np_keys, np_pos, np_has_gauss, np_gauss = snap["numpy"]
    np.random.set_state((np_name, np.array(np_keys, dtype=np.uint32), np_pos, np_has_gauss, np_gauss))
    torch.set_rng_state(snap["torch"])


def default_training_state_path(save_path: str = DEFAULT_MODEL_PATH) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(save_path)), TRAINING_STATE_NAME)


def epsilon_schedule(episodes_done: int, episodes: int, start: float, minimum: float, decay_fraction: float) -> float:
    """Exploration rate for the next episode: exponential decay from `start` that reaches
    `minimum` after `decay_fraction` of the planned episodes, so short and long runs get
    the same exploration profile. A pure function of the episode count (resume-exact)."""
    decay_episodes = max(1.0, decay_fraction * episodes)
    progress = min(1.0, episodes_done / decay_episodes)
    return max(minimum, start * (minimum / start) ** progress)


def evaluate_dqn(policy_net: nn.Module, episodes: int = 15, max_steps: int = DEFAULT_MAX_STEPS) -> Tuple[float, float]:
    """Mean score and pellets of the pure greedy policy (no inference heuristics) on the
    first `episodes` VAL seeds (the checkpoint-selection seeds)."""
    scores, pellets = evaluate_dqn_episodes(policy_net, val_seeds(episodes), max_steps)
    return statistics.mean(scores), statistics.mean(pellets)


def evaluate_dqn_episodes(policy_net: nn.Module, seeds, max_steps: int = DEFAULT_MAX_STEPS) -> Tuple[list, list]:
    """Per-episode scores and pellets of the pure greedy policy on `seeds`."""
    policy_net.eval()
    scores, pellets_cleared = [], []

    for seed in seeds:
        seed_everything(seed)
        env = Environment(seed=seed)
        score = pellets_count = 0

        steps_without_pellet = 0
        for step_index in range(max_steps):
            legal = env.get_legal_moves(*env.pacman_pos)
            obs = observe(env, steps_without_pellet, max_steps - step_index, max_steps)
            move = greedy_action(policy_net, obs, legal)

            collided, ate_pellet, won = env.step(move)
            steps_without_pellet = 0 if ate_pellet else steps_without_pellet + 1
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
    episodes: int = 1200,
    lr: float = 5e-4,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay_fraction: float = 0.6,
    per_beta_start: float = 0.4,
    target_update_steps: int = 400,
    warmup_steps: int = 1000,
    max_steps: int = DEFAULT_MAX_STEPS,
    val_every: int = 50,
    val_episodes: int = 20,
    seed: int = 0,
    save_path: str = DEFAULT_MODEL_PATH,
    make_plots: bool = True,
    buffer_capacity: int = 40000,
    checkpoint_every: int = 100,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    stop_after: Optional[int] = None,
    final_val_episodes: int = 200,
):
    """Train (or resume training) the DQN.

    Resumable runs: a training-state checkpoint (`checkpoint_path`, default
    rl/weights/dqn_training_state.pt) is written every `checkpoint_every` episodes, after
    `stop_after` episodes of this session, and on Ctrl+C. With `resume=True` the run
    continues from it using the *checkpointed* hyperparameters (the arguments above are
    ignored except `checkpoint_path`, `stop_after`, `checkpoint_every`, `make_plots` and
    `final_val_episodes`).

    Exploration decays from `epsilon_start` to `epsilon_min` over `epsilon_decay_fraction`
    of `episodes`. When the run completes, the best checkpoint and the final weights are
    re-evaluated on `final_val_episodes` VAL seeds that were *not* used for checkpoint
    selection (selection picks the best of many noisy validations, so its own score is
    optimistic); results go to the log, the return value and dqn_final_eval.json.
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
        episodes, lr, batch_size, gamma = cfg["episodes"], cfg["lr"], cfg["batch_size"], cfg["gamma"]
        epsilon_start, epsilon_min, epsilon_decay_fraction = cfg["epsilon_start"], cfg["epsilon_min"], cfg["epsilon_decay_fraction"]
        per_beta_start, target_update_steps = cfg["per_beta_start"], cfg["target_update_steps"]
        warmup_steps, max_steps, val_every, val_episodes = cfg["warmup_steps"], cfg["max_steps"], cfg["val_every"], cfg["val_episodes"]
        seed, save_path, buffer_capacity = cfg["seed"], cfg["save_path"], cfg["buffer_capacity"]

    config = {
        "episodes": episodes, "lr": lr, "batch_size": batch_size, "gamma": gamma,
        "epsilon_start": epsilon_start, "epsilon_min": epsilon_min, "epsilon_decay_fraction": epsilon_decay_fraction,
        "per_beta_start": per_beta_start, "target_update_steps": target_update_steps,
        "warmup_steps": warmup_steps, "max_steps": max_steps, "val_every": val_every,
        "val_episodes": val_episodes, "seed": seed, "save_path": save_path, "buffer_capacity": buffer_capacity,
    }

    seed_everything(seed)
    rng = random.Random(seed)

    weights_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(weights_dir, "dqn_training.log")
    json_path = os.path.join(weights_dir, "dqn_training_metrics.json")
    csv_path = os.path.join(weights_dir, "dqn_training_metrics.csv")
    final_eval_path = os.path.join(weights_dir, "dqn_final_eval.json")

    device = torch.device("cpu")
    policy_net = PacmanDQN().to(device)
    target_net = PacmanDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, episodes), eta_min=lr * 0.1)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, rng=rng)

    metrics_history = []
    best_val_score = -float("inf")
    best_val_episode = None
    epsilon = epsilon_start
    total_steps = 0
    start_episode = 0
    elapsed_before = 0.0
    recent_scores = collections.deque(maxlen=50)

    if ckpt is not None:
        policy_net.load_state_dict(ckpt["policy_net"])
        target_net.load_state_dict(ckpt["target_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        replay_buffer.load_state_dict(ckpt["replay_buffer"])
        prog = ckpt["progress"]
        start_episode, total_steps, epsilon = prog["episodes_done"], prog["total_steps"], prog["epsilon"]
        best_val_score, best_val_episode = prog["best_val_score"], prog["best_val_episode"]
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
            "policy_net": policy_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
            "progress": {
                "episodes_done": episodes_done, "total_steps": total_steps, "epsilon": epsilon,
                "best_val_score": best_val_score, "best_val_episode": best_val_episode,
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

        if start_episode:
            log_print(f"\n[RESUME] Continuing from episode {start_episode}/{episodes} "
                      f"({os.path.relpath(checkpoint_path)}, {len(replay_buffer):,} replay transitions)\n")
        else:
            log_print("======================================================================")
            log_print(f" DEEP Q-NETWORK (DQN) TRAINING: {NUM_CHANNELS} grid planes + {NUM_SCALARS} scalars")
            log_print("======================================================================")
            log_print(f"Device: {device} | Batch Size: {batch_size} | Learning Rate: {lr} | Gamma: {gamma} | Seed: {seed}")
            log_print(f"Episodes: {episodes} | Horizon: {max_steps} | Target Sync: every {target_update_steps} steps")
            log_print(f"Validation: {val_episodes} VAL seeds every {val_every} episodes (pure greedy policy)")
            log_print(f"Training state: {os.path.relpath(checkpoint_path)} every {checkpoint_every} episodes\n")
            log_print(f"Warming up replay buffer with {warmup_steps:,} random-policy steps...")

            # ---- Warm-up phase (uniform random policy) ----
            env = Environment(seed=train_seed(rng))
            for _ in range(warmup_steps):
                legal = env.get_legal_moves(*env.pacman_pos)
                s = observe(env, env.steps_without_pellet, max(0, max_steps - env.step_count), max_steps)

                m = rng.choice(legal)
                prev_move = env.last_move
                col, ate, won = env.step(m)
                r, _ = compute_reward(prev_move, m, ate, col, won, len(legal), env.steps_without_pellet)

                ns = observe(env, env.steps_without_pellet, max(0, max_steps - env.step_count), max_steps)
                next_legal = env.get_legal_moves(*env.pacman_pos)
                next_mask = np.array([a in next_legal for a in ACTION_TO_IDX], dtype=np.bool_)
                if not next_mask.any():
                    next_mask[:] = True
                done = col or won or env.step_count >= max_steps
                replay_buffer.push(s, ACTION_TO_IDX[m], r, ns, float(done), next_mask)

                if done:
                    env = Environment(seed=train_seed(rng))

            log_print("Warm-up complete! Beginning DQN neural optimization...\n")

        t0 = time.time() - elapsed_before
        session_end = episodes if stop_after is None else min(episodes, start_episode + max(0, stop_after))
        try:
            for ep in range(start_episode, session_end):
                env = Environment(seed=train_seed(rng))
                epsilon = epsilon_schedule(ep, episodes, epsilon_start, epsilon_min, epsilon_decay_fraction)
                # IS exponent annealed over the episode schedule (episodes rarely run to
                # the horizon, so a step-based schedule would never reach beta = 1).
                beta = min(1.0, per_beta_start + (1.0 - per_beta_start) * ep / max(1, episodes - 1))
                ep_reward = 0.0
                pellets_eaten = 0
                collided = won = False
                steps_without_pellet = 0
                ep_losses = []
                steps = 0

                for steps in range(1, max_steps + 1):
                    total_steps += 1
                    legal = env.get_legal_moves(*env.pacman_pos)
                    s = observe(env, steps_without_pellet, max_steps - steps + 1, max_steps)

                    # Epsilon-greedy action selection over the raw network
                    if rng.random() < epsilon:
                        m = rng.choice(legal)
                    else:
                        m = greedy_action(policy_net, s, legal)

                    prev_move = env.last_move  # heading BEFORE the step (fixes dead reversal penalty)
                    col, ate, won = env.step(m)
                    if ate:
                        pellets_eaten += 1
                        steps_without_pellet = 0
                    else:
                        steps_without_pellet += 1
                    collided = col

                    r, _ = compute_reward(prev_move, m, ate, col, won, len(legal), steps_without_pellet)
                    ep_reward += r

                    ns = observe(env, steps_without_pellet, max_steps - steps, max_steps)
                    terminal = col or won or steps >= max_steps
                    next_legal = env.get_legal_moves(*env.pacman_pos)
                    next_mask = np.array([a in next_legal for a in ACTION_TO_IDX], dtype=np.bool_)
                    if not next_mask.any():
                        next_mask[:] = True
                    replay_buffer.push(s, ACTION_TO_IDX[m], r, ns, float(terminal), next_mask)

                    # Sample batch & optimize
                    if len(replay_buffer) >= batch_size:
                        batch = replay_buffer.sample(batch_size, beta)
                        curr_q = policy_net(batch["grids"], batch["scalars"]).gather(1, batch["actions"])

                        # Double DQN target computation
                        with torch.no_grad():
                            next_q_online = policy_net(batch["next_grids"], batch["next_scalars"])
                            best_next_actions = masked_next_actions(next_q_online, batch["next_action_masks"])
                            next_target_q = target_net(batch["next_grids"], batch["next_scalars"]).gather(1, best_next_actions)
                            expected_q = batch["rewards"] + gamma * next_target_q * (1.0 - batch["dones"])

                        td_errors = expected_q - curr_q
                        per_item_loss = nn.functional.smooth_l1_loss(curr_q, expected_q, reduction="none")
                        loss = (batch["weights"] * per_item_loss).mean()
                        replay_buffer.update_priorities(batch["indices"], td_errors.detach().squeeze(1).cpu().numpy())
                        ep_losses.append(loss.item())

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                        optimizer.step()

                    # Target network synchronization
                    if total_steps % target_update_steps == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                    if terminal:
                        break

                lr_scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

                score = pellets_eaten * SCORE_PELLET + (SCORE_DEATH if collided else 0) + (SCORE_WIN if won else 0)
                recent_scores.append(score)
                avg_rec = statistics.mean(recent_scores)
                avg_loss = statistics.mean(ep_losses) if ep_losses else 0.0

                val_sc = val_pel = None
                if (ep + 1) % val_every == 0 or ep + 1 == episodes:
                    # Validation re-seeds the global RNG; preserve the training stream afterwards.
                    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
                    val_sc, val_pel = evaluate_dqn(policy_net, episodes=val_episodes, max_steps=max_steps)
                    random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(torch_state)

                    mark = ""
                    if val_sc > best_val_score:
                        best_val_score = val_sc
                        best_val_episode = ep + 1
                        _atomic_torch_save(policy_net.state_dict(), save_path)
                        mark = " [* SAVED BEST]"

                    log_print(
                        f"Ep {ep+1:04d}/{episodes} | Train Avg(50): {avg_rec:6.1f} | "
                        f"Val Score: {val_sc:6.1f} (Pellets: {val_pel:4.1f}) | "
                        f"eps: {epsilon:4.2f} | lr: {current_lr:.2e} | Time: {time.time() - t0:4.0f}s{mark}"
                    )

                metrics_history.append({
                    "episode": ep + 1,
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

                done = ep + 1
                if done < episodes and (done % checkpoint_every == 0 or done == session_end):
                    save_training_state(done, time.time() - t0)
                    if done == session_end:
                        log_print(f"[CHECKPOINT] Stopped after episode {done}/{episodes}; "
                                  f"continue with --resume ({os.path.relpath(checkpoint_path)})")

        except KeyboardInterrupt:
            # Count only fully completed episodes (also correct if Ctrl+C lands inside a
            # periodic save). The interrupted episode's partial experience and updates are
            # kept, and a resume replays that episode index.
            done = len(metrics_history)
            log_print(f"\n[INTERRUPTED] Saving training state after episode {done}/{episodes}...")
            save_training_state(done, time.time() - t0)
            log_print(f"[CHECKPOINT] Saved to {os.path.relpath(checkpoint_path)}; continue with --resume")
            return {"completed": False, "episodes_done": done, "best_val_score": best_val_score,
                    "best_val_episode": best_val_episode, "metrics": metrics_history}

        if len(metrics_history) < episodes:
            return {"completed": False, "episodes_done": len(metrics_history), "best_val_score": best_val_score,
                    "best_val_episode": best_val_episode, "metrics": metrics_history}

        final_eval = None
        if final_val_episodes > 0:
            # Unbiased re-check: seeds right after the selection seeds, never used to pick a checkpoint.
            seeds = val_seeds(final_val_episodes, offset=val_episodes)
            log_print(f"\nFinal re-evaluation on {len(seeds)} unseen VAL seeds ({seeds[0]}..{seeds[-1]})...")
            best_net = PacmanDQN().to(device)
            best_net.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            final_eval = {
                "seeds": [seeds[0], seeds[-1]],
                "selection_seeds": [val_seeds(val_episodes)[0], val_seeds(val_episodes)[-1]],
                "best_checkpoint": {"episode": best_val_episode, "selection_score": best_val_score,
                                    **_summarize(*evaluate_dqn_episodes(best_net, seeds, max_steps))},
                "final_weights": {"episode": episodes, **_summarize(*evaluate_dqn_episodes(policy_net, seeds, max_steps))},
            }
            with open(final_eval_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(final_eval, f, indent=2)
            for label, key in (("Best checkpoint", "best_checkpoint"), ("Final weights  ", "final_weights")):
                r = final_eval[key]
                log_print(f"  {label} (ep {r['episode']}): {r['mean_score']:7.1f} +/- {r['score_ci95']:5.1f} "
                          f"(pellets {r['mean_pellets']:5.1f})")

        log_print("\n======================================================================")
        # Relative path: the log is a committed artifact and must not embed local directories
        log_print(f" TRAINING COMPLETE! Best Checkpoint Saved to {os.path.relpath(save_path)}")
        log_print(f" Best Validation Score: {best_val_score:.1f} (episode {best_val_episode}; selection score, optimistic)")
        log_print("======================================================================")

    export_metrics()
    # The resumable state only serves unfinished runs; drop it so a stale state can't be resumed.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if make_plots:
        from rl.plot_metrics import plot_metrics  # lazy: pulls in pygame
        plot_metrics(json_path, weights_dir)

    return {"completed": True, "episodes_done": episodes, "best_val_score": best_val_score,
            "best_val_episode": best_val_episode, "metrics": metrics_history, "final_eval": final_eval}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN on Pac-Man",
        epilog="Resume an interrupted run with --resume; hyperparameters then come from the checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes (default 1200)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default 5e-4)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default 64)")
    parser.add_argument("--seed", type=int, default=None, help="Global RNG seed (default 0)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save best model weights")
    parser.add_argument("--resume", action="store_true", help="Continue from the training-state checkpoint")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help=f"Training-state file (default: <save-path dir>/{TRAINING_STATE_NAME})")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save training state every N episodes (default 100)")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Run at most N episodes in this session, save training state, then exit")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation at the end")
    parser.add_argument("--final-val-episodes", type=int, default=200,
                        help="Unseen VAL seeds for the final re-evaluation of best/final weights (0 = skip)")
    args = parser.parse_args()

    hyper = {"episodes": args.episodes, "lr": args.lr, "batch_size": args.batch_size,
             "seed": args.seed, "save_path": args.save_path}
    passed = {k: v for k, v in hyper.items() if v is not None}
    if args.resume and passed:
        parser.error(f"--resume uses the checkpoint's hyperparameters; drop {', '.join('--' + k.replace('_', '-') for k in passed)}")

    common = dict(checkpoint_every=args.checkpoint_every, stop_after=args.stop_after, make_plots=not args.no_plots,
                  final_val_episodes=args.final_val_episodes)
    if args.resume:
        train_dqn(resume=True, checkpoint_path=args.checkpoint_path
                  or default_training_state_path(), **common)
    else:
        train_dqn(checkpoint_path=args.checkpoint_path, **passed, **common)
