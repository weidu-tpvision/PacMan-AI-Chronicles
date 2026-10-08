"""
Training pipeline for Deep Q-Network (DQN) and Double-DQN on Pac-Man.
Implements:
- Unentangled 6-channel state encoding with velocity/momentum tracking
- Global 10x10 receptive field via MaxPool2d(2)
- Experience Replay Buffer (breaks temporal correlation)
- Target Network (Double DQN to prevent Q-value overestimation)
- Smooth L1 (Huber) Loss and Adam optimization
- Decaying epsilon-greedy exploration
- Momentum-preserving reward shaping (reversal penalty) and anti-stall truncation
- Fully seeded runs with disjoint TRAIN / VAL seed ranges (core.seeds)
- Model checkpointing to rl/weights/dqn_pacman.pt (selected on the *pure greedy* policy)
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
from rl.dqn_model import ACTION_TO_IDX, NUM_CHANNELS, PacmanDQN, encode_state

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "dqn_pacman.pt")

# Reward shaping constants
R_STEP = -0.5
R_PELLET = 15.0
R_DEATH = -150.0
R_WIN = 300.0
R_REVERSAL = -1.5
STALL_STEPS = 45


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
    Shaped reward for one transition.

    `prev_move` MUST be Pac-Man's heading *before* this step (read it before env.step()).
    Returns (reward, stalled). A stall (STALL_STEPS consecutive steps without a pellet)
    truncates the episode; it is not treated as a terminal state for bootstrapping.
    """
    r = R_STEP
    if ate:
        r += R_PELLET
    if collided:
        r += R_DEATH
    elif won:
        r += R_WIN
    if prev_move and n_legal > 1 and move == OPPOSITE_DIRECTIONS.get(prev_move):
        r += R_REVERSAL
    stalled = steps_without_pellet >= STALL_STEPS and not collided and not won
    if stalled:
        r -= 50.0
    return r, stalled


def greedy_action(policy_net, state: np.ndarray, legal) -> str:
    """Pure greedy argmax over legal actions (ties broken by the global RNG)."""
    with torch.no_grad():
        q = policy_net(torch.from_numpy(state).unsqueeze(0)).squeeze(0)
    legal_q = {m: q[ACTION_TO_IDX[m]].item() for m in legal}
    best = max(legal_q.values())
    return random.choice([m for m in legal if legal_q[m] == best])


def masked_next_actions(q_values: torch.Tensor, legal_action_masks: torch.Tensor) -> torch.Tensor:
    """Select the highest-valued legal action for each next state."""
    masked_q = q_values.masked_fill(~legal_action_masks, torch.finfo(q_values.dtype).min)
    return masked_q.argmax(dim=1, keepdim=True)


class ReplayBuffer:
    """Proportional prioritized replay with importance-sampling correction."""
    def __init__(self, capacity: int = 40000, rng: Optional[random.Random] = None, alpha: float = 0.6, priority_epsilon: float = 1e-5):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.rng = rng or random.Random()
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.position = 0

    def push(self, state, action_idx, reward, next_state, done, next_action_mask=None):
        max_priority = float(self.priorities[:len(self.buffer)].max()) if self.buffer else 1.0
        if next_action_mask is None:
            next_action_mask = np.ones(len(ACTION_TO_IDX), dtype=np.bool_)
        item = (state, action_idx, reward, next_state, done, next_action_mask)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        priorities = self.priorities[:len(self.buffer)]
        scaled = priorities ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = self.rng.choices(range(len(self.buffer)), weights=probabilities, k=batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        next_action_masks_t = torch.tensor(np.array(next_action_masks), dtype=torch.bool)

        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        return states_t, actions_t, rewards_t, next_states_t, dones_t, next_action_masks_t, indices, weights_t

    def update_priorities(self, indices, td_errors):
        for index, error in zip(indices, td_errors):
            self.priorities[index] = abs(float(error)) + self.priority_epsilon

    def __len__(self):
        return len(self.buffer)


def evaluate_dqn(policy_net: nn.Module, episodes: int = 15, max_steps: int = DEFAULT_MAX_STEPS) -> Tuple[float, float]:
    """Evaluate the pure greedy policy (no inference heuristics) on VAL seeds."""
    policy_net.eval()
    scores, pellets_cleared = [], []

    for seed in val_seeds(episodes):
        seed_everything(seed)
        env = Environment(seed=seed)
        score = pellets_count = 0
        prev_p = prev_g = None

        for _ in range(max_steps):
            legal = env.get_legal_moves(*env.pacman_pos)
            curr_p = tuple(env.pacman_pos)
            curr_g = [tuple(g) for g in env.ghost_positions]
            move = greedy_action(policy_net, encode_state(curr_p, curr_g, env.pellets, prev_p, prev_g), legal)
            prev_p, prev_g = curr_p, curr_g

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
    return statistics.mean(scores), statistics.mean(pellets_cleared)


def train_dqn(
    episodes: int = 1200,
    lr: float = 5e-4,
    batch_size: int = 64,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9975,
    per_beta_start: float = 0.4,
    target_update_steps: int = 400,
    warmup_steps: int = 1000,
    max_steps: int = DEFAULT_MAX_STEPS,
    val_every: int = 50,
    val_episodes: int = 20,
    seed: int = 0,
    save_path: str = DEFAULT_MODEL_PATH,
    make_plots: bool = True,
):
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required to train DQN.")
        return None

    seed_everything(seed)
    rng = random.Random(seed)

    weights_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(weights_dir, "dqn_training.log")
    json_path = os.path.join(weights_dir, "dqn_training_metrics.json")
    csv_path = os.path.join(weights_dir, "dqn_training_metrics.csv")

    device = torch.device("cpu")
    policy_net = PacmanDQN(in_channels=NUM_CHANNELS).to(device)
    target_net = PacmanDQN(in_channels=NUM_CHANNELS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, episodes), eta_min=lr * 0.1)
    replay_buffer = ReplayBuffer(capacity=40000, rng=rng)

    metrics_history = []
    best_val_score = -float("inf")
    best_val_episode = None

    with open(log_path, "w", encoding="utf-8") as log_file:

        def log_print(msg: str = ""):
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        log_print("======================================================================")
        log_print(" DEEP Q-NETWORK (DQN) TRAINING: 6-Channel Velocity-Aware CNN")
        log_print("======================================================================")
        log_print(f"Device: {device} | Batch Size: {batch_size} | Learning Rate: {lr} | Seed: {seed}")
        log_print(f"Episodes: {episodes} | Horizon: {max_steps} | Target Sync: every {target_update_steps} steps")
        log_print(f"Validation: {val_episodes} VAL seeds every {val_every} episodes (pure greedy policy)\n")
        log_print(f"Warming up replay buffer with {warmup_steps:,} random-policy steps...")

        # ---- Warm-up phase (uniform random policy) ----
        env = Environment(seed=train_seed(rng))
        prev_p = prev_g = None
        no_pellet = 0
        for _ in range(warmup_steps):
            legal = env.get_legal_moves(*env.pacman_pos)
            curr_p = tuple(env.pacman_pos)
            curr_g = [tuple(g) for g in env.ghost_positions]
            s = encode_state(curr_p, curr_g, env.pellets, prev_p, prev_g)

            m = rng.choice(legal)
            prev_move = env.last_move
            col, ate, won = env.step(m)
            no_pellet = 0 if ate else no_pellet + 1
            r, stalled = compute_reward(prev_move, m, ate, col, won, len(legal), no_pellet)

            ns = encode_state(tuple(env.pacman_pos), [tuple(g) for g in env.ghost_positions], env.pellets, curr_p, curr_g)
            next_legal = env.get_legal_moves(*env.pacman_pos)
            next_mask = np.array([a in next_legal for a in ACTION_TO_IDX], dtype=np.bool_)
            if not next_mask.any():
                next_mask[:] = True
            replay_buffer.push(s, ACTION_TO_IDX[m], r, ns, float(col or won), next_mask)
            prev_p, prev_g = curr_p, curr_g

            if col or won or stalled:
                env = Environment(seed=train_seed(rng))
                prev_p = prev_g = None
                no_pellet = 0

        log_print("Warm-up complete! Beginning DQN neural optimization...\n")

        epsilon = epsilon_start
        total_steps = 0
        t0 = time.time()
        recent_scores = collections.deque(maxlen=50)

        for ep in range(episodes):
            env = Environment(seed=train_seed(rng))
            ep_reward = 0.0
            pellets_eaten = 0
            collided = won = False
            steps_without_pellet = 0
            prev_p = prev_g = None
            ep_losses = []
            steps = 0

            for steps in range(1, max_steps + 1):
                total_steps += 1
                legal = env.get_legal_moves(*env.pacman_pos)
                curr_p = tuple(env.pacman_pos)
                curr_g = [tuple(g) for g in env.ghost_positions]
                s = encode_state(curr_p, curr_g, env.pellets, prev_p, prev_g)

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

                r, stalled = compute_reward(prev_move, m, ate, col, won, len(legal), steps_without_pellet)
                ep_reward += r

                ns = encode_state(
                    tuple(env.pacman_pos), [tuple(g) for g in env.ghost_positions], env.pellets, curr_p, curr_g
                )
                terminal = col or won  # stalls / horizon are truncations, not terminals
                next_legal = env.get_legal_moves(*env.pacman_pos)
                next_mask = np.array([a in next_legal for a in ACTION_TO_IDX], dtype=np.bool_)
                if not next_mask.any():
                    next_mask[:] = True
                replay_buffer.push(s, ACTION_TO_IDX[m], r, ns, float(terminal), next_mask)
                prev_p, prev_g = curr_p, curr_g

                # Sample batch & optimize
                if len(replay_buffer) >= batch_size:
                    beta = min(1.0, per_beta_start + (1.0 - per_beta_start) * total_steps / max(1, episodes * max_steps))
                    b_s, b_a, b_r, b_ns, b_d, b_next_mask, indices, is_weights = replay_buffer.sample(batch_size, beta)
                    curr_q = policy_net(b_s).gather(1, b_a)

                    # Double DQN target computation
                    with torch.no_grad():
                        best_next_actions = masked_next_actions(policy_net(b_ns), b_next_mask)
                        next_target_q = target_net(b_ns).gather(1, best_next_actions)
                        expected_q = b_r + gamma * next_target_q * (1.0 - b_d)

                    td_errors = expected_q - curr_q
                    per_item_loss = nn.functional.smooth_l1_loss(curr_q, expected_q, reduction="none")
                    loss = (is_weights * per_item_loss).mean()
                    replay_buffer.update_priorities(indices, td_errors.detach().squeeze(1).cpu().numpy())
                    ep_losses.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                    optimizer.step()

                # Target network synchronization
                if total_steps % target_update_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if terminal or stalled:
                    break

            # Decay exploration
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
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
                    torch.save(policy_net.state_dict(), save_path)
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

        log_print("\n======================================================================")
        log_print(f" TRAINING COMPLETE! Best Checkpoint Saved to {save_path}")
        log_print(f" Best Validation Score: {best_val_score:.1f} (episode {best_val_episode})")
        log_print("======================================================================")

    # Export metrics files
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    if metrics_history:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_history)

    if make_plots:
        from rl.plot_metrics import plot_metrics  # lazy: pulls in pygame
        plot_metrics(json_path, weights_dir)

    return {"best_val_score": best_val_score, "best_val_episode": best_val_episode, "metrics": metrics_history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pac-Man")
    parser.add_argument("--episodes", type=int, default=1200, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    parser.add_argument("--save-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to save model weights")
    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        save_path=args.save_path,
    )
