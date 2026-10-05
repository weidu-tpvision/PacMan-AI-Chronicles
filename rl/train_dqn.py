"""
Training pipeline for Deep Q-Network (DQN) and Double-DQN on Pac-Man.
Implements:
- Unentangled 6-channel state encoding with velocity/momentum tracking
- Global 10x10 receptive field via MaxPool2d(2)
- Experience Replay Buffer (breaks temporal correlation)
- Target Network (Double DQN to prevent Q-value overestimation)
- Smooth L1 (Huber) Loss and Adam optimization
- Decaying epsilon-greedy exploration
- Momentum-preserving reward shaping and anti-stall cutoff
- Model checkpointing to rl/weights/dqn_pacman.pt
"""

import argparse
import collections
import csv
import json
import os
import random
import statistics
import sys
import time
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from rl.plot_metrics import plot_metrics

from core.environment import Environment
from core.maze_data import OPPOSITE_DIRECTIONS
from rl.dqn_model import (
    ACTIONS,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    NUM_CHANNELS,
    PacmanDQN,
    encode_state,
)

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "dqn_pacman.pt")


class ReplayBuffer:
    def __init__(self, capacity: int = 40000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)


def evaluate_dqn(policy_net: nn.Module, episodes: int = 15, max_steps: int = 400) -> Tuple[float, float]:
    """Evaluate current policy deterministically without epsilon exploration using 6-channel state."""
    policy_net.eval()
    scores = []
    pellets_cleared = []

    with torch.no_grad():
        for seed in range(episodes):
            env = Environment(seed=10000 + seed)
            score = 0
            pellets_count = 0
            prev_p = None
            prev_g = None

            for _ in range(max_steps):
                legal = env.get_legal_moves(env.pacman_pos[0], env.pacman_pos[1])
                if not legal:
                    break

                curr_p = tuple(env.pacman_pos)
                curr_g = [tuple(g) for g in env.ghost_positions]
                s_arr = encode_state(curr_p, curr_g, env.pellets, prev_p, prev_g)
                s_t = torch.from_numpy(s_arr).unsqueeze(0)
                q_vals = policy_net(s_t).squeeze(0)

                legal_q = {m: q_vals[ACTION_TO_IDX[m]].item() for m in legal}
                if env.last_move and len(legal) > 1:
                    opp = OPPOSITE_DIRECTIONS.get(env.last_move)
                    if opp in legal_q:
                        legal_q[opp] -= 1.0

                best_q = max(legal_q.values())
                best_moves = [m for m in legal if legal_q[m] == best_q]
                move = random.choice(best_moves)

                prev_p = curr_p
                prev_g = curr_g

                collided, ate_pellet, won = env.step(move)
                if ate_pellet:
                    score += 10
                    pellets_count += 1
                if collided:
                    score -= 200
                    break
                if won:
                    score += 500
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
    target_update_steps: int = 400,
    warmup_steps: int = 1000,
    save_path: str = DEFAULT_MODEL_PATH,
):
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required to train DQN.")
        return

    weights_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(weights_dir, "dqn_training.log")
    json_path = os.path.join(weights_dir, "dqn_training_metrics.json")
    csv_path = os.path.join(weights_dir, "dqn_training_metrics.csv")

    log_file = open(log_path, "w", encoding="utf-8")

    def log_print(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    device = torch.device("cpu")

    policy_net = PacmanDQN(in_channels=NUM_CHANNELS).to(device)
    target_net = PacmanDQN(in_channels=NUM_CHANNELS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # Huber Loss for stable gradients
    replay_buffer = ReplayBuffer(capacity=40000)

    log_print("======================================================================")
    log_print(" DEEP Q-NETWORK (DQN) TRAINING: 6-Channel Velocity-Aware CNN")
    log_print("======================================================================")
    log_print(f"Device: {device} | Batch Size: {batch_size} | Learning Rate: {lr}")
    log_print(f"Episodes: {episodes} | Target Sync: every {target_update_steps} steps\n")
    log_print(f"Warming up replay buffer with {warmup_steps:,} baseline steps...")

    # Warm-up phase
    warmup_env = Environment(seed=42)
    w_prev_p = None
    w_prev_g = None

    for _ in range(warmup_steps):
        legal = warmup_env.get_legal_moves(warmup_env.pacman_pos[0], warmup_env.pacman_pos[1])
        if not legal:
            warmup_env = Environment(seed=random.randint(0, 99999))
            w_prev_p = None
            w_prev_g = None
            legal = warmup_env.get_legal_moves(warmup_env.pacman_pos[0], warmup_env.pacman_pos[1])

        curr_p = tuple(warmup_env.pacman_pos)
        curr_g = [tuple(g) for g in warmup_env.ghost_positions]
        s = encode_state(curr_p, curr_g, warmup_env.pellets, w_prev_p, w_prev_g)

        m = random.choice(legal)
        act_idx = ACTION_TO_IDX[m]

        col, ate, won = warmup_env.step(m)
        r = -0.5
        if ate:
            r += 15.0
        if col:
            r -= 150.0
        elif won:
            r += 300.0

        next_p = tuple(warmup_env.pacman_pos)
        next_g = [tuple(g) for g in warmup_env.ghost_positions]
        ns = encode_state(next_p, next_g, warmup_env.pellets, curr_p, curr_g)
        done = col or won
        replay_buffer.push(s, act_idx, r, ns, float(done))

        w_prev_p = curr_p
        w_prev_g = curr_g

        if done:
            warmup_env = Environment(seed=random.randint(0, 99999))
            w_prev_p = None
            w_prev_g = None

    log_print("Warm-up complete! Beginning DQN neural optimization...\n")

    epsilon = epsilon_start
    best_val_score = -999999.0
    total_steps = 0
    t0 = time.time()
    recent_scores = collections.deque(maxlen=30)
    metrics_history = []

    for ep in range(episodes):
        env = Environment(seed=random.randint(0, 100000))
        ep_reward = 0.0
        pellets_eaten = 0
        collided = False
        steps_without_pellet = 0
        prev_p = None
        prev_g = None
        ep_losses = []

        for step in range(350):
            total_steps += 1
            legal = env.get_legal_moves(env.pacman_pos[0], env.pacman_pos[1])
            if not legal:
                break

            curr_p = tuple(env.pacman_pos)
            curr_g = [tuple(g) for g in env.ghost_positions]
            s = encode_state(curr_p, curr_g, env.pellets, prev_p, prev_g)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                m = random.choice(legal)
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(s).unsqueeze(0).to(device)
                    q_vals = policy_net(s_t).squeeze(0)
                    legal_q = {mv: q_vals[ACTION_TO_IDX[mv]].item() for mv in legal}
                    if env.last_move and len(legal) > 1:
                        opp = OPPOSITE_DIRECTIONS.get(env.last_move)
                        if opp in legal_q:
                            legal_q[opp] -= 1.0

                    best_q = max(legal_q.values())
                    best_moves = [mv for mv in legal if legal_q[mv] == best_q]
                    m = random.choice(best_moves)

            act_idx = ACTION_TO_IDX[m]
            col, ate, won = env.step(m)

            # Reward shaping
            r = -0.5  # Base time step cost to prevent passive lingering
            if ate:
                r += 15.0
                pellets_eaten += 1
                steps_without_pellet = 0
            else:
                steps_without_pellet += 1

            if col:
                r -= 150.0
                collided = True
            elif won:
                r += 300.0

            # Reversal penalty to teach forward momentum through corridors
            if env.last_move and m == OPPOSITE_DIRECTIONS.get(env.last_move) and len(legal) > 1:
                r -= 1.5

            # Anti-stall / anti-orbit penalty:
            # If Pac-Man makes 45 consecutive steps without eating a single pellet,
            # it is stuck in an empty cycle. Terminate with stall penalty.
            stalled = False
            if steps_without_pellet >= 45 and not col and not won:
                r -= 50.0
                stalled = True

            ep_reward += r
            next_p = tuple(env.pacman_pos)
            next_g = [tuple(g) for g in env.ghost_positions]
            ns = encode_state(next_p, next_g, env.pellets, curr_p, curr_g)
            done = col or won or stalled

            replay_buffer.push(s, act_idx, r, ns, float(done))

            prev_p = curr_p
            prev_g = curr_g

            # Sample batch & optimize
            if len(replay_buffer) >= batch_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                b_s = b_s.to(device)
                b_a = b_a.to(device)
                b_r = b_r.to(device)
                b_ns = b_ns.to(device)
                b_d = b_d.to(device)

                # Current Q(s, a)
                curr_q = policy_net(b_s).gather(1, b_a)

                # Double DQN target computation
                with torch.no_grad():
                    next_policy_q = policy_net(b_ns)
                    best_next_actions = next_policy_q.argmax(dim=1, keepdim=True)
                    next_target_q = target_net(b_ns).gather(1, best_next_actions)
                    expected_q = b_r + (gamma * next_target_q * (1.0 - b_d))

                loss = loss_fn(curr_q, expected_q)
                ep_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                optimizer.step()

            # Target network synchronization
            if total_steps % target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Decay exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        score = (pellets_eaten * 10) - (200 if collided else 0)
        recent_scores.append(score)
        avg_rec = statistics.mean(recent_scores)
        avg_loss = statistics.mean(ep_losses) if ep_losses else 0.0

        is_val = ((ep + 1) % 50 == 0 or ep + 1 == episodes)
        val_sc, val_pel = None, None
        mark = ""
        if is_val:
            val_sc, val_pel = evaluate_dqn(policy_net, episodes=10)
            if val_sc > best_val_score:
                best_val_score = val_sc
                torch.save(policy_net.state_dict(), save_path)
                mark = " [* SAVED BEST]"

            elapsed = time.time() - t0
            log_print(
                f"Ep {ep+1:04d}/{episodes} | Train Avg: {avg_rec:6.1f} | "
                f"Val Score: {val_sc:6.1f} (Pellets: {val_pel:4.1f}) | "
                f"eps: {epsilon:4.2f} | Time: {elapsed:4.0f}s{mark}"
            )

        metrics_history.append({
            "episode": ep + 1,
            "score": score,
            "train_avg_score": round(avg_rec, 1),
            "pellets": pellets_eaten,
            "reward": round(ep_reward, 1),
            "steps": step + 1,
            "loss": round(avg_loss, 4),
            "epsilon": round(epsilon, 4),
            "val_score": round(val_sc, 1) if val_sc is not None else None,
            "val_pellets": round(val_pel, 1) if val_pel is not None else None,
        })

    log_print("\n======================================================================")
    log_print(f" TRAINING COMPLETE! Best Checkpoint Saved to {save_path}")
    log_print(f" Best Validation Score: {best_val_score:.1f}")
    log_print("======================================================================")
    log_file.close()

    # Export metrics files
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    if metrics_history:
        keys = list(metrics_history[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics_history)

    # Generate figures
    plot_metrics(json_path, weights_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Pac-Man")
    parser.add_argument("--episodes", type=int, default=1200, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to save model weights")
    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )
