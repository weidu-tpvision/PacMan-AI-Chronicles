"""
Approximate Q-Learning Training for Pac-Man with Linear Function Approximation.
Updates feature weights using Temporal Difference (TD) error:
w_i <- w_i + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a)) * f_i(s, a)
Saves learned parameters to rl/weights/learned_q_weights.json.
"""

import argparse
import json
import math
import os
import random
import statistics
import time
from typing import Dict, List, Optional, Tuple

from core.maze_data import (
    DIRECTIONS,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAZE_LAYOUT,
    OPPOSITE_DIRECTIONS,
    START_POSITIONS,
)

DEFAULT_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "learned_q_weights.json"
)


class TrainableQAgent:
    def __init__(
        self,
        alpha: float = 0.025,
        alpha_min: float = 0.002,
        alpha_decay: float = 0.998,
        gamma: float = 0.88,
        epsilon: float = 0.95,
        epsilon_decay: float = 0.996,
        epsilon_min: float = 0.01,
    ):
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.weights = {
            "ghost_1_step": -180.0,
            "ghost_2_step": -50.0,
            "ghost_3_step": -15.0,
            "ghost_safe_dist": 2.0,
            "eats_pellet": 35.0,
            "nearest_pellet_dist": -1.5,
            "reverse_penalty": -8.0,
        }

    def decay_parameters(self):
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def extract_features(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        action: str,
        last_move: Optional[str] = None,
    ) -> Dict[str, float]:
        px, py = pacman_pos
        dx, dy = DIRECTIONS[action]
        nx = (px + dx) % GRID_WIDTH
        ny = py + dy

        min_ghost_dist = min(
            abs(nx - gx) + abs(ny - gy) for gx, gy in ghost_positions
        )

        feats = {
            "ghost_1_step": 1.0 if min_ghost_dist <= 1 else 0.0,
            "ghost_2_step": 1.0 if min_ghost_dist == 2 else 0.0,
            "ghost_3_step": 1.0 if min_ghost_dist == 3 else 0.0,
            "ghost_safe_dist": float(min_ghost_dist) if min_ghost_dist > 3 else 0.0,
            "eats_pellet": 1.0 if (nx, ny) in pellets else 0.0,
            "nearest_pellet_dist": 0.0,
            "reverse_penalty": (
                1.0
                if (last_move and action == OPPOSITE_DIRECTIONS.get(last_move) and min_ghost_dist > 3)
                else 0.0
            ),
        }

        if pellets:
            sample_pellets = list(pellets)[:25]
            feats["nearest_pellet_dist"] = float(
                min(abs(nx - fx) + abs(ny - fy) for fx, fy in sample_pellets)
            )

        return feats

    def get_q_value(self, feats: Dict[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * feats.get(k, 0.0) for k in self.weights)

    def select_action(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
        explore: bool = True,
    ) -> Tuple[str, Dict[str, float]]:
        if explore and random.random() < self.epsilon:
            a = random.choice(legal_moves)
            return a, self.extract_features(pacman_pos, ghost_positions, pellets, a, last_move)

        best_q = -1e9
        best_acts = []
        best_feats = None

        for a in legal_moves:
            f = self.extract_features(pacman_pos, ghost_positions, pellets, a, last_move)
            q = self.get_q_value(f)
            if q > best_q:
                best_q = q
                best_acts = [a]
                best_feats = f
            elif math.isclose(q, best_q, rel_tol=1e-5):
                best_acts.append(a)

        chosen = random.choice(best_acts)
        if chosen != best_acts[0]:
            best_feats = self.extract_features(pacman_pos, ghost_positions, pellets, chosen, last_move)
        return chosen, best_feats

    def update(
        self,
        feats: Dict[str, float],
        reward: float,
        next_pacman_pos: Tuple[int, int],
        next_ghosts: List[Tuple[int, int]],
        next_pellets: set,
        next_legal_moves: List[str],
        next_last_move: str,
        terminal: bool,
    ):
        current_q = self.get_q_value(feats)
        if terminal or not next_legal_moves:
            target_q = reward
        else:
            max_next_q = max(
                self.get_q_value(
                    self.extract_features(
                        next_pacman_pos, next_ghosts, next_pellets, a, next_last_move
                    )
                )
                for a in next_legal_moves
            )
            target_q = reward + self.gamma * max_next_q

        td_error = target_q - current_q

        for k, f_val in feats.items():
            if f_val != 0.0:
                self.weights[k] += self.alpha * td_error * f_val


def train_q_learning(episodes: int = 1500, save_path: str = DEFAULT_SAVE_PATH) -> TrainableQAgent:
    walls = set()
    for y, row in enumerate(MAZE_LAYOUT):
        for x, char in enumerate(row):
            if char == "#":
                walls.add((x, y))

    def get_legal(x: int, y: int) -> List[str]:
        legal = []
        for d, (dx, dy) in DIRECTIONS.items():
            nx = (x + dx) % GRID_WIDTH
            ny = y + dy
            if (nx, ny) not in walls:
                legal.append(d)
        return legal

    agent = TrainableQAgent()

    print("======================================================================")
    print(" TRAINING APPROXIMATE Q-LEARNING (TD-LEARNING) AGENT")
    print("======================================================================")
    print(f"Episodes: {episodes} | Initial Epsilon: {agent.epsilon:.2f} | Alpha: {agent.alpha:.3f}\n")

    t_start = time.perf_counter()
    episode_scores = []
    episode_pellets = []

    for ep in range(1, episodes + 1):
        pellets = set()
        for y, row in enumerate(MAZE_LAYOUT):
            for x, char in enumerate(row):
                if char == ".":
                    pellets.add((x, y))

        pac_pos = list(START_POSITIONS["pacman"])
        if tuple(pac_pos) in pellets:
            pellets.remove(tuple(pac_pos))

        ghost_positions = [list(g) for g in START_POSITIONS["ghosts"]]
        ghost_dirs = ["up", "up", "up"]
        last_move = "left"

        score = 0
        pellets_eaten = 0

        for _ in range(400):
            legal = get_legal(pac_pos[0], pac_pos[1])
            if not legal:
                break

            action, chosen_feats = agent.select_action(
                tuple(pac_pos),
                [tuple(g) for g in ghost_positions],
                pellets,
                legal,
                last_move,
                explore=True,
            )

            dx, dy = DIRECTIONS[action]
            nx, ny = (pac_pos[0] + dx) % GRID_WIDTH, pac_pos[1] + dy
            old_pac = list(pac_pos)
            pac_pos = [nx, ny]

            ate_pellet = False
            if (nx, ny) in pellets:
                pellets.remove((nx, ny))
                ate_pellet = True
                pellets_eaten += 1

            old_ghosts = [list(g) for g in ghost_positions]
            for gi, gpos in enumerate(ghost_positions):
                glegal = get_legal(gpos[0], gpos[1])
                if glegal:
                    opp = OPPOSITE_DIRECTIONS.get(ghost_dirs[gi])
                    filtered = [m for m in glegal if m != opp] or glegal
                    if random.random() < 0.85:
                        gm = min(
                            filtered,
                            key=lambda m: abs((gpos[0] + DIRECTIONS[m][0]) % GRID_WIDTH - pac_pos[0])
                            + abs(gpos[1] + DIRECTIONS[m][1] - pac_pos[1]),
                        )
                    else:
                        gm = random.choice(filtered)
                    ghost_dirs[gi] = gm
                    ghost_positions[gi] = [
                        (gpos[0] + DIRECTIONS[gm][0]) % GRID_WIDTH,
                        gpos[1] + DIRECTIONS[gm][1],
                    ]

            # Collision check
            collided = False
            for gi in range(len(ghost_positions)):
                if pac_pos == ghost_positions[gi] or (
                    old_pac == ghost_positions[gi] and pac_pos == old_ghosts[gi]
                ):
                    collided = True
                    break

            won = len(pellets) == 0

            # Reward alignment
            reward = -0.5
            if ate_pellet:
                reward += 15.0
                score += 10
            if collided:
                reward -= 150.0
                score -= 200
            elif won:
                reward += 250.0
                score += 500

            next_legal = get_legal(pac_pos[0], pac_pos[1]) if not (collided or won) else []
            agent.update(
                chosen_feats,
                reward,
                tuple(pac_pos),
                [tuple(g) for g in ghost_positions],
                pellets,
                next_legal,
                action,
                collided or won,
            )

            last_move = action
            if collided or won:
                break

        agent.decay_parameters()
        episode_scores.append(score)
        episode_pellets.append(pellets_eaten)

        if ep % 200 == 0 or ep == episodes:
            avg_s = statistics.mean(episode_scores[-200:])
            avg_p = statistics.mean(episode_pellets[-200:])
            print(
                f"  Episode {ep:4d}/{episodes} | Epsilon: {agent.epsilon:.3f} | Alpha: {agent.alpha:.4f} | "
                f"Avg Score (last 200): {avg_s:6.1f} | Avg Pellets: {avg_p:4.1f}"
            )

    t_elapsed = time.perf_counter() - t_start
    print(f"\nTraining completed in {t_elapsed:.2f} seconds.")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(agent.weights, f, indent=2)
    print(f"[SUCCESS] Saved learned weights to {save_path}")

    # Also mirror to root if running in system_one
    root_mirror = "learned_q_weights.json"
    try:
        with open(root_mirror, "w") as f:
            json.dump(agent.weights, f, indent=2)
    except Exception:
        pass

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Approximate Q-Learning on Pac-Man")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save weights JSON")
    args = parser.parse_args()

    train_q_learning(episodes=args.episodes, save_path=args.save_path)
