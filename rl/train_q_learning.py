"""
Approximate Q-Learning Training for Pac-Man with Linear Function Approximation.
Updates feature weights using Temporal Difference (TD) error:
w_i <- w_i + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a)) * f_i(s, a)

Runs on the shared core.environment.Environment and the shared textbook feature
extractor (agents.features), so training dynamics match the tournament exactly.
Saves learned parameters to rl/weights/learned_q_weights.json.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.features import TEXTBOOK_FEATURES, linear_q, textbook_features
from agents.q_learning_agent import TEXTBOOK_DEFAULT_WEIGHTS
from core.environment import DEFAULT_MAX_STEPS, SCORE_DEATH, SCORE_PELLET, SCORE_WIN, Environment
from core.seeds import seed_everything, train_seed

DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "weights", "learned_q_weights.json")

# Reward shaping (trainer-local; independent of the DQN trainer's score-aligned reward)
R_STEP = -0.5
R_PELLET = 15.0
R_DEATH = -150.0
R_WIN = 250.0


class TrainableQAgent:
    def __init__(
        self,
        alpha: float = 0.01,
        alpha_min: float = 0.001,
        alpha_decay: float = 0.998,
        gamma: float = 0.88,
        epsilon: float = 0.95,
        epsilon_decay: float = 0.996,
        epsilon_min: float = 0.01,
        rng: Optional[random.Random] = None,
    ):
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = rng or random.Random()
        self.weights: Dict[str, float] = dict(TEXTBOOK_DEFAULT_WEIGHTS)

    def decay_parameters(self):
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def extract_features(self, pacman_pos, ghost_positions, pellets, action, last_move=None) -> Dict[str, float]:
        return textbook_features(pacman_pos, ghost_positions, pellets, action, last_move)

    def get_q_value(self, feats: Dict[str, float]) -> float:
        return linear_q(self.weights, feats)

    def select_action(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
        explore: bool = True,
    ) -> Tuple[str, Dict[str, float]]:
        if explore and self.rng.random() < self.epsilon:
            a = self.rng.choice(legal_moves)
            return a, self.extract_features(pacman_pos, ghost_positions, pellets, a, last_move)

        scored = []
        for a in legal_moves:
            f = self.extract_features(pacman_pos, ghost_positions, pellets, a, last_move)
            scored.append((self.get_q_value(f), a, f))
        best_q = max(q for q, _, _ in scored)
        best = [(a, f) for q, a, f in scored if math.isclose(q, best_q, rel_tol=1e-9, abs_tol=1e-9)]
        return self.rng.choice(best)

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
    ) -> float:
        current_q = self.get_q_value(feats)
        if terminal or not next_legal_moves:
            target_q = reward
        else:
            max_next_q = max(
                self.get_q_value(self.extract_features(next_pacman_pos, next_ghosts, next_pellets, a, next_last_move))
                for a in next_legal_moves
            )
            target_q = reward + self.gamma * max_next_q

        td_error = target_q - current_q
        for k, f_val in feats.items():
            if f_val != 0.0:
                self.weights[k] += self.alpha * td_error * f_val
        return td_error


def train_q_learning(
    episodes: int = 1500,
    save_path: str = DEFAULT_SAVE_PATH,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
    verbose: bool = True,
) -> TrainableQAgent:
    seed_everything(seed)
    rng = random.Random(seed)
    agent = TrainableQAgent(rng=rng)
    log = print if verbose else (lambda *a, **k: None)

    log("======================================================================")
    log(" TRAINING APPROXIMATE Q-LEARNING (TD-LEARNING) AGENT")
    log("======================================================================")
    log(f"Episodes: {episodes} | Horizon: {max_steps} | Seed: {seed} | "
        f"Initial Epsilon: {agent.epsilon:.2f} | Alpha: {agent.alpha:.3f}\n")

    t_start = time.perf_counter()
    episode_scores: List[int] = []
    episode_pellets: List[int] = []

    for ep in range(1, episodes + 1):
        env = Environment(seed=train_seed(rng))
        score = pellets_eaten = 0

        for _ in range(max_steps):
            legal = env.get_legal_moves(*env.pacman_pos)
            last_move = env.last_move
            action, chosen_feats = agent.select_action(
                tuple(env.pacman_pos), [tuple(g) for g in env.ghost_positions], env.pellets, legal, last_move
            )
            collided, ate, won = env.step(action)

            reward = R_STEP
            if ate:
                reward += R_PELLET
                score += SCORE_PELLET
                pellets_eaten += 1
            if collided:
                reward += R_DEATH
                score += SCORE_DEATH
            elif won:
                reward += R_WIN
                score += SCORE_WIN

            terminal = collided or won
            next_legal = [] if terminal else env.get_legal_moves(*env.pacman_pos)
            agent.update(
                chosen_feats,
                reward,
                tuple(env.pacman_pos),
                [tuple(g) for g in env.ghost_positions],
                env.pellets,
                next_legal,
                env.last_move,
                terminal,
            )
            if terminal:
                break

        agent.decay_parameters()
        episode_scores.append(score)
        episode_pellets.append(pellets_eaten)

        if ep % 200 == 0 or ep == episodes:
            log(
                f"  Episode {ep:4d}/{episodes} | Epsilon: {agent.epsilon:.3f} | Alpha: {agent.alpha:.4f} | "
                f"Avg Score (last 200): {statistics.mean(episode_scores[-200:]):6.1f} | "
                f"Avg Pellets: {statistics.mean(episode_pellets[-200:]):4.1f}"
            )

    log(f"\nTraining completed in {time.perf_counter() - t_start:.2f} seconds.")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: round(agent.weights[k], 4) for k in TEXTBOOK_FEATURES}, f, indent=2)
    log(f"[SUCCESS] Saved learned weights to {save_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Approximate Q-Learning on Pac-Man")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save weights JSON")
    args = parser.parse_args()

    train_q_learning(episodes=args.episodes, save_path=args.save_path, seed=args.seed)
