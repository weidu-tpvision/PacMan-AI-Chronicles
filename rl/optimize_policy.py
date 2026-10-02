"""
Direct Policy Search (Cross-Entropy Method / ES) for Pac-Man.
Directly optimizes policy weights to maximize game score and beat the greedy heuristic.
Saves best policy parameters to rl/weights/learned_enhanced_weights.json.
"""

import argparse
import json
import math
import os
import random
import statistics
import time
from typing import Dict, List, Tuple

from core.environment import Environment
from core.maze_data import (
    DIRECTIONS,
    GRID_WIDTH,
    MAZE_DEAD_ENDS,
    MAZE_DIST_MATRIX,
    MAZE_JUNCTIONS,
    OPPOSITE_DIRECTIONS,
)

DEFAULT_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "learned_enhanced_weights.json"
)

FEATURE_NAMES = [
    "ghost_1_step",
    "ghost_2_step",
    "ghost_3_step",
    "ghost_safe_dist",
    "dead_end_trap",
    "safe_junction",
    "eats_pellet",
    "poisoned_pellet",
    "nearest_pellet_dist",
    "reverse_penalty",
]


def extract_features(
    pacman_pos: Tuple[int, int],
    ghost_positions: List[Tuple[int, int]],
    pellets: set,
    action: str,
    last_move: str = None,
) -> Dict[str, float]:
    px, py = pacman_pos
    dx, dy = DIRECTIONS[action]
    nx = (px + dx) % GRID_WIDTH
    ny = py + dy

    ghost_bfs_dists = [
        MAZE_DIST_MATRIX.get(((nx, ny), tuple(g)), 99) for g in ghost_positions
    ]
    min_ghost_bfs = min(ghost_bfs_dists) if ghost_bfs_dists else 99

    is_dead_end = (nx, ny) in MAZE_DEAD_ENDS
    depth = MAZE_DEAD_ENDS.get((nx, ny), 0)
    is_dead_end_trap = 1.0 if (is_dead_end and min_ghost_bfs <= depth + 3) else 0.0

    is_safe_junction = 1.0 if ((nx, ny) in MAZE_JUNCTIONS and min_ghost_bfs > 2) else 0.0
    eats_pellet = 1.0 if (nx, ny) in pellets else 0.0
    poisoned_pellet = 1.0 if (eats_pellet and (min_ghost_bfs <= 2 or is_dead_end_trap)) else 0.0

    nearest_pellet_dist = 0.0
    if pellets:
        sample_p = list(pellets)[:35]
        nearest_pellet_dist = float(
            min(MAZE_DIST_MATRIX.get(((nx, ny), p), 99) for p in sample_p)
        )

    rev_pen = (
        1.0
        if (last_move and action == OPPOSITE_DIRECTIONS.get(last_move) and min_ghost_bfs > 3)
        else 0.0
    )

    return {
        "ghost_1_step": 1.0 if min_ghost_bfs <= 1 else 0.0,
        "ghost_2_step": 1.0 if min_ghost_bfs == 2 else 0.0,
        "ghost_3_step": 1.0 if min_ghost_bfs == 3 else 0.0,
        "ghost_safe_dist": float(min_ghost_bfs) if min_ghost_bfs > 3 else 0.0,
        "dead_end_trap": is_dead_end_trap,
        "safe_junction": is_safe_junction,
        "eats_pellet": eats_pellet,
        "poisoned_pellet": poisoned_pellet,
        "nearest_pellet_dist": nearest_pellet_dist,
        "reverse_penalty": rev_pen,
    }


def evaluate_policy(weights: Dict[str, float], seeds: List[int], max_moves: int = 350) -> float:
    scores = []
    for s in seeds:
        env = Environment(seed=s)
        score = 0
        pellets_eaten = 0
        last_move = "left"

        for _ in range(max_moves):
            legal = env.get_legal_moves(env.pacman_pos[0], env.pacman_pos[1])
            if not legal:
                break

            q_vals = {}
            for m in legal:
                feats = extract_features(
                    tuple(env.pacman_pos),
                    [tuple(g) for g in env.ghost_positions],
                    env.pellets,
                    m,
                    last_move,
                )
                q_vals[m] = sum(weights.get(k, 0.0) * feats.get(k, 0.0) for k in weights)

            best_q = max(q_vals.values())
            best_moves = [m for m in legal if q_vals[m] == best_q]
            move = random.choice(best_moves)

            col, ate, won = env.step(move)
            if ate:
                score += 10
                pellets_eaten += 1
            if col:
                score -= 200
                break
            if won:
                score += 500
                break
            last_move = move

        scores.append(score)
    return statistics.mean(scores)


def optimize_policy(
    generations: int = 35,
    population_size: int = 40,
    elite_ratio: float = 0.20,
    save_path: str = DEFAULT_SAVE_PATH,
) -> Dict[str, float]:
    elite_size = max(2, int(population_size * elite_ratio))

    mu = {
        "ghost_1_step": -450.0,
        "ghost_2_step": -140.0,
        "ghost_3_step": -20.0,
        "ghost_safe_dist": 3.0,
        "dead_end_trap": -280.0,
        "safe_junction": 20.0,
        "eats_pellet": 70.0,
        "poisoned_pellet": -240.0,
        "nearest_pellet_dist": -1.2,
        "reverse_penalty": -30.0,
    }
    sigma = {k: max(5.0, abs(v) * 0.25) for k, v in mu.items()}

    train_seeds = [100 + i for i in range(15)]
    val_seeds = [1000 + i for i in range(25)]

    print("======================================================================")
    print(" DIRECT POLICY SEARCH (CROSS-ENTROPY METHOD) FOR PAC-MAN")
    print("======================================================================")
    print(f"Generations: {generations} | Population: {population_size} | Elites: {elite_size}\n")

    best_global_score = -999999.0
    best_global_weights = dict(mu)
    t0 = time.perf_counter()

    for gen in range(1, generations + 1):
        candidates = []
        for _ in range(population_size):
            cand = {k: random.gauss(mu[k], sigma[k]) for k in FEATURE_NAMES}
            score = evaluate_policy(cand, train_seeds)
            candidates.append((score, cand))

        candidates.sort(key=lambda x: x[0], reverse=True)
        elites = [cand for _, cand in candidates[:elite_size]]
        top_gen_score = candidates[0][0]

        for k in FEATURE_NAMES:
            vals = [e[k] for e in elites]
            mu[k] = statistics.mean(vals)
            sigma[k] = max(1.0, statistics.stdev(vals) if len(vals) > 1 else sigma[k] * 0.9)

        val_score = evaluate_policy(elites[0], val_seeds)
        if val_score > best_global_score:
            best_global_score = val_score
            best_global_weights = dict(elites[0])

        if gen % 5 == 0 or gen == generations:
            print(
                f"  Gen {gen:2d}/{generations} | Train Top: {top_gen_score:5.1f} | "
                f"Val Score: {val_score:5.1f} | Best Val: {best_global_score:5.1f}"
            )

    t_elapsed = time.perf_counter() - t0
    print(f"\nOptimization completed in {t_elapsed:.2f} seconds.")
    print(f"Peak Validation Score: {best_global_score:.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(best_global_weights, f, indent=2)
    print(f"\n[SUCCESS] Saved best weights to {save_path}")

    return best_global_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Pac-Man Policy via Cross-Entropy Method")
    parser.add_argument("--generations", type=int, default=35, help="Number of generations")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save weights JSON")
    args = parser.parse_args()

    optimize_policy(generations=args.generations, save_path=args.save_path)
