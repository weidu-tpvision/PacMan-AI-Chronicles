"""
Direct Policy Search (Cross-Entropy Method / ES) for Pac-Man.
Directly optimizes linear policy weights over the shared ENHANCED topological features.

Seed hygiene: candidates are scored on TRAIN seeds, the elite is selected on VAL seeds,
and the tournament uses the disjoint TEST range (see core.seeds) - no selection leakage.
Saves best policy parameters to rl/weights/learned_enhanced_weights.json.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.features import ENHANCED_FEATURES, enhanced_features, linear_q
from agents.q_learning_agent import ENHANCED_DEFAULT_WEIGHTS
from core.environment import DEFAULT_MAX_STEPS, SCORE_DEATH, SCORE_PELLET, SCORE_WIN, Environment
from core.seeds import TRAIN_SEED_MIN, seed_everything, val_seeds

DEFAULT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "weights", "learned_enhanced_weights.json")

# Kept for backward compatibility with earlier imports
FEATURE_NAMES = ENHANCED_FEATURES
extract_features = enhanced_features


def evaluate_policy(weights: Dict[str, float], seeds: List[int], max_moves: int = DEFAULT_MAX_STEPS) -> float:
    """Mean tournament score of the greedy linear policy over `seeds` (deterministic per seed)."""
    scores = []
    for s in seeds:
        tie_rng = random.Random(s)
        env = Environment(seed=s)
        score = 0
        for _ in range(max_moves):
            legal = env.get_legal_moves(*env.pacman_pos)
            pac = tuple(env.pacman_pos)
            ghosts = [tuple(g) for g in env.ghost_positions]
            q_vals = {m: linear_q(weights, enhanced_features(pac, ghosts, env.pellets, m, env.last_move)) for m in legal}
            best_q = max(q_vals.values())
            move = tie_rng.choice([m for m in legal if q_vals[m] == best_q])

            col, ate, won = env.step(move)
            if ate:
                score += SCORE_PELLET
            if col:
                score += SCORE_DEATH
                break
            if won:
                score += SCORE_WIN
                break
        scores.append(score)
    return statistics.mean(scores)


def optimize_policy(
    generations: int = 35,
    population_size: int = 40,
    elite_ratio: float = 0.20,
    n_train_seeds: int = 30,
    n_val_seeds: int = 50,
    seed: int = 0,
    save_path: str = DEFAULT_SAVE_PATH,
    verbose: bool = True,
) -> Dict[str, float]:
    seed_everything(seed)
    rng = random.Random(seed)
    log = print if verbose else (lambda *a, **k: None)
    elite_size = max(2, int(population_size * elite_ratio))

    mu = dict(ENHANCED_DEFAULT_WEIGHTS)
    sigma = {k: max(1.0, abs(v) * 0.25) for k, v in mu.items()}
    # Relative sigma floor: 5% of the initial magnitude (absolute 1.0 floor swamped small weights)
    sigma_floor = {k: max(0.05, abs(v) * 0.05) for k, v in mu.items()}

    train_seed_pool = lambda gen: [TRAIN_SEED_MIN + gen * n_train_seeds + i for i in range(n_train_seeds)]
    vseeds = val_seeds(n_val_seeds)

    log("======================================================================")
    log(" DIRECT POLICY SEARCH (CROSS-ENTROPY METHOD) FOR PAC-MAN")
    log("======================================================================")
    log(f"Generations: {generations} | Population: {population_size} | Elites: {elite_size} | Seed: {seed}")
    log(f"TRAIN seeds: {n_train_seeds} fresh per generation from {TRAIN_SEED_MIN}+ | "
        f"VAL seeds: {vseeds[0]}..{vseeds[-1]}\n")

    best_global_score = -float("inf")
    best_global_weights = dict(mu)
    t0 = time.perf_counter()

    for gen in range(1, generations + 1):
        train_seeds = train_seed_pool(gen)
        candidates = []
        for _ in range(population_size):
            cand = {k: rng.gauss(mu[k], sigma[k]) for k in ENHANCED_FEATURES}
            candidates.append((evaluate_policy(cand, train_seeds), cand))

        candidates.sort(key=lambda x: x[0], reverse=True)
        elites = [cand for _, cand in candidates[:elite_size]]
        top_gen_score = candidates[0][0]

        for k in ENHANCED_FEATURES:
            vals = [e[k] for e in elites]
            mu[k] = statistics.mean(vals)
            sigma[k] = max(sigma_floor[k], statistics.stdev(vals))

        # Validate both the top elite and the (usually more robust) distribution mean
        val_score = -float("inf")
        for label, w in (("elite", elites[0]), ("mean", dict(mu))):
            sc = evaluate_policy(w, vseeds)
            if sc > val_score:
                val_score = sc
            if sc > best_global_score:
                best_global_score = sc
                best_global_weights = dict(w)

        if gen % 5 == 0 or gen == generations:
            log(
                f"  Gen {gen:2d}/{generations} | Train Top: {top_gen_score:6.1f} | "
                f"Val Score: {val_score:6.1f} | Best Val: {best_global_score:6.1f} | "
                f"{time.perf_counter() - t0:5.0f}s"
            )

    log(f"\nOptimization completed in {time.perf_counter() - t0:.2f} seconds.")
    log(f"Peak Validation Score (selection metric, optimistic): {best_global_score:.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 4) for k, v in best_global_weights.items()}, f, indent=2)
    log(f"\n[SUCCESS] Saved best weights to {save_path}")

    return best_global_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Pac-Man Policy via Cross-Entropy Method")
    parser.add_argument("--generations", type=int, default=35, help="Number of generations")
    parser.add_argument("--population", type=int, default=40, help="Population size")
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save weights JSON")
    args = parser.parse_args()

    optimize_policy(
        generations=args.generations, population_size=args.population, seed=args.seed, save_path=args.save_path
    )
