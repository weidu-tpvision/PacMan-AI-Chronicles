"""
PacMan-AI-Chronicles: Multi-Agent Tournament Benchmark.
Pits 40 years of artificial intelligence decision paradigms against each other:
1. Random Selection (Noise floor / Lower bound)
2. Greedy Heuristic (Expert Symbolic Rules)
3. Q-Learning (Textbook TD-Learning)
4. Deep Q-Network (PyTorch Convolutional DQN) - raw network AND network + inference heuristics
5. Policy-Optimized RL (Evolutionary Cross-Entropy Search)
6. System 1 (Ollama / Jev-style Foundation Model, or offline simulator)

Fairness guarantees:
- Every agent plays the identical TEST seeds (core.seeds, disjoint from TRAIN/VAL).
- Global RNGs (random / numpy / torch) are re-seeded per (episode) so tie-breaking is reproducible.
- Simulated (synthetic) latencies are flagged with '*'.
"""

import argparse
import json
import math
import os
import statistics
from typing import Dict, List

from agents import (
    DQNAgent,
    GreedyHeuristicAgent,
    PretrainedQLearningAgent,
    RandomAgent,
    SystemOneBaselineAgent,
    TrainedQLearningAgent,
)
from agents.features import min_ghost_bfs, next_cell
from core.environment import DEFAULT_MAX_STEPS, SCORE_DEATH, SCORE_PELLET, SCORE_WIN, Environment
from core.maze_data import DIRECTIONS
from core.seeds import seed_everything, test_seeds
from llm.decision_client import SystemOneAgent

DEFAULT_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tournament_results.json")


def run_episode(agent, seed: int, max_moves: int = DEFAULT_MAX_STEPS) -> Dict:
    """Play one seeded episode. Score = 10/pellet - 200 on death + 500 on clearing the board."""
    seed_everything(seed)
    if hasattr(agent, "reset"):
        agent.reset()
    env = Environment(seed=seed)
    moves_count = pellets_eaten = blunders = errors = invalid_actions = 0
    collided = won = False
    latencies: List[float] = []
    simulated = False

    for _ in range(max_moves):
        legal = env.get_legal_moves(*env.pacman_pos)
        pac = tuple(env.pacman_pos)
        ghosts = [tuple(g) for g in env.ghost_positions]

        safe_moves = [m for m in legal if min_ghost_bfs(next_cell(pac, m), ghosts) > 1]

        res = agent.decide(pac, ghosts, env.pellets, legal, env.last_move)
        latencies.append(res.latency_ms)
        simulated = simulated or res.latency_simulated
        if res.error_msg:
            errors += 1

        if res.choice in legal and min_ghost_bfs(next_cell(pac, res.choice), ghosts) <= 1 and safe_moves:
            blunders += 1

        move = res.choice
        if not isinstance(move, str) or move not in legal:
            invalid_actions += 1
            if not res.error_msg:
                errors += 1
            # Preserve Environment's counted no-op behavior for known blocked directions.
            # Unknown values have no direction vector, so use a stable legal fallback.
            if not isinstance(move, str) or move not in DIRECTIONS:
                move = legal[0] if legal else "left"

        moves_count += 1
        collided, ate, won = env.step(move)
        if ate:
            pellets_eaten += 1
        if collided or won:
            break

    score = pellets_eaten * SCORE_PELLET + (SCORE_DEATH if collided else 0) + (SCORE_WIN if won else 0)
    return {
        "seed": seed,
        "moves": moves_count,
        "pellets": pellets_eaten,
        "score": score,
        "won": won,
        "survived": not collided,
        "blunder_rate": (blunders / moves_count * 100.0) if moves_count else 0.0,
        "illegal_moves": env.illegal_moves + invalid_actions,
        "invalid_actions": invalid_actions,
        "errors": errors,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "latency_simulated": simulated,
    }


def _ci95(values: List[float]) -> float:
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def build_agents(model: str, host: str, sys1_live: bool):
    sys1_backend = SystemOneAgent(model=model, host=host, prefer_live=sys1_live)
    is_live = sys1_live and sys1_backend.is_ollama_online(force_refresh=True)
    mode_label = f"Live {model}" if is_live else "Simulator"
    return [
        RandomAgent("Random Agent (Baseline)"),
        GreedyHeuristicAgent("Greedy Heuristic"),
        PretrainedQLearningAgent("Q-Learning (Textbook TD)"),
        DQNAgent(name="DQN (raw network)", heuristics=False, require_weights=True),
        DQNAgent(name="DQN (+inference heuristics)", heuristics=True, require_weights=True),
        TrainedQLearningAgent(name="RL (Policy Optimized, CEM)"),
        SystemOneBaselineAgent(sys1_backend, f"System 1 [{mode_label}]"),
    ]


def run_tournament(
    episodes: int = 100,
    max_moves: int = DEFAULT_MAX_STEPS,
    model: str = "nimble",
    host: str = "http://localhost:11434",
    sys1_live: bool = True,
    results_path: str = DEFAULT_RESULTS_PATH,
):
    width = 118
    print("=" * width)
    print(f" PACMAN-AI-CHRONICLES: Tournament Benchmark ({episodes} TEST seeds, horizon {max_moves} moves)")
    print("=" * width)

    agents = build_agents(model, host, sys1_live)
    seeds = test_seeds(episodes)
    results = {agent.name: [] for agent in agents}

    print(f"Running tournament across identical seeds {seeds[0]}..{seeds[-1]}...")
    for i, seed in enumerate(seeds):
        for agent in agents:
            results[agent.name].append(run_episode(agent, seed=seed, max_moves=max_moves))
        if (i + 1) % 20 == 0 or i + 1 == episodes:
            print(f"  * Completed seed {i+1}/{episodes}")

    print("\n" + "=" * width)
    print(
        f" {'Agent Name':<30} | {'Score (±95% CI)':<17} | {'Moves':>6} | {'Pellets':>7} | "
        f"{'Survive':>7} | {'Wins':>4} | {'Blunder':>7} | {'Illegal':>7} | {'Latency':>10}"
    )
    print("-" * width)

    summary = []
    any_simulated = False
    for agent in agents:
        ep = results[agent.name]
        scores = [s["score"] for s in ep]
        row = {
            "name": agent.name,
            "score": statistics.mean(scores),
            "score_ci95": _ci95(scores),
            "moves": statistics.mean(s["moves"] for s in ep),
            "pellets": statistics.mean(s["pellets"] for s in ep),
            "survival_rate": 100.0 * sum(s["survived"] for s in ep) / len(ep),
            "wins": sum(s["won"] for s in ep),
            "blunder": statistics.mean(s["blunder_rate"] for s in ep),
            "illegal_moves": sum(s["illegal_moves"] for s in ep),
            "errors": sum(s["errors"] for s in ep),
            "latency": statistics.mean(s["avg_latency"] for s in ep),
            "latency_simulated": any(s["latency_simulated"] for s in ep),
        }
        summary.append(row)
        any_simulated = any_simulated or row["latency_simulated"]

        lat = f"{row['latency']:.2f} ms" if row["latency"] < 10 else f"{row['latency']:.1f} ms"
        lat += "*" if row["latency_simulated"] else " "
        score_str = f"{row['score']:7.1f} ± {row['score_ci95']:5.1f}"
        print(
            f" {agent.name:<30} | {score_str:<17} | {row['moves']:6.1f} | {row['pellets']:7.1f} | "
            f"{row['survival_rate']:6.1f}% | {row['wins']:4d} | {row['blunder']:6.1f}% | "
            f"{row['illegal_moves']:7d} | {lat:>10}"
        )

    print("=" * width)
    if any_simulated:
        print(" * Synthetic latency from the offline System 1 simulator (not a measured model round-trip).")

    ranked = sorted(summary, key=lambda r: r["score"], reverse=True)
    print("\nRANKING:")
    for i, r in enumerate(ranked, 1):
        delta = r["score"] - ranked[0]["score"]
        print(f"  {i}. {r['name']:<30} {r['score']:7.1f}  ({delta:+.1f} vs. leader)")

    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                {"episodes": episodes, "max_moves": max_moves, "seeds": [seeds[0], seeds[-1]],
                 "summary": summary, "per_episode": results},
                f, indent=2,
            )
        print(f"\n[RESULTS] Saved to {results_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pac-Man Agent Tournament Comparison")
    parser.add_argument("--episodes", type=int, default=100, help="Number of seeded TEST episodes")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_STEPS, help="Maximum moves per episode")
    parser.add_argument("--model", type=str, default="nimble", help="Ollama model name")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--offline", action="store_true", help="Force the offline System 1 simulator")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS_PATH, help="JSON output path ('' to skip)")
    args = parser.parse_args()

    run_tournament(
        episodes=args.episodes,
        max_moves=args.max_moves,
        model=args.model,
        host=args.host,
        sys1_live=not args.offline,
        results_path=args.results,
    )
