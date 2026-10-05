"""
PacMan-AI-Chronicles: Multi-Agent Tournament Benchmark.
Pits 40 years of artificial intelligence decision paradigms against each other:
1. Random Selection (Noise floor / Lower bound)
2. Greedy Heuristic (Expert Symbolic Rules)
3. Q-Learning (Textbook TD-Learning)
4. Deep Q-Network (PyTorch Convolutional DQN)
5. Policy-Optimized RL (Evolutionary Cross-Entropy Search)
6. System 1 (Ollama / Jev-style Foundation Model)
"""


import argparse
import random
import statistics
import time
from typing import Dict, List, Tuple

from agents import (
    DQNAgent,
    GreedyHeuristicAgent,
    PretrainedQLearningAgent,
    RandomAgent,
    SystemOneBaselineAgent,
    TrainedQLearningAgent,
)
from core.environment import Environment
from core.maze_data import DIRECTIONS, GRID_HEIGHT, GRID_WIDTH, START_POSITIONS
from llm.decision_client import SystemOneAgent



def run_episode(agent, seed: int, max_moves: int = 150) -> Dict:
    if hasattr(agent, "reset"):
        agent.reset()
    env = Environment(seed=seed)
    moves_count = 0
    pellets_eaten = 0
    blunders = 0
    latencies = []

    for _ in range(max_moves):
        legal = env.get_legal_moves(env.pacman_pos[0], env.pacman_pos[1])
        if not legal:
            break

        safe_moves = []
        for m in legal:
            dx, dy = DIRECTIONS[m]
            nx = (env.pacman_pos[0] + dx) % GRID_WIDTH
            ny = env.pacman_pos[1] + dy
            min_g = min(
                min(abs(nx - gx), GRID_WIDTH - abs(nx - gx)) + abs(ny - gy)
                for gx, gy in env.ghost_positions
            )
            if min_g > 1:
                safe_moves.append(m)

        res = agent.decide(
            tuple(env.pacman_pos),
            [tuple(g) for g in env.ghost_positions],
            env.pellets,
            legal,
            env.last_move,
        )
        choice = res.choice
        latencies.append(res.latency_ms)

        dx, dy = DIRECTIONS[choice]
        nx = (env.pacman_pos[0] + dx) % GRID_WIDTH
        ny = env.pacman_pos[1] + dy
        chosen_min_g = min(
            min(abs(nx - gx), GRID_WIDTH - abs(nx - gx)) + abs(ny - gy)
            for gx, gy in env.ghost_positions
        )
        if chosen_min_g <= 1 and len(safe_moves) > 0:
            blunders += 1

        moves_count += 1
        collided, ate, won = env.step(choice)
        if ate:
            pellets_eaten += 1

        if collided or won:
            break

    score = (pellets_eaten * 10) - (200 if collided else 0)
    blunder_rate = (blunders / moves_count * 100.0) if moves_count > 0 else 0.0

    return {
        "moves": moves_count,
        "pellets": pellets_eaten,
        "score": score,
        "blunder_rate": blunder_rate,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "survived": not collided,
    }


def run_tournament(episodes: int = 100, max_moves: int = 120, model: str = "nimble", host: str = "http://localhost:11434"):
    print("==========================================================================================")
    print(f" PACMAN-AI-CHRONICLES: Tournament Benchmark ({episodes} Seeded Episodes Each)")
    print("==========================================================================================")


    sys1_backend = SystemOneAgent(model=model, host=host, prefer_live=True)
    is_live = sys1_backend.is_ollama_online()
    mode_label = f"Live Ollama ({model})" if is_live else "Heuristic Simulator"

    agents = [
        RandomAgent("Random Agent (Baseline)"),
        GreedyHeuristicAgent("Greedy Heuristic"),
        PretrainedQLearningAgent("Q-Learning (Textbook Baseline)"),
        DQNAgent(name="Deep Q-Network (PyTorch DQN)"),
        TrainedQLearningAgent(name="RL (Policy Optimized)"),
        SystemOneBaselineAgent(sys1_backend, f"System 1 [{mode_label}]"),
    ]

    results = {agent.name: [] for agent in agents}

    print("Running tournament across identical seeds...")
    for ep in range(episodes):
        seed = 1000 + ep
        for agent in agents:
            stats = run_episode(agent, seed=seed, max_moves=max_moves)
            results[agent.name].append(stats)
        if (ep + 1) % 20 == 0 or ep + 1 == episodes:
            print(f"  * Completed seed {ep+1}/{episodes}")

    print("\n==========================================================================================")
    print(f" {'Agent Name':<34} | {'Avg Score':<10} | {'Avg Moves':<10} | {'Pellets':<8} | {'Blunder %':<10} | {'Latency':<9}")
    print("------------------------------------------------------------------------------------------")

    summary = []
    for agent in agents:
        ep_stats = results[agent.name]
        avg_score = statistics.mean(s["score"] for s in ep_stats)
        avg_moves = statistics.mean(s["moves"] for s in ep_stats)
        avg_pellets = statistics.mean(s["pellets"] for s in ep_stats)
        avg_blunder = statistics.mean(s["blunder_rate"] for s in ep_stats)
        avg_lat = statistics.mean(s["avg_latency"] for s in ep_stats)

        summary.append({
            "name": agent.name,
            "score": avg_score,
            "moves": avg_moves,
            "pellets": avg_pellets,
            "blunder": avg_blunder,
            "latency": avg_lat,
        })

        lat_str = f"{avg_lat:5.2f} ms" if avg_lat < 10 else f"{avg_lat:5.1f} ms"
        print(f" {agent.name:<34} | {avg_score:9.1f}  | {avg_moves:9.1f}  | {avg_pellets:7.1f}  | {avg_blunder:8.1f}%  | {lat_str:<9}")

    print("==========================================================================================\n")

    summary_map = {s["name"]: s for s in summary}
    opt_rl = summary_map.get("RL (Policy Optimized)", summary[0])
    dqn_rl = summary_map.get("Deep Q-Network (PyTorch DQN)", summary[0])
    pre_rl = summary_map.get("Q-Learning (Textbook Baseline)", summary[0])
    greedy = summary_map.get("Greedy Heuristic", summary[0])

    print("HEAD-TO-HEAD COMPARISON:")
    diff_vs_pre = opt_rl["score"] - pre_rl["score"]
    diff_vs_greedy = opt_rl["score"] - greedy["score"]
    diff_vs_dqn = opt_rl["score"] - dqn_rl["score"]

    print(f"[+] Policy-Optimized RL Score: {opt_rl['score']:.1f} ({opt_rl['pellets']:.1f} pellets)")
    print(f"[+] Deep Q-Network (DQN) Score: {dqn_rl['score']:.1f} ({dqn_rl['pellets']:.1f} pellets, {dqn_rl['latency']:.2f} ms)")
    print(f"[+] Policy-RL vs. DQN:          +{diff_vs_dqn:.1f} points")
    print(f"[+] Policy-RL vs. Textbook RL:  +{diff_vs_pre:.1f} points")
    print(f"[+] Policy-RL vs. Greedy:       +{diff_vs_greedy:.1f} points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pac-Man Agent Tournament Comparison")
    parser.add_argument("--episodes", type=int, default=100, help="Number of seeded episodes to test")
    parser.add_argument("--max-moves", type=int, default=120, help="Maximum moves per episode")
    parser.add_argument("--model", type=str, default="nimble", help="Ollama model name")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL")
    args = parser.parse_args()

    run_tournament(
        episodes=args.episodes,
        max_moves=args.max_moves,
        model=args.model,
        host=args.host,
    )
