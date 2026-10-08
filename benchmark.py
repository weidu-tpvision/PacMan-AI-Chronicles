"""
PacMan-AI-Chronicles: Latency & Throughput Benchmark Runner.
Evaluates decision latency, throughput (decisions/sec), P50/P95 latency,
and game survival metrics for the System 1 agent (Ollama or offline simulator).

Runs on the shared core.environment.Environment (same dynamics as the tournament),
with life-loss handled by Environment.respawn().
"""

import argparse
import statistics
import time
from typing import List

from core.environment import Environment
from core.seeds import TEST_SEED_BASE, seed_everything
from llm.decision_client import SystemOneAgent


def run_benchmark(
    num_moves: int = 50,
    model: str = "nimble",
    host: str = "http://localhost:11434",
    force_mock: bool = False,
    seed: int = TEST_SEED_BASE,
):
    print("================================================================")
    print(f" PacMan-AI-Chronicles: Decision Benchmark ({model})")
    print("================================================================")

    # Measure real compute time offline - never report synthetic latency in a latency benchmark.
    agent = SystemOneAgent(model=model, host=host, prefer_live=not force_mock, simulate_latency=False)
    is_live = (not force_mock) and agent.is_ollama_online(force_refresh=True)
    mode_str = f"Live Ollama ({model} @ {host})" if is_live else "Offline heuristic simulator (measured compute time)"
    print(f"Execution Mode: {mode_str}\n")

    seed_everything(seed)
    env = Environment(seed=seed)
    latencies: List[float] = []
    confidences: List[float] = []
    pellets_eaten = collisions = live_decisions = 0

    print(f"Running simulation for up to {num_moves} decisions...")
    start_total_time = time.perf_counter()

    for step in range(1, num_moves + 1):
        legal = env.get_legal_moves(*env.pacman_pos)
        res = agent.decide_move(
            tuple(env.pacman_pos), [tuple(g) for g in env.ghost_positions], env.pellets, legal, env.last_move
        )
        latencies.append(res.latency_ms)
        confidences.append(res.confidence)
        live_decisions += int(res.is_live)

        collided, ate, won = env.step(res.choice)
        pellets_eaten += int(ate)
        if collided:
            collisions += 1
            env.respawn()
        if won:
            print(f"  Board cleared at step {step}!")
            break

        if step % 10 == 0 or step == num_moves:
            print(f"  Step {step:3d}/{num_moves} | Choice: {res.choice:5} | Latency: {res.latency_ms:7.2f}ms | Conf: {res.confidence:.2f}")

    total_wall_sec = time.perf_counter() - start_total_time
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    print("\n================================================================")
    print(" BENCHMARK RESULTS")
    print("================================================================")
    print(f"Completed Decisions:    {len(latencies)} ({live_decisions} live model calls)")
    print(f"Total Wall-clock Time:  {total_wall_sec:.2f} s")
    print(f"Throughput:             {len(latencies) / total_wall_sec:.1f} decisions/sec")
    print("--- Latency Metrics ---")
    print(f"Mean Latency:           {statistics.mean(latencies):.2f} ms")
    print(f"Median (P50) Latency:   {statistics.median(latencies):.2f} ms")
    print(f"P95 Latency:            {p95_lat:.2f} ms")
    print(f"Min / Max Latency:      {min(latencies):.2f} ms / {max(latencies):.2f} ms")
    print("--- Quality & Game Metrics ---")
    print(f"Mean Model Confidence:  {statistics.mean(confidences):.3f}")
    print(f"Pellets Eaten:          {pellets_eaten}")
    print(f"Ghost Collisions:       {collisions}")
    print("================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System 1 Decision Benchmark")
    parser.add_argument("--moves", type=int, default=50, help="Number of decisions to simulate")
    parser.add_argument("--model", type=str, default="nimble", help="Ollama model name")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--mock", action="store_true", help="Force heuristic simulation mode")
    parser.add_argument("--seed", type=int, default=TEST_SEED_BASE, help="Environment seed")
    args = parser.parse_args()

    run_benchmark(num_moves=args.moves, model=args.model, host=args.host, force_mock=args.mock, seed=args.seed)
