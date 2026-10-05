"""
PacMan-AI-Chronicles: Latency & Throughput Benchmark Runner.
Evaluates decision latency, throughput (decisions/sec), P50/P95 latency,
and game survival metrics against Ollama or heuristic simulator.
"""

import argparse
import random
import statistics
import time
from typing import List, Tuple

from llm.decision_client import SystemOneAgent
from core.maze_data import (
    DIRECTIONS,
    GRID_HEIGHT,
    GRID_WIDTH,
    MAZE_LAYOUT,
    OPPOSITE_DIRECTIONS,
    START_POSITIONS,
)


def run_benchmark(
    num_moves: int = 50,
    model: str = "nimble",
    host: str = "http://localhost:11434",
    force_mock: bool = False,
):
    print("================================================================")
    print(f" PacMan-AI-Chronicles: Decision Benchmark ({model})")
    print("================================================================")


    agent = SystemOneAgent(model=model, host=host, prefer_live=(not force_mock))
    is_live = agent.is_ollama_online(force_refresh=True) if not force_mock else False
    mode_str = f"Live Ollama ({model} @ {host})" if is_live else "Heuristic Simulator"
    print(f"Execution Mode: {mode_str}\n")

    # Set up board
    walls = set()
    pellets = set()
    for y, row in enumerate(MAZE_LAYOUT):
        for x, char in enumerate(row):
            if char == "#":
                walls.add((x, y))
            elif char == ".":
                pellets.add((x, y))

    pacman_pos = list(START_POSITIONS["pacman"])
    if tuple(pacman_pos) in pellets:
        pellets.remove(tuple(pacman_pos))

    ghost_positions = [list(g) for g in START_POSITIONS["ghosts"]]
    ghost_dirs = ["up", "up", "up"]
    last_move = "left"

    def get_legal_moves(x: int, y: int) -> List[str]:
        legal = []
        for d, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if nx < 0:
                nx = GRID_WIDTH - 1
            elif nx >= GRID_WIDTH:
                nx = 0
            if (nx, ny) not in walls:
                legal.append(d)
        return legal

    latencies: List[float] = []
    confidences: List[float] = []
    pellets_eaten = 0
    collisions = 0

    print(f"Running simulation for up to {num_moves} decisions...")
    start_total_time = time.perf_counter()

    for step in range(1, num_moves + 1):
        legal = get_legal_moves(pacman_pos[0], pacman_pos[1])
        if not legal:
            break

        res = agent.decide_move(
            tuple(pacman_pos),
            [tuple(g) for g in ghost_positions],
            pellets,
            legal,
            last_move,
        )

        latencies.append(res.latency_ms)
        confidences.append(res.confidence)
        chosen = res.choice
        last_move = chosen

        # Advance Pac-Man
        dx, dy = DIRECTIONS[chosen]
        old_pac = list(pacman_pos)
        nx, ny = old_pac[0] + dx, old_pac[1] + dy
        if nx < 0:
            nx = GRID_WIDTH - 1
        elif nx >= GRID_WIDTH:
            nx = 0
        pacman_pos = [nx, ny]

        if (nx, ny) in pellets:
            pellets.remove((nx, ny))
            pellets_eaten += 1

        # Advance Ghosts
        old_ghosts = [list(g) for g in ghost_positions]
        for i, gpos in enumerate(ghost_positions):
            glegal = get_legal_moves(gpos[0], gpos[1])
            if glegal:
                opp = OPPOSITE_DIRECTIONS.get(ghost_dirs[i])
                filtered = [m for m in glegal if m != opp] or glegal
                # Chase logic
                gm = min(
                    filtered,
                    key=lambda m: abs(gpos[0] + DIRECTIONS[m][0] - pacman_pos[0])
                    + abs(gpos[1] + DIRECTIONS[m][1] - pacman_pos[1]),
                )
                ghost_dirs[i] = gm
                gx, gy = gpos[0] + DIRECTIONS[gm][0], gpos[1] + DIRECTIONS[gm][1]
                if gx < 0:
                    gx = GRID_WIDTH - 1
                elif gx >= GRID_WIDTH:
                    gx = 0
                ghost_positions[i] = [gx, gy]

        # Check collision
        collided = False
        for i in range(len(ghost_positions)):
            if pacman_pos == ghost_positions[i] or (
                old_pac == ghost_positions[i] and pacman_pos == old_ghosts[i]
            ):
                collided = True
                break

        if collided:
            collisions += 1
            pacman_pos = list(START_POSITIONS["pacman"])
            ghost_positions = [list(g) for g in START_POSITIONS["ghosts"]]

        # Progress log every 10 steps
        if step % 10 == 0 or step == num_moves:
            print(f"  Step {step:3d}/{num_moves} | Choice: {chosen:5} | Latency: {res.latency_ms:5.1f}ms | Conf: {res.confidence:.2f}")

    total_wall_sec = time.perf_counter() - start_total_time

    # Calculate statistics
    avg_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    avg_conf = statistics.mean(confidences)
    decisions_per_sec = len(latencies) / total_wall_sec

    print("\n================================================================")
    print(" BENCHMARK RESULTS")
    print("================================================================")
    print(f"Completed Decisions:    {len(latencies)}")
    print(f"Total Wall-clock Time:  {total_wall_sec:.2f} s")
    print(f"Throughput:             {decisions_per_sec:.1f} decisions/sec")
    print("--- Latency Metrics ---")
    print(f"Mean Latency:           {avg_lat:.1f} ms")
    print(f"Median (P50) Latency:   {median_lat:.1f} ms")
    print(f"P95 Latency:            {p95_lat:.1f} ms")
    print(f"Min / Max Latency:      {min(latencies):.1f} ms / {max(latencies):.1f} ms")
    print("--- Quality & Game Metrics ---")
    print(f"Mean Model Confidence:  {avg_conf:.3f}")
    print(f"Pellets Eaten:          {pellets_eaten}")
    print(f"Ghost Collisions:       {collisions}")
    print("================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System 1 Decision Benchmark")
    parser.add_argument("--moves", type=int, default=50, help="Number of decisions to simulate")
    parser.add_argument("--model", type=str, default="nimble", help="Ollama model name")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--mock", action="store_true", help="Force heuristic simulation mode")
    args = parser.parse_args()

    run_benchmark(
        num_moves=args.moves,
        model=args.model,
        host=args.host,
        force_mock=args.mock,
    )
