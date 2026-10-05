"""
Decision client for Ollama System 1 / Jev-style decision models.
Supports POST /v1/systemone with enriched spatial context,
rolling latency telemetry, and heuristic fallback simulation.
"""

import collections
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple, Any

from agents.base import DecisionResult, softmax
from core.maze_data import DIRECTIONS, GRID_WIDTH, OPPOSITE_DIRECTIONS

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "nimble")


class SystemOneAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        prefer_live: bool = True,
        timeout_sec: float = 2.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.api_url = f"{self.host}/v1/systemone"
        self.prefer_live = prefer_live
        self.timeout_sec = timeout_sec

        self.last_result: Optional[DecisionResult] = None
        self._cached_online_status: Optional[bool] = None
        self._last_status_check = 0.0

        # Telemetry history (fixed rolling window)
        self.latencies: collections.deque = collections.deque(maxlen=100)

    def reset(self) -> None:
        """Reset agent session / latency telemetry."""
        self.last_result = None

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def min_latency(self) -> float:
        return min(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    def is_ollama_online(self, force_refresh: bool = False) -> bool:
        """Check if Ollama service is reachable."""
        now = time.time()
        if not force_refresh and self._cached_online_status is not None:
            if now - self._last_status_check < 5.0:
                return self._cached_online_status

        self._last_status_check = now
        try:
            req = urllib.request.Request(
                f"{self.host}/api/version",
                headers={"User-Agent": "PacMan-SystemOne/1.0"},
            )
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                self._cached_online_status = (resp.status == 200)
                return self._cached_online_status
        except Exception:
            self._cached_online_status = False
            return False

    def build_spatial_description(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        legal_moves: List[str],
        pellets: set,
    ) -> str:
        """Constructs a natural spatial summary for transformer reasoning."""
        px, py = pacman_pos
        ghost_desc = []
        for i, (gx, gy) in enumerate(ghost_positions):
            dist = abs(px - gx) + abs(py - gy)
            dx = gx - px
            dy = gy - py
            rel_x = "right" if dx > 0 else ("left" if dx < 0 else "aligned")
            rel_y = "down" if dy > 0 else ("up" if dy < 0 else "aligned")
            ghost_desc.append(f"Ghost {i+1} at distance {dist} ({rel_x}, {rel_y})")

        has_pellet = "Yes" if pacman_pos in pellets else "No"
        return (
            f"Pacman at position ({px}, {py}). Current pellet: {has_pellet}. "
            f"Legal corridor moves: {', '.join(legal_moves)}. "
            f"Threats: {'; '.join(ghost_desc)}."
        )

    def decide_move(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        """
        Query Ollama /v1/systemone (or heuristic mock if offline / requested).
        """
        if not legal_moves:
            return DecisionResult("left", {"left": 1.0}, 1.0, 0.0, False)

        if len(legal_moves) == 1:
            only_move = legal_moves[0]
            res = DecisionResult(only_move, {only_move: 1.0}, 1.0, 0.5, self.is_ollama_online())
            self.last_result = res
            return res

        if self.prefer_live and self.is_ollama_online():
            try:
                res = self._call_ollama_system_one(
                    pacman_pos, ghost_positions, legal_moves, pellets
                )
            except Exception as e:
                res = self._heuristic_decision(
                    pacman_pos, ghost_positions, pellets, legal_moves, last_move, error_msg=str(e)
                )
        else:
            res = self._heuristic_decision(
                pacman_pos, ghost_positions, pellets, legal_moves, last_move
            )

        self.latencies.append(res.latency_ms)
        self.last_result = res
        return res

    def _call_ollama_system_one(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        legal_moves: List[str],
        pellets: set,
    ) -> DecisionResult:
        """Issue a real POST to /v1/systemone."""
        criteria = {m: f"Move {m}" for m in legal_moves}
        spatial_summary = self.build_spatial_description(
            pacman_pos, ghost_positions, legal_moves, pellets
        )

        payload = {
            "model": self.model,
            "state": {
                "description": spatial_summary,
                "pacman": list(pacman_pos),
                "ghosts": [list(g) for g in ghost_positions],
                "legal_moves": legal_moves,
            },
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Which direction should Pac-Man move to eat pellets and avoid ghosts?",
                    "criteria": criteria,
                }
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp_body = json.loads(response.read().decode("utf-8"))

        answers = resp_body.get("answers", {})
        move_answer = answers.get("move", {})

        choice = move_answer.get("choice")
        if choice not in legal_moves:
            choice = legal_moves[0]

        probabilities = move_answer.get("probabilities", {})
        for m in legal_moves:
            if m not in probabilities:
                probabilities[m] = 0.0

        confidence = move_answer.get("confidence", 0.0)

        return DecisionResult(
            choice=choice,
            probabilities=probabilities,
            confidence=confidence,
            latency_ms=latency_ms,
            is_live=True,
            raw_response=resp_body,
        )

    def _heuristic_decision(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
        error_msg: Optional[str] = None,
    ) -> DecisionResult:
        """
        Simulates System 1 decision-making with softmax probabilities
        and realistic ~80-95ms response latency.
        """
        t0 = time.perf_counter()
        scores: Dict[str, float] = {}

        px, py = pacman_pos
        nearby_pellets = (
            sorted(pellets, key=lambda p: abs(px - p[0]) + abs(py - p[1]))[:30]
            if pellets else []
        )

        for move in legal_moves:
            dx, dy = DIRECTIONS[move]
            nx, ny = (px + dx) % GRID_WIDTH, py + dy

            # Distance to nearest ghost
            min_ghost_dist = min(
                abs(nx - gx) + abs(ny - gy) for gx, gy in ghost_positions
            )

            score = 0.0
            if min_ghost_dist <= 1:
                score -= 120.0  # Immediate lethal threat!
            elif min_ghost_dist == 2:
                score -= 40.0   # Danger zone
            elif min_ghost_dist == 3:
                score -= 12.0
            else:
                score += min_ghost_dist * 2.0

            # Pellet proximity
            if (nx, ny) in pellets:
                score += 18.0

            if nearby_pellets:
                min_pellet_dist = min(
                    abs(nx - px2) + abs(ny - py2) for px2, py2 in nearby_pellets
                )
                score -= min_pellet_dist * 1.8

            # Discourage immediate U-turns unless escaping
            if last_move and move == OPPOSITE_DIRECTIONS.get(last_move) and min_ghost_dist > 3:
                score -= 10.0

            scores[move] = score

        probabilities = softmax(scores, temp=10.0)

        best_move = max(probabilities.keys(), key=lambda m: probabilities[m])
        sorted_probs = sorted(probabilities.values(), reverse=True)
        confidence = (
            sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        )
        confidence = round(confidence, 3)

        calc_time = (time.perf_counter() - t0) * 1000.0
        simulated_latency = max(calc_time, 86.0 + (hash(pacman_pos) % 16))

        return DecisionResult(
            choice=best_move,
            probabilities=probabilities,
            confidence=confidence,
            latency_ms=simulated_latency,
            is_live=False,
            error_msg=error_msg,
        )


LocalFastSimulator = SystemOneAgent
