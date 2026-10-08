"""
Decision client for Ollama System 1 / Jev-style decision models.
Supports POST /v1/systemone with enriched spatial context (ghost maze distances and
per-move food distances), rolling latency telemetry, and a heuristic fallback simulator.

Offline simulator latency is SYNTHETIC by default (to emulate a ~90 ms model round-trip
in the visual arena). Every such result carries `latency_simulated=True`; benchmarks
must label or exclude it. Pass `simulate_latency=False` to report measured compute time.
"""

import collections
import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from agents.base import DecisionResult, margin_confidence, softmax
from agents.features import FAR, min_ghost_bfs, nearest_pellet_bfs, next_cell, toroidal_manhattan
from core.maze_data import MAZE_DIST_MATRIX, OPPOSITE_DIRECTIONS

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "nimble")
SIMULATED_LATENCY_BASE_MS = 86.0


class SystemOneAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        prefer_live: bool = True,
        timeout_sec: float = 2.0,
        simulate_latency: bool = True,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.api_url = f"{self.host}/v1/systemone"
        self.prefer_live = prefer_live
        self.timeout_sec = timeout_sec
        self.simulate_latency = simulate_latency

        self.last_result: Optional[DecisionResult] = None
        self._cached_online_status: Optional[bool] = None
        self._last_status_check = 0.0

        # Telemetry history (fixed rolling window)
        self.latencies: collections.deque = collections.deque(maxlen=100)

    def reset(self) -> None:
        """Reset agent session / latency telemetry."""
        self.last_result = None
        self.latencies.clear()

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
        """Check if Ollama service is reachable (cached for 5 s)."""
        now = time.time()
        if not force_refresh and self._cached_online_status is not None:
            if now - self._last_status_check < 5.0:
                return self._cached_online_status

        self._last_status_check = now
        try:
            req = urllib.request.Request(f"{self.host}/api/version", headers={"User-Agent": "PacMan-SystemOne/1.0"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                self._cached_online_status = resp.status == 200
        except Exception:
            self._cached_online_status = False
        return self._cached_online_status

    # ------------------------------------------------------------------ prompt
    @staticmethod
    def move_context(
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        legal_moves: List[str],
        pellets: set,
    ) -> Dict[str, Dict[str, int]]:
        """Per-move maze distances: nearest ghost and nearest pellet after taking the move."""
        ctx = {}
        for m in legal_moves:
            n = next_cell(pacman_pos, m)
            ctx[m] = {
                "nearest_ghost_steps": min_ghost_bfs(n, ghost_positions),
                "nearest_pellet_steps": nearest_pellet_bfs(n, pellets),
                "eats_pellet": int(n in pellets),
            }
        return ctx

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
            maze = MAZE_DIST_MATRIX.get(((px, py), (gx, gy)))
            dist = maze if maze is not None else toroidal_manhattan((px, py), (gx, gy))
            dx, dy = gx - px, gy - py
            rel_x = "right" if dx > 0 else ("left" if dx < 0 else "aligned")
            rel_y = "down" if dy > 0 else ("up" if dy < 0 else "aligned")
            ghost_desc.append(f"Ghost {i+1} at distance {dist} ({rel_x}, {rel_y})")

        ctx = self.move_context(pacman_pos, ghost_positions, legal_moves, pellets)
        move_desc = []
        for m, c in ctx.items():
            g = "none" if c["nearest_ghost_steps"] >= FAR else c["nearest_ghost_steps"]
            p = "none" if c["nearest_pellet_steps"] >= FAR else c["nearest_pellet_steps"]
            move_desc.append(f"{m}: ghost {g} steps, food {p} steps{' (eats pellet)' if c['eats_pellet'] else ''}")

        has_pellet = "Yes" if tuple(pacman_pos) in pellets else "No"
        return (
            f"Pacman at position ({px}, {py}). Current pellet: {has_pellet}. Pellets remaining: {len(pellets)}. "
            f"Legal corridor moves: {', '.join(legal_moves)}. "
            f"Threats: {'; '.join(ghost_desc) if ghost_desc else 'none'}. "
            f"Per-move outlook: {'; '.join(move_desc)}."
        )

    # --------------------------------------------------------------- decisions
    def decide_move(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        """Query Ollama /v1/systemone (or heuristic simulator if offline / requested)."""
        if not legal_moves:
            return DecisionResult("left", {}, 0.0, 0.0, False, error_msg="no legal moves")

        if len(legal_moves) == 1:
            only_move = legal_moves[0]
            res = DecisionResult(only_move, {only_move: 1.0}, 1.0, 0.0, False)
            self.last_result = res
            return res

        if self.prefer_live and self.is_ollama_online():
            try:
                res = self._call_ollama_system_one(pacman_pos, ghost_positions, legal_moves, pellets)
            except Exception as e:
                res = self._heuristic_decision(
                    pacman_pos, ghost_positions, pellets, legal_moves, last_move, error_msg=f"live call failed: {e}"
                )
        else:
            res = self._heuristic_decision(pacman_pos, ghost_positions, pellets, legal_moves, last_move)

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
        """Issue a real POST to /v1/systemone and validate the answer."""
        payload = {
            "model": self.model,
            "state": {
                "description": self.build_spatial_description(pacman_pos, ghost_positions, legal_moves, pellets),
                "pacman": list(pacman_pos),
                "ghosts": [list(g) for g in ghost_positions],
                "legal_moves": legal_moves,
                "moves": self.move_context(pacman_pos, ghost_positions, legal_moves, pellets),
            },
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Which direction should Pac-Man move to eat pellets and avoid ghosts?",
                    "criteria": {m: f"Move {m}" for m in legal_moves},
                }
            },
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            resp_body = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        move_answer = (resp_body.get("answers") or {}).get("move") or {}
        error_msg = None

        raw_probs = move_answer.get("probabilities") or {}
        probs = {}
        for m in legal_moves:
            try:
                probs[m] = max(0.0, float(raw_probs.get(m, 0.0)))
            except (TypeError, ValueError):
                probs[m] = 0.0
        total = sum(probs.values())
        probs = {m: p / total for m, p in probs.items()} if total > 0 else {m: 1.0 / len(legal_moves) for m in legal_moves}

        choice = move_answer.get("choice")
        if choice not in legal_moves:
            error_msg = f"invalid model choice {choice!r}; fell back to highest-probability legal move"
            choice = max(legal_moves, key=lambda m: probs[m])

        try:
            confidence = float(move_answer.get("confidence", margin_confidence(probs)))
        except (TypeError, ValueError):
            confidence = margin_confidence(probs)

        return DecisionResult(
            choice=choice,
            probabilities=probs,
            confidence=max(0.0, min(1.0, confidence)),
            latency_ms=latency_ms,
            is_live=True,
            raw_response=resp_body,
            error_msg=error_msg,
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
        """Offline System 1 stand-in: softmax over a light heuristic, using the same context as the prompt."""
        t0 = time.perf_counter()
        ctx = self.move_context(pacman_pos, ghost_positions, legal_moves, pellets)
        scores: Dict[str, float] = {}

        for move, c in ctx.items():
            g = c["nearest_ghost_steps"]
            score = 0.0
            if g <= 1:
                score -= 120.0  # Immediate lethal threat!
            elif g == 2:
                score -= 40.0   # Danger zone
            elif g == 3:
                score -= 12.0
            else:
                score += min(g, 30) * 2.0

            if c["eats_pellet"]:
                score += 18.0
            if pellets:
                score -= min(c["nearest_pellet_steps"], 30) * 1.8

            # Discourage immediate U-turns unless escaping
            if last_move and move == OPPOSITE_DIRECTIONS.get(last_move) and g > 3:
                score -= 10.0
            scores[move] = score

        probabilities = softmax(scores, temp=10.0)
        best_move = max(legal_moves, key=lambda m: probabilities[m])

        calc_time = (time.perf_counter() - t0) * 1000.0
        if self.simulate_latency:
            latency = max(calc_time, SIMULATED_LATENCY_BASE_MS + (hash(tuple(pacman_pos)) % 16))
        else:
            latency = calc_time

        return DecisionResult(
            choice=best_move,
            probabilities=probabilities,
            confidence=margin_confidence(probabilities),
            latency_ms=latency,
            is_live=False,
            error_msg=error_msg,
            latency_simulated=self.simulate_latency,
        )


LocalFastSimulator = SystemOneAgent
