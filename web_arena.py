"""
PacMan-AI-Chronicles: Web-Based Interactive AI Arena Server.
Serves a zero-dependency HTML5/Canvas frontend and connects it in real-time
to our PyTorch Deep Q-Network, Policy-Optimized RL, Greedy Heuristic, and System 1 LLM agents.

Security: binds to 127.0.0.1 by default and serves ONLY index.html plus the JSON API
(no static-file fallback, so the working directory is never exposed).
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.registry import build_controllers
from agents.dqn_agent import DQNAgent
from core.environment import DEFAULT_MAX_STEPS, SCORE_DEATH, SCORE_PELLET, SCORE_WIN, Environment
from core.maze_data import WALL_CELLS
from llm.decision_client import SystemOneAgent

HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
MAX_BODY_BYTES = 4096


class GameSession:
    def __init__(self, model: str = "nimble", host: str = "http://localhost:11434", live: bool = False):
        self.sys1_backend = SystemOneAgent(model=model, host=host, prefer_live=live)
        self.controllers = build_controllers(self.sys1_backend, model)
        self.active_idx = 1  # Default to DQN
        self.reset()

    @property
    def current_agent(self):
        return self.controllers[self.active_idx]["agent"]

    def switch(self, idx: int) -> dict:
        self.active_idx = idx % len(self.controllers)
        self.current_agent.reset()
        c = self.controllers[self.active_idx]
        return {k: c[k] for k in ("id", "name", "badge", "sub")}

    def reset(self):
        self.env = Environment()
        self.current_agent.reset()
        self.score = 0
        self.moves = 0
        self.lives = 3
        self.level = 1
        self.last_event: Optional[str] = None
        self.last_decision = {
            "choice": "...",
            "confidence": 0.0,
            "latency_ms": 0.0,
            "probabilities": {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25},
        }

    def step(self):
        self.last_event = None
        legal = self.env.get_legal_moves(*self.env.pacman_pos)
        agent = self.current_agent
        args = (
            tuple(self.env.pacman_pos), [tuple(g) for g in self.env.ghost_positions],
            self.env.pellets, legal, self.env.last_move,
        )
        if isinstance(agent, DQNAgent):
            res = agent.decide(
                *args, mode_step=self.env.mode_step, ghost_dirs=self.env.ghost_dirs,
                steps_without_pellet=self.env.steps_without_pellet,
                # Open-ended session: keep the horizon plane at "full episode ahead"
                # instead of letting it count down to 0 (never seen in training).
                steps_remaining=DEFAULT_MAX_STEPS, horizon=DEFAULT_MAX_STEPS,
            )
        else:
            res = agent.decide(*args)

        col, ate, won = self.env.step(res.choice)
        self.moves += 1
        if ate:
            self.score += SCORE_PELLET

        if col:
            self.score += SCORE_DEATH
            self.lives -= 1
            if self.lives <= 0:
                self.last_event = f"game_over:{self.score}"
                final_event = self.last_event
                self.reset()
                self.last_event = final_event
            else:
                self.last_event = "life_lost"
                self.env.respawn()
                self.current_agent.reset()
        elif won:
            # Keep score & lives, advance to a fresh board (win bonus stays visible)
            self.score += SCORE_WIN
            self.level += 1
            self.last_event = "level_cleared"
            self.env = Environment()
            self.current_agent.reset()

        self.last_decision = {
            "choice": res.choice,
            "confidence": res.confidence,
            "latency_ms": res.latency_ms,
            "latency_simulated": res.latency_simulated,
            "probabilities": res.probabilities,
        }
        return self.get_state(include_walls=False)

    def get_state(self, include_walls: bool = True):
        c = self.controllers[self.active_idx]
        state = {
            "pacman": list(self.env.pacman_pos),
            "ghosts": [list(g) for g in self.env.ghost_positions],
            "pellets": [list(p) for p in self.env.pellets],
            "pellets_left": len(self.env.pellets),
            "score": self.score,
            "moves": self.moves,
            "lives": self.lives,
            "level": self.level,
            "event": self.last_event,
            "last_move": self.env.last_move,
            "decision": self.last_decision,
            "agent_info": {"name": c["name"], "badge": c["badge"], "sub": c["sub"]},
        }
        if include_walls:
            # Static: the client fetches them once via GET /api/state instead of on every step
            state["walls"] = [list(w) for w in WALL_CELLS]
        return state


session: Optional[GameSession] = None
# Serializes all access to the shared session: the server is thread-per-request.
session_lock = threading.Lock()


class ArenaHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Thread-per-request server: a slow live System 1 call must not freeze the UI."""
    daemon_threads = True
    allow_reuse_address = True


class ArenaHandler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, payload, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(HTML_FILE, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with session_lock:
                state = session.get_state()
            self._send_json(state)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/api/step":
                with session_lock:
                    state = session.step()
                self._send_json(state)
            elif self.path == "/api/switch_agent":
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length > MAX_BODY_BYTES:
                    self._send_json({"error": "body too large"}, 413)
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                index = int(body.get("index", 1))
                with session_lock:
                    info = session.switch(index)
                self._send_json(info)
            elif self.path == "/api/reset":
                with session_lock:
                    session.reset()
                self._send_json({"status": "ok"})
            else:
                self._send_json({"error": "not found"}, 404)
        except (ValueError, TypeError, AttributeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, 400)

    def log_message(self, format, *args):
        # Silence verbose per-request HTTP access logging
        pass


def main():
    global session
    parser = argparse.ArgumentParser(description="Pac-Man Web AI Arena Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--bind", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1, local only)")
    parser.add_argument("--model", type=str, default="nimble", help="Ollama model name for System 1")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--live", action="store_true", help="Query live Ollama for System 1 (default: offline simulator)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    session = GameSession(model=args.model, host=args.host, live=args.live)

    url = f"http://localhost:{args.port}"
    print("\n==========================================================================================")
    print(" PACMAN-AI-CHRONICLES: Web Interactive AI Arena")
    print("==========================================================================================")
    print(f" * Local Server URL:  {url}  (bound to {args.bind})")
    print(f" * Active AI Model:   {session.controllers[session.active_idx]['name']}")
    print("------------------------------------------------------------------------------------------")
    print("     Keys: [1]-[6] Switch AI | [Space] Pause | [R] Reset | [Ctrl+C] to stop")
    print("==========================================================================================\n", flush=True)

    if not args.no_browser:
        webbrowser.open(url)

    with ArenaHTTPServer((args.bind, args.port), ArenaHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[ARENA] Web server stopped cleanly.")


if __name__ == "__main__":
    main()
