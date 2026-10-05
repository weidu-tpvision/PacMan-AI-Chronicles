"""
PacMan-AI-Chronicles: Web-Based Interactive AI Arena Server.
Serves a zero-dependency HTML5/Canvas frontend and connects it in real-time
to our PyTorch Deep Q-Network, Policy-Optimized RL, Greedy Heuristic, and System 1 LLM agents.
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import webbrowser

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.environment import Environment
from core.maze_data import GRID_HEIGHT, GRID_WIDTH, MAZE_LAYOUT, START_POSITIONS, WALL_CELLS
from agents import (
    DQNAgent,
    GreedyHeuristicAgent,
    PretrainedQLearningAgent,
    RandomAgent,
    SystemOneBaselineAgent,
    TrainedQLearningAgent,
)
from llm.decision_client import SystemOneAgent

PORT = 8080
HTML_FILE = os.path.join(os.path.dirname(__file__), "web", "index.html")


class GameSession:
    def __init__(self, model: str = "nimble", host: str = "http://localhost:11434"):
        self.env = Environment()
        self.sys1_backend = SystemOneAgent(model=model, host=host, prefer_live=False)

        self.controllers = [
            {
                "id": "rl_opt",
                "name": "RL (Policy Optimized)",
                "badge": "Graph-Aware RL (920+ pts)",
                "sub": "Cross-Entropy policy optimization on topological features",
                "agent": TrainedQLearningAgent(),
            },
            {
                "id": "dqn",
                "name": "Deep Q-Network (DQN)",
                "badge": "6-Channel PyTorch CNN",
                "sub": "Learned end-to-end directly from raw spatial grid tensors",
                "agent": DQNAgent(),
            },
            {
                "id": "sys1",
                "name": f"System 1 ({model})",
                "badge": "Neural Zero-Shot (~90ms)",
                "sub": "Spatial prompt reasoning via structured JSON logits",
                "agent": SystemOneBaselineAgent(self.sys1_backend),
            },
            {
                "id": "greedy",
                "name": "Greedy Heuristic",
                "badge": "Handcrafted Rules (570 pts)",
                "sub": "Symbolic priority queue balancing BFS food search & ghost evasion",
                "agent": GreedyHeuristicAgent(),
            },
            {
                "id": "rl_textbook",
                "name": "Q-Learning (Textbook)",
                "badge": "Classic TD-Learning (717 pts)",
                "sub": "Standard linear approximation with classic Bellman updates",
                "agent": PretrainedQLearningAgent(),
            },
            {
                "id": "random",
                "name": "Random Agent",
                "badge": "Noise Floor (Baseline)",
                "sub": "Uniform random distribution across legal corridors",
                "agent": RandomAgent(),
            },
        ]

        self.active_idx = 1  # Default to DQN
        self.score = 0
        self.moves = 0
        self.lives = 3
        self.last_decision = {
            "choice": "...",
            "confidence": 0.0,
            "latency_ms": 0.0,
            "probabilities": {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25},
        }

    @property
    def current_agent(self):
        return self.controllers[self.active_idx]["agent"]

    def reset(self):
        self.env = Environment()
        if hasattr(self.current_agent, "reset"):
            self.current_agent.reset()
        self.score = 0
        self.moves = 0
        self.lives = 3

    def step(self):
        legal = self.env.get_legal_moves(self.env.pacman_pos[0], self.env.pacman_pos[1])
        if not legal:
            return self.get_state()

        res = self.current_agent.decide(
            tuple(self.env.pacman_pos),
            [tuple(g) for g in self.env.ghost_positions],
            self.env.pellets,
            legal,
            self.env.last_move,
        )

        col, ate, won = self.env.step(res.choice)
        self.moves += 1
        if ate:
            self.score += 10

        if col:
            self.lives -= 1
            self.score = max(0, self.score - 100)
            if self.lives <= 0:
                self.reset()
            else:
                self.env.pacman_pos = list(START_POSITIONS["pacman"])
                self.env.ghost_positions = [list(g) for g in START_POSITIONS["ghosts"]]
                if hasattr(self.current_agent, "reset"):
                    self.current_agent.reset()
        elif won:
            self.score += 500
            self.reset()

        self.last_decision = {
            "choice": res.choice,
            "confidence": res.confidence,
            "latency_ms": res.latency_ms,
            "probabilities": res.probabilities,
        }

        return self.get_state()

    def get_state(self):
        return {
            "pacman": list(self.env.pacman_pos),
            "ghosts": [list(g) for g in self.env.ghost_positions],
            "pellets": [list(p) for p in self.env.pellets],
            "walls": list(WALL_CELLS),
            "pellets_left": len(self.env.pellets),
            "score": self.score,
            "moves": self.moves,
            "lives": self.lives,
            "last_move": self.env.last_move,
            "decision": self.last_decision,
            "agent_info": {
                "name": self.controllers[self.active_idx]["name"],
                "badge": self.controllers[self.active_idx]["badge"],
                "sub": self.controllers[self.active_idx]["sub"],
            },
        }


session = GameSession()


class ArenaHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(HTML_FILE, "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(session.get_state()).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/step":
            data = session.step()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif self.path == "/api/switch_agent":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            idx = int(body.get("index", 1)) % len(session.controllers)
            session.active_idx = idx
            if hasattr(session.current_agent, "reset"):
                session.current_agent.reset()
            info = session.controllers[session.active_idx]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(info).encode("utf-8"))
        elif self.path == "/api/reset":
            session.reset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence verbose per-request HTTP access logging
        pass


def main():
    parser = argparse.ArgumentParser(description="Pac-Man Web AI Arena Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print("\n==========================================================================================")
    print(" PACMAN-AI-CHRONICLES: Web Interactive AI Arena")
    print("==========================================================================================")
    print(f" * Local Server URL:  {url}")
    print(f" * Active AI Model:   {session.controllers[session.active_idx]['name']}")
    print(f" * Backend Runtime:   PyTorch DQN & Policy-Optimized RL")
    print("------------------------------------------------------------------------------------------")
    print(f" [!] Opening {url} in your default browser...")
    print("     Keys: [1]-[6] Switch AI | [Space] Pause | [R] Reset | [Ctrl+C] to stop")
    print("==========================================================================================\n", flush=True)

    if not args.no_browser:
        webbrowser.open(url)

    # Allow immediate address reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), ArenaHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[ARENA] Web server stopped cleanly.")


if __name__ == "__main__":
    main()
