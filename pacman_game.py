"""
PacMan-AI-Chronicles: Multi-Agent Visualizer & Decision Arena (Pygame)
Seamlessly compare 40 years of AI decision paradigms in real-time:
[1] Policy-Optimized RL, [2] Deep Q-Network (PyTorch DQN), [3] Greedy Heuristic,
[4] Textbook Q-Learning, [5] System 1 (Ollama / Jev), [6] Random Baseline.
"""


import argparse
import collections
import math
import os
import sys
import threading
from typing import List, Optional

try:
    import pygame
except ImportError:
    print("\n[ERROR] Pygame is not installed in this environment.")
    print("Please install it with: pip install pygame-ce\n")
    sys.exit(1)

from agents.base import DecisionResult
from agents.dqn_agent import DQNAgent
from agents.registry import build_controllers
from core.environment import DEFAULT_MAX_STEPS, SCORE_DEATH, SCORE_PELLET, SCORE_WIN, Environment
from core.maze_data import GRID_HEIGHT, GRID_WIDTH
from llm.decision_client import SystemOneAgent


# Colors
COLOR_BG = (15, 23, 42)          # Slate 900
COLOR_PANEL_BG = (30, 41, 59)    # Slate 800
COLOR_PANEL_BORDER = (51, 65, 85) # Slate 700
COLOR_WALL = (71, 85, 105)       # Slate 600
COLOR_PELLET = (251, 191, 36)    # Amber 400
COLOR_PACMAN = (250, 204, 21)    # Yellow 400

# Ghost Colors matching the demo
COLOR_GHOSTS = [
    (217, 86, 92),   # Red / Blinky
    (220, 127, 176), # Pink / Pinky
    (224, 141, 85),  # Orange / Clyde
]

COLOR_TEXT_PRIMARY = (248, 250, 252)
COLOR_TEXT_MUTED = (148, 163, 184)
COLOR_ACCENT = (56, 189, 248)    # Sky 400
COLOR_BAR_BG = (51, 65, 85)
COLOR_BAR_FILL = (99, 102, 241)  # Indigo 500


class PacmanGame:
    def __init__(
        self,
        tile_size: int = 26,
        model: str = "nimble",
        host: str = "http://localhost:11434",
        force_mock: bool = False,
        speed: float = 6.0,
        initial_agent_idx: int = 1,
    ):
        # Ensure display driver uses native OS window (defend against headless dummy flags)
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            os.environ.pop("SDL_VIDEODRIVER", None)

        pygame.init()
        pygame.display.set_caption("PacMan-AI-Chronicles | 40 Years of AI Decision Paradigms")

        self.tile_size = tile_size
        self.maze_width = GRID_WIDTH * self.tile_size
        self.maze_height = GRID_HEIGHT * self.tile_size
        self.panel_width = 420

        self.screen_width = self.maze_width + self.panel_width
        self.screen_height = max(self.maze_height, 640)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_bold = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.font_regular = pygame.font.SysFont("Segoe UI", 12)
        self.font_mono = pygame.font.SysFont("Consolas", 11)
        self.font_small = pygame.font.SysFont("Segoe UI", 10)

        # System 1 backend
        self.sys1_backend = SystemOneAgent(
            model=model,
            host=host,
            prefer_live=(not force_mock),
        )

        # Register all 6 controllers in clean 1-6 order (shared with web_arena.py)
        self.controllers = build_controllers(self.sys1_backend, model)

        self.active_idx = initial_agent_idx % len(self.controllers)

        # Control flags
        self.paused = False
        self.step_once = False
        self.move_speed = speed
        self.speed_levels = [3.0, 6.0, 12.0]
        self.mouth_angle = 0.2
        self.mouth_dir = 1
        self.banner_text: Optional[str] = None
        self.banner_timer = 0.0

        # Async query support & active decision telemetry.
        # `decision_epoch` is bumped on reset / agent switch / respawn so results computed
        # for a stale game state are discarded instead of being applied to the new one.
        self.pending_decision = False
        self.decision_epoch = 0
        self.decision_lock = threading.RLock()
        self.latest_decision_result: Optional[DecisionResult] = None
        self.active_decision: Optional[DecisionResult] = None
        self.move_history = collections.deque(maxlen=6)
        # Game State
        self.reset_game()

    @property
    def current_controller(self):
        return self.controllers[self.active_idx]

    def set_controller(self, idx: int):
        idx = idx % len(self.controllers)
        if idx != self.active_idx:
            self._invalidate_pending_decision()
            self.active_idx = idx
            c = self.current_controller
            if hasattr(c["agent"], "reset"):
                c["agent"].reset()
            self.banner_text = f"Switched AI: {c['name']}"
            self.banner_timer = 2.0

    def _invalidate_pending_decision(self):
        """Discard any decision computed for the previous agent or game state."""
        with self.decision_lock:
            self.decision_epoch += 1
            self.pending_decision = False
            self.latest_decision_result = None
            self.active_decision = None

    def reset_game(self):
        """Reset full game to initial state."""
        self._invalidate_pending_decision()
        for c in self.controllers:
            if hasattr(c["agent"], "reset"):
                c["agent"].reset()

        self.env = Environment()
        self.pacman_pos = list(self.env.pacman_pos)
        self.pacman_visual = [float(self.pacman_pos[0]), float(self.pacman_pos[1])]
        self.pacman_dir = "left"
        self.last_move = "left"

        self.ghost_positions = [list(g) for g in self.env.ghost_positions]
        self.ghost_visuals = [[float(g[0]), float(g[1])] for g in self.ghost_positions]
        self.ghost_dirs = list(self.env.ghost_dirs)

        self.walls = self.env.walls
        self.pellets = self.env.pellets
        self.initial_pellet_count = len(self.pellets)
        self.score = 0
        self.lives = 3
        self.move_count = 0
        self.game_over = False
        self.victory = False
        self.collision_flash = 0
        self.move_history.clear()

    def get_legal_moves(self, x: int, y: int) -> List[str]:
        return self.env.get_legal_moves(x, y)

    def start_decision_query(self, legal_moves: List[str]):
        """Run decision query in background thread."""
        with self.decision_lock:
            if self.pending_decision:
                return
            self.pending_decision = True
            request_epoch = self.decision_epoch
            pac_pos = tuple(self.pacman_pos)
            ghost_pos = [tuple(g) for g in self.ghost_positions]
            pellets_copy = set(self.pellets)
            legal_copy = list(legal_moves)
            last_move = self.last_move
            env = getattr(self, "env", None)
            mode_step = getattr(env, "mode_step", 0)
            ghost_dirs = list(getattr(env, "ghost_dirs", ["up"] * len(ghost_pos)))
            steps_without_pellet = getattr(env, "steps_without_pellet", 0)
            active_agent = self.current_controller["agent"]

        def worker():
            if isinstance(active_agent, DQNAgent):
                res = active_agent.decide(
                    pac_pos, ghost_pos, pellets_copy, legal_copy, last_move,
                    mode_step=mode_step, ghost_dirs=ghost_dirs,
                    steps_without_pellet=steps_without_pellet,
                    # Open-ended session: keep the horizon plane at "full episode ahead"
                    # instead of letting it count down to 0 (never seen in training).
                    steps_remaining=DEFAULT_MAX_STEPS, horizon=DEFAULT_MAX_STEPS,
                )
            else:
                res = active_agent.decide(pac_pos, ghost_pos, pellets_copy, legal_copy, last_move)
            with self.decision_lock:
                if request_epoch == self.decision_epoch:
                    self.latest_decision_result = res
                    self.pending_decision = False

        threading.Thread(target=worker, daemon=True).start()

    def update(self, dt: float):
        if self.banner_timer > 0:
            self.banner_timer -= dt

        if self.game_over or self.victory:
            return

        if self.collision_flash > 0:
            self.collision_flash -= 1

        step = dt * self.move_speed
        for dim in [0, 1]:
            diff = self.pacman_pos[dim] - self.pacman_visual[dim]
            if abs(diff) > 2.0:
                self.pacman_visual[dim] = float(self.pacman_pos[dim])
            elif abs(diff) <= step:
                self.pacman_visual[dim] = float(self.pacman_pos[dim])
            else:
                self.pacman_visual[dim] += step if diff > 0 else -step

        for i in range(len(self.ghost_positions)):
            for dim in [0, 1]:
                diff = self.ghost_positions[i][dim] - self.ghost_visuals[i][dim]
                if abs(diff) > 2.0:
                    self.ghost_visuals[i][dim] = float(self.ghost_positions[i][dim])
                elif abs(diff) <= step:
                    self.ghost_visuals[i][dim] = float(self.ghost_positions[i][dim])
                else:
                    self.ghost_visuals[i][dim] += step if diff > 0 else -step

        # Mouth animation
        self.mouth_angle += self.mouth_dir * dt * 4.0
        if self.mouth_angle > 0.45:
            self.mouth_angle = 0.45
            self.mouth_dir = -1
        elif self.mouth_angle < 0.05:
            self.mouth_angle = 0.05
            self.mouth_dir = 1

        if self.paused and not self.step_once:
            return

        if (
            self.pacman_visual[0] != float(self.pacman_pos[0])
            or self.pacman_visual[1] != float(self.pacman_pos[1])
        ):
            return

        with self.decision_lock:
            if self.latest_decision_result is not None:
                decision = self.latest_decision_result
                self.latest_decision_result = None
                self.active_decision = decision  # Store active decision for telemetry display

                chosen_move = decision.choice
                legal = self.env.get_legal_moves(self.pacman_pos[0], self.pacman_pos[1])
                if chosen_move in legal:
                    collided, ate, won = self.env.step(chosen_move)

                    self.pacman_pos = list(self.env.pacman_pos)
                    self.ghost_positions = [list(g) for g in self.env.ghost_positions]
                    self.ghost_dirs = list(self.env.ghost_dirs)
                    self.pacman_dir = chosen_move
                    self.last_move = chosen_move
                    self.move_count += 1
                    self.move_history.append((self.move_count, chosen_move, decision.confidence, decision.latency_ms))

                    if ate:
                        self.score += SCORE_PELLET
                    if won:
                        self.score += SCORE_WIN
                        self.victory = True

                    if collided:
                        self.score += SCORE_DEATH
                        self.lives -= 1
                        self.collision_flash = 20
                        if self.lives <= 0:
                            self.game_over = True
                        else:
                            # Reset shared simulation state, including its Scatter/Chase clock.
                            self._invalidate_pending_decision()
                            self.env.respawn()
                            self.pacman_pos = list(self.env.pacman_pos)
                            self.ghost_positions = [list(g) for g in self.env.ghost_positions]
                            self.ghost_dirs = list(self.env.ghost_dirs)
                            self.pacman_visual = [float(self.pacman_pos[0]), float(self.pacman_pos[1])]
                            self.ghost_visuals = [[float(g[0]), float(g[1])] for g in self.ghost_positions]
                            # CRITICAL: reset agent temporal velocity tracking on respawn!
                            if hasattr(self.current_controller["agent"], "reset"):
                                self.current_controller["agent"].reset()

                    if self.step_once:
                        self.step_once = False
                        self.paused = True
                return

        if not self.pending_decision:
            legal = self.get_legal_moves(self.pacman_pos[0], self.pacman_pos[1])
            self.start_decision_query(legal)

    def draw_maze(self):
        maze_rect = pygame.Rect(0, 0, self.maze_width, self.screen_height)
        pygame.draw.rect(self.screen, COLOR_BG, maze_rect)

        for x, y in self.walls:
            rect = pygame.Rect(
                x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size
            )
            pygame.draw.rect(self.screen, COLOR_WALL, rect, border_radius=3)
            pygame.draw.rect(
                self.screen, (100, 116, 139), rect.inflate(-4, -4), width=1, border_radius=2
            )

        for x, y in self.pellets:
            cx = x * self.tile_size + self.tile_size // 2
            cy = y * self.tile_size + self.tile_size // 2
            pygame.draw.circle(self.screen, COLOR_PELLET, (cx, cy), 3)

        px = int(self.pacman_visual[0] * self.tile_size + self.tile_size / 2)
        py = int(self.pacman_visual[1] * self.tile_size + self.tile_size / 2)
        r = int(self.tile_size * 0.44)

        dir_angles = {
            "right": 0.0,
            "down": math.pi / 2,
            "left": math.pi,
            "up": -math.pi / 2,
        }
        base_angle = dir_angles.get(self.pacman_dir, 0.0)
        start_a = base_angle + self.mouth_angle * math.pi
        end_a = base_angle + (2 - self.mouth_angle) * math.pi

        points = [(px, py)]
        num_steps = 24
        for step in range(num_steps + 1):
            ang = start_a + (end_a - start_a) * (step / num_steps)
            points.append((px + r * math.cos(ang), py + r * math.sin(ang)))

        if len(points) > 2:
            pygame.draw.polygon(self.screen, COLOR_PACMAN, points)

        # Compute visual offsets so overlapping or adjacent ghosts remain 100% visible
        offsets = [(0, 0)] * len(self.ghost_visuals)
        n_ghosts = len(self.ghost_visuals)
        overlapping_pairs = []
        for a in range(n_ghosts):
            for b in range(a + 1, n_ghosts):
                dist = math.hypot(
                    self.ghost_visuals[a][0] - self.ghost_visuals[b][0],
                    self.ghost_visuals[a][1] - self.ghost_visuals[b][1],
                )
                if dist < 0.75:
                    overlapping_pairs.append((a, b))

        if overlapping_pairs:
            if len(overlapping_pairs) >= 3 or (
                (0, 1) in overlapping_pairs and (0, 2) in overlapping_pairs and (1, 2) in overlapping_pairs
            ):
                offsets[0] = (0, -5)
                offsets[1] = (-5, 4)
                offsets[2] = (5, 4)
            else:
                for a, b in overlapping_pairs:
                    offsets[a] = (-4, -3)
                    offsets[b] = (4, 3)

        for i, gvis in enumerate(self.ghost_visuals):
            ox, oy = offsets[i]
            gx = int(gvis[0] * self.tile_size + self.tile_size / 2) + ox
            gy = int(gvis[1] * self.tile_size + self.tile_size / 2) + oy
            is_staggered = (ox != 0 or oy != 0)
            gr = int(self.tile_size * 0.38) if is_staggered else int(self.tile_size * 0.42)
            color = COLOR_GHOSTS[i % len(COLOR_GHOSTS)]

            # Dark silhouette contour so overlapping ghosts stand out crisply
            if is_staggered:
                pygame.draw.circle(self.screen, (15, 23, 42), (gx, gy - 2), gr + 2)
                pygame.draw.rect(
                    self.screen,
                    (15, 23, 42),
                    pygame.Rect(gx - gr - 2, gy - 2, (gr + 2) * 2, gr + 2),
                )

            pygame.draw.circle(self.screen, color, (gx, gy - 2), gr)
            body_rect = pygame.Rect(gx - gr, gy - 2, gr * 2, gr)
            pygame.draw.rect(self.screen, color, body_rect)
            for j in range(3):
                wx = gx - gr + (j * 2 + 1) * (gr / 3)
                pygame.draw.circle(self.screen, color, (int(wx), gy + gr - 3), int(gr / 3))

            eye_offset_x = 3 if self.ghost_dirs[i] == "right" else (-3 if self.ghost_dirs[i] == "left" else 0)
            eye_offset_y = 3 if self.ghost_dirs[i] == "down" else (-3 if self.ghost_dirs[i] == "up" else 0)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx - 4 + eye_offset_x, gy - 4 + eye_offset_y), 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx + 4 + eye_offset_x, gy - 4 + eye_offset_y), 3)
            pygame.draw.circle(self.screen, (30, 41, 59), (gx - 4 + eye_offset_x * 1.5, gy - 4 + eye_offset_y * 1.5), 1.5)
            pygame.draw.circle(self.screen, (30, 41, 59), (gx + 4 + eye_offset_x * 1.5, gy - 4 + eye_offset_y * 1.5), 1.5)

        if self.collision_flash > 0:
            flash_surf = pygame.Surface((self.maze_width, self.screen_height), pygame.SRCALPHA)
            flash_surf.fill((239, 68, 68, 90))
            self.screen.blit(flash_surf, (0, 0))

        if self.banner_timer > 0 and self.banner_text:
            b_surf = pygame.Surface((self.maze_width - 40, 36), pygame.SRCALPHA)
            b_surf.fill((15, 23, 42, 230))
            pygame.draw.rect(b_surf, self.current_controller["color"], (0, 0, self.maze_width - 40, 36), width=2, border_radius=6)
            txt = self.font_bold.render(self.banner_text, True, self.current_controller["color"])
            b_surf.blit(txt, (15, 8))
            self.screen.blit(b_surf, (20, 20))

    def draw_telemetry_panel(self):
        panel_x = self.maze_width
        panel_rect = pygame.Rect(panel_x, 0, self.panel_width, self.screen_height)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(
            self.screen, COLOR_PANEL_BORDER, (panel_x, 0), (panel_x, self.screen_height), 2
        )

        pad = 18
        y = 14
        curr = self.current_controller
        curr_color = curr["color"]

        title_surf = self.font_title.render("PAC-MAN AI ARENA", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_surf, (panel_x + pad, y))
        y += 24

        # Active AI Controller Card
        ctrl_card = pygame.Rect(panel_x + pad, y, self.panel_width - pad * 2, 54)
        pygame.draw.rect(self.screen, COLOR_BG, ctrl_card, border_radius=8)
        pygame.draw.rect(self.screen, curr_color, ctrl_card, width=2, border_radius=8)

        self.screen.blit(
            self.font_bold.render(curr["name"], True, curr_color),
            (panel_x + pad + 12, y + 6),
        )
        self.screen.blit(
            self.font_small.render(f"{curr['badge']}  •  [TAB] to switch", True, COLOR_TEXT_MUTED),
            (panel_x + pad + 12, y + 30),
        )
        y += 64

        # Action Decision Card - Uses active_decision
        card_rect = pygame.Rect(panel_x + pad, y, self.panel_width - pad * 2, 102)
        pygame.draw.rect(self.screen, COLOR_BG, card_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, card_rect, width=1, border_radius=8)

        card_y = y + 8
        self.screen.blit(
            self.font_small.render(f"ACTIVE DECISION ({curr['type_label']})", True, COLOR_TEXT_MUTED),
            (panel_x + pad + 12, card_y),
        )
        card_y += 18

        last_res = self.active_decision
        choice_str = last_res.choice.upper() if last_res else "..."
        dir_symbols = {"LEFT": "◀ LEFT", "RIGHT": "▶ RIGHT", "UP": "▲ UP", "DOWN": "▼ DOWN"}
        choice_display = dir_symbols.get(choice_str, choice_str)
        choice_surf = self.font_title.render(choice_display, True, curr_color)
        self.screen.blit(choice_surf, (panel_x + pad + 12, card_y))

        lat_ms = last_res.latency_ms if last_res else 0.0
        conf = last_res.confidence if last_res else 0.0

        card_y += 30
        lat_fmt = f"{lat_ms:5.1f} ms" if lat_ms >= 1.0 else f"{lat_ms:5.2f} ms"
        metric_line = f"Latency: {lat_fmt}  |  Confidence: {conf:.2f}"
        self.screen.blit(
            self.font_mono.render(metric_line, True, COLOR_TEXT_PRIMARY),
            (panel_x + pad + 12, card_y),
        )

        card_y += 18
        cbar_w = self.panel_width - pad * 2 - 24
        pygame.draw.rect(
            self.screen, COLOR_BAR_BG, (panel_x + pad + 12, card_y, cbar_w, 5), border_radius=3
        )
        pygame.draw.rect(
            self.screen,
            curr_color,
            (panel_x + pad + 12, card_y, int(cbar_w * max(0.0, min(1.0, conf))), 5),
            border_radius=3,
        )
        y += 112

        # Evaluation Distribution Bars
        self.screen.blit(
            self.font_bold.render(curr["type_label"], True, COLOR_TEXT_PRIMARY),
            (panel_x + pad, y),
        )
        y += 20

        probs = (
            last_res.probabilities
            if (last_res and last_res.probabilities)
            else {"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25}
        )

        for move_name in ["up", "down", "left", "right"]:
            p = probs.get(move_name, 0.0)
            is_chosen = (last_res and last_res.choice == move_name)

            lbl_color = curr_color if is_chosen else COLOR_TEXT_MUTED
            lbl = self.font_mono.render(f"{move_name.upper():5}", True, lbl_color)
            self.screen.blit(lbl, (panel_x + pad, y + 2))

            bar_x = panel_x + pad + 65
            bar_w = self.panel_width - pad * 2 - 125
            pygame.draw.rect(
                self.screen, COLOR_BAR_BG, (bar_x, y + 4, bar_w, 13), border_radius=3
            )

            fill_w = int(bar_w * max(0.0, min(1.0, p)))
            fill_color = curr_color if is_chosen else COLOR_BAR_FILL
            if fill_w > 0:
                pygame.draw.rect(
                    self.screen, fill_color, (bar_x, y + 4, fill_w, 13), border_radius=3
                )

            pct_surf = self.font_mono.render(f"{p * 100:4.1f}%", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(pct_surf, (bar_x + bar_w + 10, y + 2))
            y += 22

        y += 10

        # Controller Quick-Select Menu Card
        self.screen.blit(
            self.font_bold.render("AVAILABLE CONTROLLERS (Press [1-6] or [TAB])", True, COLOR_TEXT_PRIMARY),
            (panel_x + pad, y),
        )
        y += 20

        menu_card = pygame.Rect(panel_x + pad, y, self.panel_width - pad * 2, 118)
        pygame.draw.rect(self.screen, COLOR_BG, menu_card, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, menu_card, width=1, border_radius=6)

        m_y = y + 6
        for i, c in enumerate(self.controllers):
            is_active = (i == self.active_idx)
            prefix = "▶ " if is_active else "  "
            text_color = c["color"] if is_active else COLOR_TEXT_MUTED
            line_str = f"{prefix}[{i+1}] {c['name']:<24}"
            self.screen.blit(self.font_mono.render(line_str, True, text_color), (panel_x + pad + 8, m_y))
            m_y += 18

        y += 128

        # Game Stats
        self.screen.blit(
            self.font_bold.render("GAME TELEMETRY", True, COLOR_TEXT_PRIMARY),
            (panel_x + pad, y),
        )
        y += 20

        stats = [
            f"Score: {self.score}   |   Moves: {self.move_count}",
            f"Pellets Remaining: {len(self.pellets)} / {self.initial_pellet_count}",
        ]
        for s in stats:
            self.screen.blit(self.font_regular.render(s, True, COLOR_TEXT_MUTED), (panel_x + pad, y))
            y += 18

        lives_txt = self.font_regular.render("Lives: ", True, COLOR_TEXT_MUTED)
        self.screen.blit(lives_txt, (panel_x + pad, y))
        for li in range(self.lives):
            lx = panel_x + pad + 60 + li * 18
            pygame.draw.circle(self.screen, COLOR_PACMAN, (lx, y + 7), 6)
        y += 26

        # Controls Hint
        controls_str = "[SPACE] Pause/Step  |  [R] Reset  |  [S] Speed"
        self.screen.blit(self.font_small.render(controls_str, True, COLOR_TEXT_MUTED), (panel_x + pad, y))

        if self.game_over:
            over_txt = self.font_title.render("GAME OVER - Press R", True, (239, 68, 68))
            self.screen.blit(over_txt, (panel_x + pad, self.screen_height - 30))
        elif self.victory:
            win_txt = self.font_title.render("VICTORY! All Pellets Eaten!", True, (34, 197, 94))
            self.screen.blit(win_txt, (panel_x + pad, self.screen_height - 30))
        elif self.paused:
            pause_txt = self.font_bold.render("|| PAUSED (Press SPACE to step)", True, (245, 158, 11))
            self.screen.blit(pause_txt, (panel_x + pad, self.screen_height - 30))

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.paused:
                            self.step_once = True
                            self.paused = False
                        else:
                            self.paused = True
                    elif event.key == pygame.K_TAB:
                        self.set_controller(self.active_idx + 1)
                    # Clean 1-6 keys for the 6 controllers
                    elif event.key == pygame.K_1:
                        self.set_controller(0)  # RL (Policy Optimized)
                    elif event.key == pygame.K_2:
                        self.set_controller(1)  # Deep Q-Network (DQN)
                    elif event.key == pygame.K_3:
                        self.set_controller(2)  # System 1
                    elif event.key == pygame.K_4:
                        self.set_controller(3)  # Greedy Heuristic
                    elif event.key == pygame.K_5:
                        self.set_controller(4)  # Q-Learning (Textbook)
                    elif event.key == pygame.K_6:
                        self.set_controller(5)  # Random Agent
                    elif event.key == pygame.K_s:
                        # Cycle speed: 3.0 -> 6.0 -> 12.0 -> 3.0
                        cur_idx = self.speed_levels.index(self.move_speed) if self.move_speed in self.speed_levels else 1
                        self.move_speed = self.speed_levels[(cur_idx + 1) % len(self.speed_levels)]
                        self.banner_text = f"Speed: {self.move_speed:.0f}x"
                        self.banner_timer = 1.5
                    elif event.key == pygame.K_m:
                        self.sys1_backend.prefer_live = not self.sys1_backend.prefer_live
                    elif event.key == pygame.K_r:
                        self.reset_game()

            self.update(dt)

            self.screen.fill(COLOR_BG)
            self.draw_maze()
            self.draw_telemetry_panel()
            pygame.display.flip()

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Pac-Man Multi-Agent AI Arena")
    parser.add_argument("--model", type=str, default="nimble", help="Ollama model name (default: nimble)")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--speed", type=float, default=6.0, help="Initial movement speed in tiles/sec (default: 6.0)")
    parser.add_argument("--mock", action="store_true", help="Force heuristic mock mode for System 1")
    parser.add_argument(
        "--agent",
        type=int,
        default=0,
        help="Initial active agent index (0: RL Optimized, 1: DQN, 2: System 1, 3: Greedy, 4: Q-Textbook, 5: Random)",
    )
    args = parser.parse_args()

    game = PacmanGame(
        model=args.model,
        host=args.host,
        force_mock=args.mock,
        speed=args.speed,
        initial_agent_idx=args.agent,
    )
    game.run()


if __name__ == "__main__":
    main()
