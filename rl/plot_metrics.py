"""
Visualization generator for DQN training and validation metrics.
Generates publication-quality 4-panel figures in both PNG and SVG formats.
"""

import json
import math
import os
from typing import Dict, List

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


def generate_svg(metrics: List[Dict], output_svg_path: str):
    """Generate high-resolution 4-panel SVG vector figure."""
    width = 1100
    height = 750
    margin = 70
    pw = 450
    ph = 270

    # Panel positions: (x, y)
    panels = [
        (margin, 60, "Training Loss (Huber Loss)", "#e74c3c"),
        (margin + pw + 80, 60, "Training Score & 50-Ep Moving Average", "#3498db"),
        (margin, 60 + ph + 80, "Validation Deterministic Score", "#2ecc71"),
        (margin + pw + 80, 60 + ph + 80, "Validation Pellets Cleared & Epsilon", "#f39c12"),
    ]

    episodes = [m["episode"] for m in metrics]
    max_ep = max(episodes) if episodes else 1

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="background-color: #0f141c; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        f'<style>',
        f'  .title {{ fill: #ecf0f1; font-size: 20px; font-weight: bold; }}',
        f'  .subtitle {{ fill: #95a5a6; font-size: 13px; }}',
        f'  .panel-title {{ fill: #ecf0f1; font-size: 14px; font-weight: 600; }}',
        f'  .axis-label {{ fill: #7f8c8d; font-size: 11px; }}',
        f'  .grid {{ stroke: #1e293b; stroke-width: 1; }}',
        f'  .axis {{ stroke: #334155; stroke-width: 1.5; }}',
        f'  .legend {{ font-size: 11px; font-weight: 500; }}',
        f'</style>',
        f'<text x="{margin}" y="35" class="title">Deep Q-Network (DQN) Pac-Man Training Diagnostics</text>',
        f'<text x="{margin + 580}" y="35" class="subtitle">Architecture: 30-Channel Spatial Encoder, Dueling Double DQN, MaxPool2d(2)</text>',
    ]

    def draw_panel(px, py, title, color):
        # Background
        svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#161f2e" rx="8" stroke="#253347" stroke-width="1"/>')
        svg.append(f'<text x="{px + 15}" y="{py + 22}" class="panel-title">{title}</text>')
        # Axis lines
        ax_x0 = px + 45
        ax_y0 = py + ph - 35
        ax_w = pw - 65
        ax_h = ph - 65
        svg.append(f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0 + ax_w}" y2="{ax_y0}" class="axis"/>')
        svg.append(f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0}" y2="{ax_y0 - ax_h}" class="axis"/>')
        return ax_x0, ax_y0, ax_w, ax_h

    # 1. Panel 1: Loss
    px, py, title, col = panels[0]
    x0, y0, w, h = draw_panel(px, py, title, col)
    losses = [m.get("loss", 0.0) for m in metrics]
    max_loss = max(losses) if losses and max(losses) > 0 else 1.0
    pts = []
    for i, m in enumerate(metrics):
        x = x0 + (m["episode"] / max_ep) * w
        y = y0 - (m.get("loss", 0.0) / max_loss) * h
        pts.append(f"{x:.1f},{y:.1f}")
    if pts:
        svg.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.8" stroke-linecap="round" points="{" ".join(pts)}"/>')
    svg.append(f'<text x="{x0 + w - 40}" y="{y0 + 20}" class="axis-label">Ep {max_ep}</text>')
    svg.append(f'<text x="{x0 - 40}" y="{y0 - h + 10}" class="axis-label">{max_loss:.2f}</text>')
    svg.append(f'<text x="{x0 - 40}" y="{y0}" class="axis-label">0.0</text>')

    # 2. Panel 2: Training Score & Moving Avg
    px, py, title, col = panels[1]
    x0, y0, w, h = draw_panel(px, py, title, col)
    scores = [m.get("score", 0.0) for m in metrics]
    min_s = min(scores) if scores else -200.0
    max_s = max(scores) if scores else 500.0
    if max_s == min_s:
        max_s += 1.0
    # Zero line
    zero_y = y0 - ((0 - min_s) / (max_s - min_s)) * h
    if y0 - h <= zero_y <= y0:
        svg.append(f'<line x1="{x0}" y1="{zero_y}" x2="{x0 + w}" y2="{zero_y}" stroke="#334155" stroke-dasharray="3,3"/>')

    # Raw scores (faint)
    raw_pts = [f"{x0 + (m['episode']/max_ep)*w:.1f},{y0 - ((m['score']-min_s)/(max_s-min_s))*h:.1f}" for m in metrics]
    svg.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="1.0" opacity="0.35" points="{" ".join(raw_pts)}"/>')
    # Moving avg
    ma_pts = [f"{x0 + (m['episode']/max_ep)*w:.1f},{y0 - ((m['train_avg_score']-min_s)/(max_s-min_s))*h:.1f}" for m in metrics if "train_avg_score" in m]
    if ma_pts:
        svg.append(f'<polyline fill="none" stroke="#60a5fa" stroke-width="2.2" stroke-linecap="round" points="{" ".join(ma_pts)}"/>')
    svg.append(f'<text x="{x0 - 40}" y="{y0 - h + 10}" class="axis-label">{int(max_s)}</text>')
    svg.append(f'<text x="{x0 - 40}" y="{y0}" class="axis-label">{int(min_s)}</text>')
    svg.append(f'<text x="{px + pw - 130}" y="{py + 22}" class="legend" fill="#60a5fa">― 50-Ep Avg</text>')

    # 3. Panel 3: Validation Score
    px, py, title, col = panels[2]
    x0, y0, w, h = draw_panel(px, py, title, col)
    val_pts = []
    val_metrics = [m for m in metrics if m.get("val_score") is not None]
    if val_metrics:
        v_scores = [m["val_score"] for m in val_metrics]
        min_v = min(v_scores + [-150.0])
        max_v = max(v_scores + [200.0])
        for m in val_metrics:
            x = x0 + (m["episode"] / max_ep) * w
            y = y0 - ((m["val_score"] - min_v) / (max_v - min_v)) * h
            val_pts.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" stroke-linecap="round" points="{" ".join(val_pts)}"/>')
        for pt in val_pts:
            cx, cy = pt.split(",")
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="{col}"/>')
        svg.append(f'<text x="{x0 - 40}" y="{y0 - h + 10}" class="axis-label">{int(max_v)}</text>')
        svg.append(f'<text x="{x0 - 40}" y="{y0}" class="axis-label">{int(min_v)}</text>')

    # 4. Panel 4: Pellets & Epsilon
    px, py, title, col = panels[3]
    x0, y0, w, h = draw_panel(px, py, title, col)
    # Pellets
    pel_pts = []
    for m in val_metrics:
        x = x0 + (m["episode"] / max_ep) * w
        y = y0 - (m.get("val_pellets", 0) / 100.0) * h
        pel_pts.append(f"{x:.1f},{y:.1f}")
    if pel_pts:
        svg.append(f'<polyline fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-linecap="round" points="{" ".join(pel_pts)}"/>')
        for pt in pel_pts:
            cx, cy = pt.split(",")
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="#f59e0b"/>')
    # Epsilon
    eps_pts = [f"{x0 + (m['episode']/max_ep)*w:.1f},{y0 - (m.get('epsilon', 1.0))*h:.1f}" for m in metrics]
    if eps_pts:
        svg.append(f'<polyline fill="none" stroke="#a855f7" stroke-width="1.8" stroke-dasharray="4,4" points="{" ".join(eps_pts)}"/>')
    svg.append(f'<text x="{px + pw - 200}" y="{py + 22}" class="legend" fill="#f59e0b">― Val Pellets</text>')
    svg.append(f'<text x="{px + pw - 100}" y="{py + 22}" class="legend" fill="#a855f7">┄ Epsilon</text>')
    svg.append(f'<text x="{x0 - 40}" y="{y0 - h + 10}" class="axis-label">100</text>')
    svg.append(f'<text x="{x0 - 40}" y="{y0}" class="axis-label">0</text>')

    svg.append("</svg>")

    with open(output_svg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(svg))


def generate_png(metrics: List[Dict], output_png_path: str):
    """Generate high-resolution PNG using PyGame surface rasterization."""
    if not PYGAME_AVAILABLE:
        return

    pygame.font.init()
    w, h = 1100, 750
    surf = pygame.Surface((w, h))
    surf.fill((15, 20, 28))  # Dark background

    font_title = pygame.font.SysFont("Arial", 20, bold=True)
    font_sub = pygame.font.SysFont("Arial", 12)
    font_lbl = pygame.font.SysFont("Arial", 11)

    t_img = font_title.render("Deep Q-Network (DQN) Pac-Man Training Diagnostics", True, (236, 240, 241))
    s_img = font_sub.render("Architecture: 30-Channel Spatial Encoder, Dueling Double DQN, MaxPool2d(2)", True, (149, 165, 166))
    surf.blit(t_img, (70, 15))
    surf.blit(s_img, (600, 20))

    margin = 70
    pw, ph = 450, 270
    panels = [
        (margin, 60, "Training Loss (Huber Loss)", (231, 76, 60)),
        (margin + pw + 80, 60, "Training Score & 50-Ep Moving Average", (52, 152, 219)),
        (margin, 60 + ph + 80, "Validation Deterministic Score", (46, 204, 113)),
        (margin + pw + 80, 60 + ph + 80, "Validation Pellets Cleared & Epsilon", (243, 156, 18)),
    ]

    episodes = [m["episode"] for m in metrics]
    max_ep = max(episodes) if episodes else 1

    for px, py, title, color in panels:
        pygame.draw.rect(surf, (22, 31, 46), (px, py, pw, ph), border_radius=8)
        pygame.draw.rect(surf, (37, 51, 71), (px, py, pw, ph), width=1, border_radius=8)
        title_img = font_lbl.render(title, True, (236, 240, 241))
        surf.blit(title_img, (px + 15, py + 12))

        ax_x0 = px + 45
        ax_y0 = py + ph - 35
        ax_w = pw - 65
        ax_h = ph - 65
        pygame.draw.line(surf, (51, 65, 85), (ax_x0, ax_y0), (ax_x0 + ax_w, ax_y0), 1)
        pygame.draw.line(surf, (51, 65, 85), (ax_x0, ax_y0), (ax_x0, ax_y0 - ax_h), 1)

    # Panel 1: Loss
    px, py, _, col = panels[0]
    ax_x0, ax_y0 = px + 45, py + ph - 35
    ax_w, ax_h = pw - 65, ph - 65
    losses = [m.get("loss", 0.0) for m in metrics]
    max_l = max(losses) if losses and max(losses) > 0 else 1.0
    pts = [(int(ax_x0 + (m["episode"] / max_ep) * ax_w), int(ax_y0 - (m.get("loss", 0.0) / max_l) * ax_h)) for m in metrics]
    if len(pts) > 1:
        pygame.draw.lines(surf, col, False, pts, 2)

    # Panel 2: Scores
    px, py, _, col = panels[1]
    ax_x0, ax_y0 = px + 45, py + ph - 35
    ax_w, ax_h = pw - 65, ph - 65
    scores = [m.get("score", 0.0) for m in metrics]
    min_s = min(scores) if scores else -200.0
    max_s = max(scores) if scores else 500.0
    if max_s == min_s: max_s += 1.0
    raw_pts = [(int(ax_x0 + (m["episode"]/max_ep)*ax_w), int(ax_y0 - ((m["score"]-min_s)/(max_s-min_s))*ax_h)) for m in metrics]
    if len(raw_pts) > 1:
        pygame.draw.lines(surf, (37, 99, 235), False, raw_pts, 1)
    ma_pts = [(int(ax_x0 + (m["episode"]/max_ep)*ax_w), int(ax_y0 - ((m["train_avg_score"]-min_s)/(max_s-min_s))*ax_h)) for m in metrics if "train_avg_score" in m]
    if len(ma_pts) > 1:
        pygame.draw.lines(surf, (96, 165, 250), False, ma_pts, 2)

    # Panel 3: Validation Score
    px, py, _, col = panels[2]
    ax_x0, ax_y0 = px + 45, py + ph - 35
    ax_w, ax_h = pw - 65, ph - 65
    val_metrics = [m for m in metrics if m.get("val_score") is not None]
    if val_metrics:
        v_scores = [m["val_score"] for m in val_metrics]
        min_v = min(v_scores + [-150.0])
        max_v = max(v_scores + [200.0])
        val_pts = [(int(ax_x0 + (m["episode"]/max_ep)*ax_w), int(ax_y0 - ((m["val_score"]-min_v)/(max_v-min_v))*ax_h)) for m in val_metrics]
        if len(val_pts) > 1:
            pygame.draw.lines(surf, col, False, val_pts, 2)
        for pt in val_pts:
            pygame.draw.circle(surf, col, pt, 4)

    # Panel 4: Pellets & Epsilon
    px, py, _, col = panels[3]
    ax_x0, ax_y0 = px + 45, py + ph - 35
    ax_w, ax_h = pw - 65, ph - 65
    pel_pts = [(int(ax_x0 + (m["episode"]/max_ep)*ax_w), int(ax_y0 - (m.get("val_pellets", 0)/100.0)*ax_h)) for m in val_metrics]
    if len(pel_pts) > 1:
        pygame.draw.lines(surf, (245, 158, 11), False, pel_pts, 2)
        for pt in pel_pts:
            pygame.draw.circle(surf, (245, 158, 11), pt, 3)
    eps_pts = [(int(ax_x0 + (m["episode"]/max_ep)*ax_w), int(ax_y0 - (m.get("epsilon", 1.0))*ax_h)) for m in metrics]
    if len(eps_pts) > 1:
        pygame.draw.lines(surf, (168, 85, 247), False, eps_pts, 2)

    pygame.image.save(surf, output_png_path)


def plot_metrics(metrics_json_path: str, output_dir: str = None):
    """Load metrics JSON and generate both SVG and PNG visualizations."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(metrics_json_path))

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    svg_path = os.path.join(output_dir, "dqn_training_figures.svg")
    png_path = os.path.join(output_dir, "dqn_training_figures.png")

    generate_svg(metrics, svg_path)
    generate_png(metrics, png_path)

    print(f"[PLOTS] Vector SVG saved to: {svg_path}")
    if os.path.exists(png_path):
        print(f"[PLOTS] Raster PNG saved to: {png_path}")


if __name__ == "__main__":
    default_json = os.path.join(os.path.dirname(__file__), "weights", "dqn_training_metrics.json")
    if os.path.exists(default_json):
        plot_metrics(default_json)
    else:
        print(f"Metrics file not found: {default_json}")
