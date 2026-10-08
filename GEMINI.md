# 🕹️ GEMINI.md: PacMan-AI-Chronicles Context & Architecture Guide

Welcome to **PacMan-AI-Chronicles**. This document serves as the canonical context and engineering guide for AI agents and developers working across this codebase.

---

## 🌟 Project Purpose & Vision

**PacMan-AI-Chronicles** is an empirical research arena benchmarking 40 years of artificial intelligence decision paradigms under identical deterministic conditions (same maze layout, same seeds, head-to-head comparison):

```text
 1980s - 1990s        1990s - 2000s          2000s - 2010s               2013 - 2015                2024 - 2026
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ Expert Rules │ ──> │ Classical RL │ ──> │ Policy Optim (ES) │ ──> │    Deep RL (DQN)    │ ──> │   System 1 LLM  │
│ (Greedy BFS) │     │ (Linear TD)  │     │   (CEM Search)    │     │  (PyTorch ConvNet)  │     │  (Transformer)  │
└──────────────┘     └──────────────┘     └───────────────────┘     └─────────────────────┘     └─────────────────┘
   µs-scale            µs-scale              µs-scale                   sub-ms, CPU-only         network round-trip
```

> Experiment scores and timings are **not** recorded in this document — the engine and
> policies change too quickly for prose snapshots to stay true. The single sources of truth
> are the generated artifacts: `results/tournament_results.json` (tournament) and
> `rl/weights/` (DQN checkpoint, metrics JSON/CSV, training log, diagnostic figures).

The core research question investigates: *Can Large Language Models (LLMs) act as fast "System 1" reflexive game controllers, and how do they scale against Classical RL, Evolutionary Search, and Deep Q-Networks?*

---

## 🏗️ Repository Structure

```text
system_one/
│
├── core/                        # Core Engine & Simulation
│   ├── maze_data.py             # ASCII layout, dimensions, graph BFS analytics, junctions
│   ├── environment.py           # Standardized headless Pac-Man simulator with collision logic
│   ├── seeds.py                 # Disjoint TRAIN/VAL/TEST seed ranges & global seeding
│   └── __init__.py              # Core exports
│
├── agents/                      # Decoupled AI Agent Implementations
│   ├── base.py                  # AgentProtocol, DecisionResult, softmax utilities
│   ├── features.py              # Shared linear-feature extractors (train/inference parity)
│   ├── paths.py                 # Checkpoint / weight path resolution
│   ├── registry.py              # Controller registry shared by both arenas
│   ├── random_agent.py          # Uniform random baseline
│   ├── greedy_agent.py          # Hand-crafted multi-objective heuristic planner
│   ├── q_learning_agent.py      # Linear feature Q-learning agents (Textbook & Trained)
│   ├── dqn_agent.py             # PyTorch ConvNet inference agent with temporal position tracking
│   ├── system_one_agent.py      # LLM / Ollama wrapper agent
│   └── __init__.py              # Agent registry exports
│
├── rl/                          # Reinforcement Learning Subsystem
│   ├── dqn_model.py             # PacmanDQN CNN and the 30-channel encode_state
│   ├── train_dqn.py             # Double-DQN training pipeline with ReplayBuffer & metrics
│   ├── plot_metrics.py          # Visualization generator (PNG & vector SVG figures)
│   ├── train_q_learning.py      # Approximate linear TD Q-learning trainer
│   ├── optimize_policy.py       # Cross-Entropy Method (CEM) evolutionary policy search
│   ├── weights/                 # Checkpoints, learned weights & diagnostics
│   │   ├── dqn_pacman.pt        # Latest best-validation DQN checkpoint (see metrics JSON)
│   │   ├── dqn_training.log     # Detailed milestone console log
│   │   ├── dqn_training_metrics.json # Full per-episode metrics
│   │   ├── dqn_training_metrics.csv  # CSV metrics export
│   │   ├── dqn_training_figures.svg  # 4-panel vector diagnostic dashboard
│   │   ├── dqn_training_figures.png  # 4-panel raster image dashboard
│   │   ├── learned_q_weights.json    # Classical TD weights
│   │   └── learned_enhanced_weights.json # Policy-optimized weights
│   └── __init__.py
│
├── llm/                         # System 1 Subsystem
│   ├── decision_client.py       # Ollama REST client & heuristic simulator fallback
│   ├── check_client.py          # Manual live/offline smoke runner (python -m llm.check_client)
│   └── __init__.py
│
├── tests/                       # Automated Test Suite
│   ├── test_agents.py           # Agent instantiation and decision verification
│   ├── test_environment.py      # Scatter/Chase clock, collision rule, tunnel wrap
│   ├── test_facades.py          # Root compatibility facades
│   ├── test_system_one.py       # Schema and fallback verification
│   └── test_review_regressions.py # Regression coverage for previously fixed bugs
│
├── pacman_game.py               # Interactive visual Pygame arena (Live hot-swapping [1]-[6])
├── compare_baselines.py         # Multi-agent tournament benchmark runner
├── benchmark.py                 # Latency / throughput profiler
├── web_arena.py                 # Zero-dependency browser visualizer (HTML5/Canvas)
├── dqn_model.py                 # Backward-compatibility facade for rl.dqn_model
├── requirements.txt             # Runtime dependencies (minimum versions)
└── GEMINI.md                    # This document
```

---

## 🧠 Deep Q-Network (DQN) Specifications

The DQN subsystem implements DeepMind-inspired Deep Reinforcement Learning with unentangled multi-channel perception and velocity awareness:

### 1. Markov-Oriented 30-Channel State Encoding (`encode_state`)
Rather than compressing objects into a scalar matrix where negative values collide under ReLU, the state is a 30-channel tensor `(30, 21, 19)` built from binary spatial maps plus normalized scalar planes (see `rl/dqn_model.py` and `dqn.md` for the authoritative description):
* **Channel 0**: Static Walls ($1.0 = \text{Wall}$, $0.0 = \text{Corridor}$)
* **Channel 1**: Pellets remaining ($1.0 = \text{Pellet}$, $0.0 = \text{Empty}$)
* **Channels 2–3**: Pac-Man current position at $t$ and previous position at $t-1$
* **Channels 4–9**: Per-ghost current and previous position maps (identity-preserving: one pair per ghost, since ghosts have distinct behaviors)
* **Channels 10–13**: Pac-Man heading (one-hot `up/down/left/right` planes)
* **Channels 14–25**: Per-ghost heading one-hot planes (4 per ghost)
* **Channels 26–29**: Global scalar planes — scatter-mode flag, Scatter/Chase cycle phase, stall progress, remaining episode horizon

### 2. Temporal Velocity Tracking & Inertia
* By cross-correlating current/previous position pairs — Pac-Man $(2, 3)$ and ghosts $(4, 7)$, $(5, 8)$, $(6, 9)$ — 2D convolutional kernels can compute **velocity vectors $(\Delta x, \Delta y)$**; the heading planes (10–25) supply directions explicitly.
* **Momentum Regulation**: Reversal penalties during training (`R_REVERSAL` in `rl/train_dqn.py`) and inference (`REVERSAL_PENALTY` in `agents/dqn_agent.py`) discourage micro-oscillations between adjacent empty corridor cells.
* **Anti-Stall Cutoff**: During training, `STALL_STEPS` consecutive steps without a pellet end the episode with a penalty, so empty corridors cannot become infinite orbit havens.
* **Anti-Orbit Dynamic Memory**: A rolling position buffer (`recent_positions`) tracks repeated tile visits during inference. When no pellets have been eaten and a candidate move leads into repeatedly visited empty corridors, a penalty proportional to the visit count (`ORBIT_PENALTY_PER_VISIT`) is subtracted, allowing the agent to exit local corridor limit cycles without retraining.

### 3. Global Receptive Field Neural Architecture (`PacmanDQN`)
* **Input**: `(batch, 30, 21, 19)`
* **Conv 1**: `Conv2d(30, 32, kernel=3, padding=1)` + ReLU (local entity & velocity detection)
* **Conv 2**: `Conv2d(32, 64, kernel=3, padding=1)` + ReLU (corridor & intersection features)
* **Pooling**: `MaxPool2d(kernel=2, stride=2)` ($21 \times 19 \to 10 \times 9$, expanding the receptive field so the agent perceives distant pellet clusters across the maze)
* **Shared feature head**: `Linear(5760, 128)` + ReLU
* **Dueling heads**: `Linear(128, 1)` for state value $V(s)$ and `Linear(128, 4)` for action advantages $A(s,a)$.
* **Aggregation**: $Q(s,a)=V(s)+A(s,a)-\operatorname{mean}_{a'}A(s,a')$, outputting Q-values for `[up, down, left, right]`.

### 4. Double DQN & Optimization
* **Target Network**: Decouples action selection from action evaluation to eliminate maximization bias; next-action selection is masked to legal moves:
  $$y = r + \gamma (1 - d) Q_{\text{target}}\left(s', \arg\max_{a' \in A_{\text{legal}}(s')} Q_{\text{policy}}(s', a')\right)$$
* **Prioritized replay**: Samples transitions in proportion to $(|\delta|+\epsilon)^{\alpha}$ and applies annealed importance-sampling weights to the per-item Huber loss.
* **Loss**: Smooth L1 (Huber) Loss with gradient norm clipping.
* **Optimizer**: Adam with cosine learning-rate annealing over the requested episode count.
* **Hyperparameters**: defaults live in `rl/train_dqn.py` (`train_dqn()` signature and module constants). They are working values, not tuned or final — do not copy them into this document.
* **Tournament scores**: see `results/tournament_results.json` (regenerate with `compare_baselines.py`); never record them here.

### 5. Training Diagnostic Protocol (Mandatory for Every Run)
Every DQN training run must follow this standardized automated diagnostic workflow:

1. **Per-Episode Metrics Logging**: Record `episode`, `score`, `train_avg_score`, `pellets`, `reward`, `steps`, `loss`, `epsilon`, `learning_rate`, `val_score`, and `val_pellets` in [`rl/weights/dqn_training_metrics.json`](rl/weights/dqn_training_metrics.json) and `.csv`.
2. **Automated Multi-Panel Figure Generation**: The training runner must execute [`rl/plot_metrics.py`](rl/plot_metrics.py) at the end of training to generate:
   * **Vector Dashboard**: [`rl/weights/dqn_training_figures.svg`](rl/weights/dqn_training_figures.svg) (scalable publication quality).
   * **Raster Dashboard**: [`rl/weights/dqn_training_figures.png`](rl/weights/dqn_training_figures.png) (4-panel visual dashboard).
3. **Artifact-Only Results**: Training conclusions (validation means, best-checkpoint episode, quartile analyses) and run parameters live in the artifacts above — **not** in this document, so this guide can never drift from the latest run.
4. **Checkpoint Status**: No full DQN training run has been performed since the latest DQN and environment changes (including the collision-rule fix); the shipped checkpoint predates them and must be retrained before its scores are meaningful.

### 6. Standardized Simulation Engine (`core.environment.Environment`)
* **Single Source of Truth**: All game arenas ([`pacman_game.py`](pacman_game.py), [`web_arena.py`](web_arena.py)), training pipelines ([`rl/train_dqn.py`](rl/train_dqn.py), [`rl/train_q_learning.py`](rl/train_q_learning.py), [`rl/optimize_policy.py`](rl/optimize_policy.py)), and tournament runners ([`compare_baselines.py`](compare_baselines.py)) share the exact same `core.environment.Environment` engine.
* **Arcade Scatter / Chase Dynamics**: Ghosts cycle between Chase Mode (`CHASE_STEPS`: direct pursuit, ambush, flanking) and Scatter Mode (`SCATTER_STEPS`: heading to designated home corners), faithfully replicating Namco 1980 arcade behavior and naturally shattering static phase-locked stalemates.
* **Seeded Reproducibility**: Each environment instance uses an isolated `self.rng = random.Random(seed)` with subtle ($10\%$) junction exploration noise, guaranteeing that tournament benchmarks evaluate diverse, realistic game trajectories across seeds while remaining fully reproducible.
* **Episodic & Life-Loss Reset**: When Pac-Man loses a life or resets, calling `agent.reset()` immediately purges temporal velocity buffers, preventing corrupted post-respawn momentum vectors.

## 🐍 Python Virtual Environment & Runtime Setup

> [!IMPORTANT]
> **Pre-Configured Virtual Environment**:
> All project runtime dependencies (`torch`, `pygame-ce`, `numpy`) are installed and maintained in the dedicated virtual environment located at:
> * **Path**: `$HOME\venv\systemone`
> * **Interpreter**: `$HOME\venv\systemone\Scripts\python.exe`
>
> **Caution**: The system's default global Python (`AppData\Local\Microsoft\WindowsApps\python.exe` / Python 3.14) lacks `torch` and `pygame`. Always run scripts using the `systemone` virtual environment to prevent `ModuleNotFoundError`.
>
> **Usage Options**:
> * **Direct PowerShell execution**:
>   ```powershell
>   & "$HOME\venv\systemone\Scripts\python.exe" tests/test_agents.py
>   ```
> * **Shell Activation**:
>   ```powershell
>   & "$HOME\venv\systemone\Scripts\Activate.ps1"
>   python tests/test_agents.py
>   ```

---

## 🛠️ Common Developer Commands

All commands should be executed within the `systemone` virtual environment:

```bash
# 1. Run Automated Test Suite (standard unittest discovery)
python -m unittest discover tests

# Or run individual test modules:
python tests/test_agents.py
python tests/test_system_one.py

# 2. Run Tournament Benchmark (head-to-head on identical TEST seeds at the canonical DEFAULT_MAX_STEPS horizon)
python compare_baselines.py --offline     # options: --help

# 3. Train DQN (defaults in rl/train_dqn.py; options: --help)
python rl/train_dqn.py

# 4. Generate Training Diagnostic Figures (PNG & SVG)
python rl/plot_metrics.py

# 5. Launch Interactive Visual Game Arena
python pacman_game.py       # Native desktop Pygame window
python web_arena.py         # Browser-based visualizer (auto-opens Chrome/Edge, zero display issues)
```

---

## 📐 Agent Contract (`AgentProtocol`)

All agents implement the unified interface in [`agents/base.py`](agents/base.py):

```python
class AgentProtocol(Protocol):
    name: str
    category: str

    def reset(self) -> None:
        """Reset internal temporal / episodic state if applicable."""
        ...

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: Set[Tuple[int, int]],
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        ...
```

* `reset()`: Lifecycle hook called between game episodes to clear temporal velocity buffers (`DQNAgent`) or session metrics.
* `legal_moves`: The agent must only select among valid corridor directions.
* `DecisionResult`: Standard dataclass containing `choice`, `probabilities`, `confidence`, `latency_ms`, `is_live`, and optional `raw_response` and `error_msg`.
