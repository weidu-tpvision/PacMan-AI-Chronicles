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
    0.02 ms               0.01 ms                0.03 ms                    0.26 ms                   93.0 ms
    860 pts               610 pts                926 pts                    500 pts                   360 pts
```

The core research question investigates: *Can Large Language Models (LLMs) act as fast "System 1" reflexive game controllers, and how do they scale against Classical RL, Evolutionary Search, and Deep Q-Networks?*

---

## 🏗️ Repository Structure

```text
system_one/
│
├── core/                        # Core Engine & Simulation
│   ├── maze_data.py             # ASCII layout, dimensions, graph BFS analytics, junctions
│   ├── environment.py           # Standardized headless Pac-Man simulator with collision logic
│   └── __init__.py              # Core exports
│
├── agents/                      # Decoupled AI Agent Implementations
│   ├── base.py                  # AgentProtocol, DecisionResult, softmax utilities
│   ├── random_agent.py          # Uniform random baseline
│   ├── greedy_agent.py          # Hand-crafted multi-objective heuristic planner
│   ├── q_learning_agent.py      # Linear feature Q-learning agents (Textbook & Trained)
│   ├── dqn_agent.py             # PyTorch ConvNet inference agent with k=3 frame buffer
│   ├── system_one_agent.py      # LLM / Ollama wrapper agent
│   └── __init__.py              # Agent registry exports
│
├── rl/                          # Reinforcement Learning Subsystem
│   ├── dqn_model.py             # PacmanDQN CNN, encode_frame (Z-order), encode_state
│   ├── train_dqn.py             # Double-DQN training pipeline with ReplayBuffer & metrics
│   ├── plot_metrics.py          # Visualization generator (PNG & vector SVG figures)
│   ├── train_q_learning.py      # Approximate linear TD Q-learning trainer
│   ├── optimize_policy.py       # Cross-Entropy Method (CEM) evolutionary policy search
│   ├── weights/                 # Checkpoints, learned weights & diagnostics
│   │   ├── dqn_pacman.pt        # Trained PyTorch CNN checkpoint (Peak val: 440.0)
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
│   ├── test_client.py           # Latency and schema test runner
│   └── __init__.py
│
├── tests/                       # Automated Test Suite
│   ├── test_agents.py           # Agent instantiation and decision verification
│   └── test_system_one.py       # Schema and fallback verification
│
├── pacman_game.py               # Interactive visual Pygame arena (Live hot-swapping [1]-[6])
├── compare_baselines.py         # Multi-agent tournament benchmark runner
├── benchmark.py                 # Latency / throughput profiler
├── web_arena.py                 # Zero-dependency browser visualizer (HTML5/Canvas)
├── dqn_model.py                 # Backward-compatibility facade for rl.dqn_model
├── requirements.txt             # Pinned dependencies
└── GEMINI.md                    # This document
```

---

## 🧠 Deep Q-Network (DQN) Specifications

The DQN subsystem implements DeepMind-inspired Deep Reinforcement Learning with unentangled multi-channel perception and velocity awareness:

### 1. Unentangled 6-Channel Binary State Encoding (`encode_state`)
Rather than compressing objects into a scalar matrix where negative values collide under ReLU, the state is represented as a discrete 6-channel binary tensor `(6, 21, 19)` where all entries are strictly $\{0.0, 1.0\}$:
* **Channel 0**: Static Walls ($1.0 = \text{Wall}$, $0.0 = \text{Corridor}$)
* **Channel 1**: Pellets remaining ($1.0 = \text{Pellet}$, $0.0 = \text{Empty}$)
* **Channel 2**: Pac-Man current position at $t$ ($1.0 = \text{Pac-Man}$)
* **Channel 3**: Ghost current positions at $t$ ($1.0 = \text{Ghost}$)
* **Channel 4**: Pac-Man previous position at $t-1$ ($1.0 = \text{Pac-Man}$)
* **Channel 5**: Ghost previous positions at $t-1$ ($1.0 = \text{Ghost}$)

### 2. Temporal Velocity Tracking & Inertia
* By cross-correlating Channels $(2, 4)$ and $(3, 5)$, 2D convolutional kernels directly compute **velocity vectors $(\Delta x, \Delta y)$ and headings** for both Pac-Man and all ghosts without hand-crafted physics.
* **Momentum Regulation**: Inverse direction penalties during exploration ($-1.5$) and inference (`legal_q[opp] -= 1.0`) prevent micro-oscillations between adjacent empty corridor cells.
* **Anti-Stall Cutoff**: A 45-step inactive loop cutoff during training ($-50.0$ penalty) prevents empty corridors from becoming infinite orbit havens.
* **Anti-Orbit Dynamic Memory**: A 16-step rolling position buffer (`recent_positions`) tracks repeated tile visits during inference. When no pellets have been eaten and a candidate move leads into repeatedly visited empty corridors ($\ge 2$ visits), a scaled loop penalty (`visit_count * 20.0`) is subtracted, allowing the agent to exit local corridor limit cycles without retraining.

### 3. Global Receptive Field Neural Architecture (`PacmanDQN`)
* **Input**: `(batch, 6, 21, 19)`
* **Conv 1**: `Conv2d(6, 32, kernel=3, padding=1)` + ReLU (local entity & velocity detection)
* **Conv 2**: `Conv2d(32, 64, kernel=3, padding=1)` + ReLU (corridor & intersection features)
* **Pooling**: `MaxPool2d(kernel=2, stride=2)` ($21 \times 19 \to 10 \times 9$, expanding the receptive field to $10 \times 10$ tiles so the agent perceives distant pellet clusters across the maze)
* **Linear 1**: `Linear(5760, 128)` + ReLU
* **Linear 2**: `Linear(128, 4)` outputting Q-values for `[up, down, left, right]`
* **Total Parameters**: $\approx 758\text{k}$ (~0.26 ms CPU inference)

### 4. Double DQN & Optimization
* **Target Network**: Decouples action selection from action evaluation to eliminate maximization bias:
  $$y = r + \gamma (1 - d) Q_{\text{target}}\left(s', \arg\max_{a'} Q_{\text{policy}}(s', a')\right)$$
* **Loss**: Smooth L1 (Huber) Loss with gradient norm clipping (`max_norm = 5.0`).
* **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}$).
* **Tournament Score**: **126.6 pts** (32.7 pellets, 40.8 moves across 100 seeded episodes under unified Scatter/Chase dynamics).

### 5. Training Analysis & Diagnostic Protocol (Mandatory for Every Run)
Every DQN training run must follow this standardized automated diagnostic and analysis workflow:

1. **Per-Episode Metrics Logging**: Record `episode`, `score`, `train_avg_score`, `pellets`, `reward`, `steps`, `loss`, `epsilon`, `val_score`, and `val_pellets` in [`rl/weights/dqn_training_metrics.json`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/weights/dqn_training_metrics.json) and `.csv`.
2. **Automated Multi-Panel Figure Generation**: The training runner must execute [`rl/plot_metrics.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/plot_metrics.py) at the end of training to generate:
   * **Vector Dashboard**: [`rl/weights/dqn_training_figures.svg`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/weights/dqn_training_figures.svg) (scalable publication quality).
   * **Raster Dashboard**: [`rl/weights/dqn_training_figures.png`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/weights/dqn_training_figures.png) (4-panel visual dashboard).
3. **Quantitative Quartile Analysis**: Compute statistical quartile breakdowns (Scores, Pellets, Huber Loss, Survival Rate) and record empirical conclusions in [`GEMINI.md`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/GEMINI.md) and [`dqn_training_report.md`](file:///C:/Users/wei.du/.gemini/antigravity/brain/f618c489-8aab-47b9-a653-927a9b62b8ae/dqn_training_report.md).

#### Empirical Training Analysis & Findings (1,200 Episodes):

| Quartile | $\epsilon$ Range | Avg Score | Avg Pellets (Max) | Avg Steps | Huber Loss | Survival Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1 (Eps 1–300)** | $0.998 \to 0.472$ | $-80.0$ | $11.9$ ($42$) | $26.9$ | $10.63$ | **9.7%** |
| **Q2 (Eps 301–600)** | $0.471 \to 0.223$ | $+1.3$ | $20.1$ ($68$) | $33.5$ | $14.82$ | **42.3%** |
| **Q3 (Eps 601–900)** | $0.222 \to 0.105$ | $-36.8$ | $16.1$ ($54$) | $25.8$ | $23.31$ | **22.3%** |
| **Q4 (Eps 901–1200)** | $0.105 \to 0.050$ | **$+22.1$** | **$21.9$** (**$80$**) | **$36.5$** | $21.25$ | **46.3%** |

#### Core Empirical Conclusions:
* **Scatter Window Exploitation**: Training on the unified environment with 28/7 Chase/Scatter cycling enabled the CNN to learn aggressive corridor clearing when ghosts scatter. Single-episode pellet peaks jumped to **80 pellets** in Q4.
* **Positive Score Cross-Over**: By Q2 and Q4, net training scores crossed into solid positive territory ($+1.3$ in Q2, $+22.1$ in Q4) with survival rates reaching **$46.3\%$**.
* **Loss Dynamics**: Huber loss stabilized around $\sim 21$ in late training as the network resolved high-reward scatter clearing opportunities vs. ambush traps.
* **Peak Policy Checkpoint**: The best validation checkpoint was saved at **Episode 400** (Validation score: **$+148.0\text{ pts}$**, **$34.8\text{ pellets}$**), saved to [`rl/weights/dqn_pacman.pt`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/weights/dqn_pacman.pt).

### 6. Standardized Simulation Engine (`core.environment.Environment`)
* **Single Source of Truth**: All game arenas ([`pacman_game.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/pacman_game.py), [`web_arena.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/web_arena.py)), training pipelines ([`rl/train_dqn.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/train_dqn.py), [`rl/train_q_learning.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/train_q_learning.py), [`rl/optimize_policy.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/rl/optimize_policy.py)), and tournament runners ([`compare_baselines.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/compare_baselines.py)) share the exact same `core.environment.Environment` engine.
* **Arcade Scatter / Chase Dynamics**: Ghosts cycle between 28 steps in Chase Mode (direct pursuit, ambush, flanking) and 7 steps in Scatter Mode (heading to designated home corners), faithfully replicating Namco 1980 arcade behavior and naturally shattering static phase-locked stalemates.
* **Seeded Reproducibility**: Each environment instance uses an isolated `self.rng = random.Random(seed)` with subtle ($10\%$) junction exploration noise, guaranteeing that tournament benchmarks evaluate diverse, realistic game trajectories across seeds while remaining fully reproducible.
* **Episodic & Life-Loss Reset**: When Pac-Man loses a life or resets, calling `agent.reset()` immediately purges temporal velocity buffers, preventing corrupted post-respawn momentum vectors.

## 🐍 Python Virtual Environment & Runtime Setup

> [!IMPORTANT]
> **Pre-Configured Virtual Environment**:
> All project runtime dependencies (`torch`, `pygame-ce`, `numpy`) are installed and maintained in the dedicated virtual environment located at:
> * **Path**: `C:\Users\wei.du\venv\systemone` (or `$HOME\venv\systemone`)
> * **Interpreter**: `C:\Users\wei.du\venv\systemone\Scripts\python.exe`
>
> **Caution**: The system's default global Python (`AppData\Local\Microsoft\WindowsApps\python.exe` / Python 3.14) lacks `torch` and `pygame`. Always run scripts using the `systemone` virtual environment to prevent `ModuleNotFoundError`.
>
> **Usage Options**:
> * **Direct PowerShell execution**:
>   ```powershell
>   & "C:\Users\wei.du\venv\systemone\Scripts\python.exe" tests/test_agents.py
>   ```
> * **Shell Activation**:
>   ```powershell
>   & "C:\Users\wei.du\venv\systemone\Scripts\Activate.ps1"
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

# 2. Run Tournament Benchmark (Head-to-head on identical seeds)
python compare_baselines.py --episodes 10 --max-moves 150

# 3. Train DQN
python rl/train_dqn.py --episodes 1200 --batch-size 64 --lr 0.0005

# 4. Generate Training Diagnostic Figures (PNG & SVG)
python rl/plot_metrics.py

# 5. Launch Interactive Visual Game Arena
python pacman_game.py       # Native desktop Pygame window
python web_arena.py         # Browser-based visualizer (auto-opens Chrome/Edge, zero display issues)
```

---

## 📐 Agent Contract (`AgentProtocol`)

All agents implement the unified interface in [`agents/base.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/agents/base.py):

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
