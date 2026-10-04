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
    0.02 ms               0.01 ms                0.03 ms                    0.45 ms                   93.0 ms
    860 pts               610 pts                926 pts                    440+ pts                  360 pts
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
├── dqn_model.py                 # Root facade re-exporting rl.dqn_model
├── dqn_pacman.pt                # Root checkpoint mirror
├── requirements.txt             # Pinned dependencies
└── GEMINI.md                    # This document
```

---

## 🧠 Deep Q-Network (DQN) Specifications

The DQN subsystem implements DeepMind-inspired Deep Reinforcement Learning with domain-adapted visual perception:

### 1. Visual Single-Plane Frame Encoding (`encode_frame`)
Each time step renders the $21 \times 19$ maze into a single 2D plane using natural **Z-ordering priority**:
* **Empty Corridor**: `0.0`
* **Wall**: `-0.5`
* **Pellet**: `+0.5` (reward target)
* **Pac-Man**: `+1.0` (self agent)
* **Ghost**: `-1.0` (overrides pellet when overlapping, accurately reflecting lethal hazard)

### 2. Temporal Frame Stacking ($k=3$)
* States are represented as **3 stacked consecutive frames**: $[f_{t-2},\; f_{t-1},\; f_t]$ with tensor shape `(3, 21, 19)`.
* By computing convolutions across the stacked channels, the network naturally infers **ghost velocity, trajectory, and momentum** with zero hand-crafted physics.

### 3. Coordinate-Preserving Neural Architecture (`PacmanDQN`)
* **Input**: `(batch, 3, 21, 19)`
* **Conv 1**: `Conv2d(3, 32, kernel=3, padding=1)` + ReLU (local entity & velocity detection)
* **Conv 2**: `Conv2d(32, 64, kernel=3, padding=1)` + ReLU (corridor & intersection features)
* **Conv 3 ($1\times1$ Bottleneck)**: `Conv2d(64, 16, kernel=1)` + ReLU (compresses feature channels while maintaining $100\%$ of the $21 \times 19$ spatial coordinates—**no destructive MaxPool downsampling**)
* **Linear 1**: `Linear(6384, 128)` + ReLU
* **Linear 2**: `Linear(128, 4)` outputting Q-values for `[up, down, left, right]`
* **Total Parameters**: $\approx 838\text{k}$ (~0.45 ms CPU inference)

### 4. Double DQN & Optimization
* **Target Network**: Decouples action selection from action evaluation to eliminate maximization bias:
  $$y = r + \gamma (1 - d) Q_{\text{target}}\left(s', \arg\max_{a'} Q_{\text{policy}}(s', a')\right)$$
* **Loss**: Smooth L1 (Huber) Loss with gradient norm clipping (`max_norm = 5.0`).
* **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}$).

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
# 1. Run Automated Test Suite
python tests/test_agents.py
python tests/test_system_one.py

# 2. Run Tournament Benchmark (Head-to-head on identical seeds)
python compare_baselines.py --episodes 10 --max-moves 150

# 3. Train DQN
python rl/train_dqn.py --episodes 1200 --batch-size 64 --lr 0.0005

# 4. Generate Training Diagnostic Figures (PNG & SVG)
python rl/plot_metrics.py

# 5. Launch Interactive Visual Game Arena
python pacman_game.py
```

---

## 📐 Agent Contract (`AgentProtocol`)

All agents implement the unified interface in [`agents/base.py`](file:///c:/Users/wei.du/WorkAtTPVision/test/system_one/agents/base.py):

```python
class AgentProtocol(Protocol):
    name: str
    category: str

    def decide(
        self,
        pacman_pos: Tuple[int, int],
        ghost_positions: List[Tuple[int, int]],
        pellets: set,
        legal_moves: List[str],
        last_move: Optional[str] = None,
    ) -> DecisionResult:
        ...
```

* `legal_moves`: The agent must only select among valid corridor directions.
* `DecisionResult`: Standard payload containing `choice`, `probabilities`, `confidence`, and `latency_ms`.
