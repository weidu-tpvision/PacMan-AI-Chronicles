# 🕹️ PacMan-AI-Chronicles
> **40 Years of Artificial Intelligence Decision Paradigms in a Single Arena**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Pygame-CE](https://img.shields.io/badge/Engine-Pygame--CE-green.svg)](https://pyga.me/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Overview

What happens when you pit modern **Generative AI (Large Language Models)** against **Classical Reinforcement Learning**, **Deep Q-Networks**, and **Symbolic Heuristics** on the exact same board?

**PacMan-AI-Chronicles** is an empirical research sandbox and real-time visualization arena. It tests the provocative claim of using LLMs as *"System 1"* fast reflexive game controllers by comparing them head-to-head across **identical random seeds** against algorithms from every major era of AI history.

> [!NOTE]
> ### 🚧 Work in Progress & DQN Continuation Notice
> This project is under active research and development. In particular, the **Deep Q-Network (DQN)** subsystem is in its Phase 1 checkpoint and **actively being continued**. 
>
> While our feature-optimized RL agent currently holds the top tournament score (926.2 pts), it relies on human-engineered topological features (graph BFS, junction detection, dead-end depth). In contrast, **DQN learns its entire representation and policy end-to-end directly from raw spatial grid tensors without human inductive bias**. Because it is unconstrained by human feature ceilings, an extended DQN policy is theoretically destined to become the definitive arena champion. See the [DQN Research Roadmap](#-ongoing-research--dqn-roadmap) below.

```text

 1980s - 1990s        1990s - 2000s          2000s - 2010s               2013 - 2015                2024 - 2026
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ Expert Rules │ ──> │ Classical RL │ ──> │ Policy Optim (ES) │ ──> │    Deep RL (DQN)    │ ──> │   System 1 LLM  │
│ (Greedy BFS) │     │ (Linear TD)  │     │   (CEM Search)    │     │  (PyTorch ConvNet)  │     │  (Transformer)  │
└──────────────┘     └──────────────┘     └───────────────────┘     └─────────────────────┘     └─────────────────┘
    0.02 ms               0.01 ms                0.03 ms                    0.27 ms                   93.0 ms
    860 pts               610 pts                926 pts                    570 pts                   360 pts
```

---

## 🥊 The Contenders

| Hotkey | Agent Name | Paradigm | Technical Implementation | Inference Latency |
| :-: | :--- | :--- | :--- | :-: |
| **[1]** | **Policy-Optimized RL** | *Evolutionary / Policy Search* | Cross-Entropy Method optimizing topological graph features (Junctions, traps, BFS). | `0.03 ms` |
| **[2]** | **Deep Q-Network (DQN)** | *Deep Reinforcement Learning* | Unentangled 6-channel binary state $\rightarrow$ 2D ConvNet with 10×10 receptive field pooling $\rightarrow$ Double-DQN. | `0.26 ms` |
| **[3]** | **System 1 (LLM)** | *Foundation Model Zero-Shot* | Structured JSON / spatial reasoning via Ollama `POST /v1/systemone`. | `93.4 ms` |
| **[4]** | **Greedy Heuristic** | *Symbolic / Expert Rules* | Hand-crafted priority rules balancing BFS food seeking and ghost evasion. | `0.02 ms` |
| **[5]** | **Textbook Q-Learning** | *Classical TD-Learning* | Bellman equation updates on classic linear feature approximations. | `0.01 ms` |
| **[6]** | **Random Baseline** | *Empirical Floor* | Uniform random distribution across legal corridors. | `0.00 ms` |

---

## 📊 Empirical Tournament Results (1,000 Identical Seeds)

All agents competed on the identical 19×21 maze layout across 1,000 deterministic seeded runs (max 150 moves per episode):

```text
==========================================================================================
 Agent Name                         | Avg Score  | Avg Moves  | Pellets  | Blunder %  | Latency  
------------------------------------------------------------------------------------------
 Random Agent (Baseline)            |    -130.7  |      17.3  |     6.9  |      8.4%  |  0.00 ms 
 Greedy Heuristic                   |     570.0  |     111.0  |    77.0  |      0.0%  |  0.01 ms 
 Q-Learning (Textbook Baseline)     |     813.8  |     147.7  |   100.7  |      0.0%  |  0.01 ms 
 Deep Q-Network (PyTorch DQN)       |     570.0  |     120.0  |    57.0  |      0.0%  |  0.27 ms 
 RL (Policy Optimized)              |     926.2  |     197.2  |   111.2  |      0.0%  |  0.03 ms 
 System 1 [Heuristic Simulator]     |     360.0  |      73.0  |    56.0  |      0.0%  |  93.4 ms 
==========================================================================================
```

### Key Takeaways:
1. **The Latency Divide ($9,000\times$ Gap):** The Policy-Optimized RL agent makes decisions in **0.03 ms**, while the LLM takes **93.4 ms**. For fast real-time games, running a billion-parameter transformer per frame is radically inefficient.
2. **Topological Feature Advantage:** The Policy-Optimized RL agent achieved a score of **926.2** (+356.2 over Greedy Heuristics) by learning multi-exit junctions and dead-end trap evasion.
3. **End-to-End Visual Learning:** The Deep Q-Network learned spatial navigation from scratch in 12 minutes on CPU, achieving **570.0 pts** at 0.27 ms without any human feature engineering.

---

## 🔬 Ongoing Research & DQN Roadmap

### Why DQN Has the Highest Theoretical Ceiling
In classical reinforcement learning, policy search methods (such as the Cross-Entropy Method) can quickly converge to high scores when supplied with **human-engineered features** (`dead_end_trap`, `safe_junction`, BFS distance maps). However, this introduces a hard limitation: **the policy's intelligence is strictly bounded by human domain knowledge and representation bias**.

**The Deep Q-Network (DQN) operates on a fundamentally purer principle:**
- **Zero Human Guidance:** It receives only a raw 4-channel spatial grid tensor (walls, Pac-Man, ghosts, pellets).
- **Autonomous Representation Learning:** The convolutional filters learn their own spatial kernels for corridor recognition, proximity gradients, and escape pathways directly from Bellman temporal difference errors.
- **Unbounded Potential:** In our initial Phase 1 training run (1,200 episodes, ~12 minutes on CPU), DQN already reached **570.0 points** and **0.27 ms inference**. Because its representation capacity is vast and unconstrained by linear assumptions, extended training is expected to surpass all handcrafted heuristics.

### Phase 2 DQN Roadmap
- [x] **Unentangled Multi-Channel Frame Stacking (6 Channels)**: Separated discrete binary channels for Walls, Pellets, Pac-Man $(t, t-1)$, and Ghosts $(t, t-1)$ to eliminate ReLU sign ambiguity while capturing velocity vectors.
- [x] **Global Receptive Field Pooling & Momentum Shaping**: 10×10 pooling for global pellet perception, anti-stall loop cutoff, directional momentum preservation, and inference anti-orbit dynamic memory.
- [x] **Standardized Simulation Engine & Authentic Arcade Dynamics**: Unified single simulation engine (`core.environment.Environment`) across desktop Pygame, web arena, training, and tournaments with authentic 28/7 Chase/Scatter cycling and seeded RNG.
- [x] **Automated Training Diagnostics**: Real-time metrics logging and 4-panel visual figure generation (vector SVG and raster PNG).
- [ ] **Prioritized Experience Replay (PER)**: Transition from uniform replay buffer sampling to TD-error proportional sampling to accelerate learning on rare, critical ghost escape events.
- [ ] **Dueling DQN Architecture**: Decouple state value estimation $V(s)$ from action advantages $A(s, a)$ to stabilize Q-values in non-critical corridors.
- [ ] **Extended Training Runs**: Scale from 1,200 episodes to 5,000+ episodes with cosine learning rate scheduling.

---


## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/<YOUR_USERNAME>/PacMan-AI-Chronicles.git
cd PacMan-AI-Chronicles

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

> [!TIP]
> On the development host, a pre-configured virtual environment containing all required dependencies (`torch`, `pygame-ce`, `numpy`) is located at `C:\Users\wei.du\venv\systemone` (accessible directly via `& "C:\Users\wei.du\venv\systemone\Scripts\python.exe"`).

### 2. Launch the Interactive Pygame Arena

```bash
python pacman_game.py
```

- **Keys `[1] - [6]`:** Instantly swap the live AI controller in real-time.
- **Spacebar:** Pause / Resume the game.
- **Live HUD:** Shows real-time action probability distributions, decision confidence, and telemetry latency.

### 3. Run the Head-to-Head Tournament Benchmark

Run a reproducible tournament across all 6 agents:

```bash
# Run 100 seeded tournament episodes
python compare_baselines.py --episodes 100

# Run 1,000 seeded tournament episodes
python compare_baselines.py --episodes 1000
```

### 4. Train New Models

```bash
# Train the Deep Q-Network (PyTorch CNN)
python rl/train_dqn.py --episodes 1200

# Run Direct Policy Search (Cross-Entropy Method)
python rl/optimize_policy.py --generations 35

# Train Approximate TD Q-Learning
python rl/train_q_learning.py --episodes 1500
```

### 5. Run the Automated Test Suite

```bash
# Run full automated test discovery:
python -m unittest discover tests

# Or run individual test suites:
python tests/test_agents.py
python tests/test_system_one.py
```

---

## 🏗️ Repository Architecture

```text
PacMan-AI-Chronicles/
│
├── rl/                          # Reinforcement Learning Subsystem
│   ├── __init__.py              # Unified exports for RL models & algorithms
│   ├── dqn_model.py             # PacmanDQN CNN & 6-channel unentangled binary state encoder
│   ├── train_dqn.py             # Double-DQN training pipeline with ReplayBuffer & metrics
│   ├── plot_metrics.py          # Diagnostic figure generator (SVG vector & PNG raster)
│   ├── train_q_learning.py      # Approximate TD Q-learning trainer
│   ├── optimize_policy.py       # Direct policy search (Cross-Entropy Method / ES)
│   └── weights/                 # Checkpoints, learned weights & diagnostics
│       ├── dqn_pacman.pt        # Trained PyTorch CNN checkpoint (Peak val: 440.0)
│       ├── dqn_training.log     # Detailed milestone training telemetry log
│       ├── dqn_training_metrics.json # Per-episode training metrics (JSON)
│       ├── dqn_training_metrics.csv  # Per-episode training metrics (CSV)
│       ├── dqn_training_figures.svg  # Scalable vector diagnostic dashboard
│       ├── dqn_training_figures.png  # 4-panel raster diagnostic dashboard
│       ├── learned_q_weights.json    # Classical TD weights
│       └── learned_enhanced_weights.json # Policy-optimized weights
│
├── agents/                      # Decoupled Agent Implementations
│   ├── __init__.py              # Unified agent registry
│   ├── base.py                  # DecisionResult, AgentProtocol, softmax utilities
│   ├── random_agent.py          # Uniform random baseline agent
│   ├── greedy_agent.py          # Multi-objective heuristic planner
│   ├── q_learning_agent.py      # Linear feature Q-learning agents
│   ├── dqn_agent.py             # PyTorch CNN inference agent
│   └── system_one_agent.py      # Ollama / Jev System 1 wrapper agent
│
├── core/                        # Core Game Engine & Simulation
│   ├── __init__.py              # Core exports
│   ├── maze_data.py             # ASCII grid, dimensions, graph BFS analytics
│   └── environment.py           # Standardized headless Pac-Man environment simulator
│
├── llm/                         # System 1 / Ollama Client Subsystem
│   ├── __init__.py              # LLM client exports
│   ├── decision_client.py       # SystemOneAgent client & fallback heuristic simulator
│   └── test_client.py           # Verification & latency runner
│
├── tests/                       # Automated Test Suite
│   ├── test_agents.py           # Multi-agent decision verification test
│   └── test_system_one.py       # Ollama schema & response verification test
│
├── pacman_game.py               # Interactive Pygame visualizer and arena
├── compare_baselines.py         # Head-to-Head tournament runner
├── benchmark.py                 # Latency / throughput profiler
├── web_arena.py                 # Zero-dependency browser visualizer (HTML5/Canvas)
├── LICENSE                      # MIT Open Source License
└── requirements.txt             # Pinned runtime dependencies
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). Contributions, experiments, and PRs are welcome!
