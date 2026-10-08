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
> This project is under active research and development. In particular, the **Deep Q-Network (DQN)** subsystem is **actively being continued**.
>
> The feature-optimized RL agents rely on human-engineered topological features (graph BFS, junction detection, dead-end depth). In contrast, **DQN learns its entire representation and policy end-to-end directly from raw spatial grid tensors without human inductive bias**. Because it is unconstrained by human feature ceilings, an extended DQN policy has the highest theoretical ceiling in the arena. See the [DQN Research Roadmap](#-ongoing-research--dqn-roadmap) below.
>
> **Checkpoint status:** no full DQN training run has been performed since the latest DQN and environment changes (including the collision-rule fix). The shipped `rl/weights/dqn_pacman.pt` predates them, so any DQN figures in the generated artifacts are provisional. Experiment results and run parameters are deliberately kept out of the docs while the project is a work in progress.

```text

 1980s - 1990s        1990s - 2000s          2000s - 2010s               2013 - 2015                2024 - 2026
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│ Expert Rules │ ──> │ Classical RL │ ──> │ Policy Optim (ES) │ ──> │    Deep RL (DQN)    │ ──> │   System 1 LLM  │
│ (Greedy BFS) │     │ (Linear TD)  │     │   (CEM Search)    │     │  (PyTorch ConvNet)  │     │  (Transformer)  │
└──────────────┘     └──────────────┘     └───────────────────┘     └─────────────────────┘     └─────────────────┘
   µs-scale            µs-scale              µs-scale                   sub-ms, CPU-only         network round-trip
```

---

## 🥊 The Contenders

| Hotkey | Agent Name | Paradigm | Technical Implementation | Latency Class |
| :-: | :--- | :--- | :--- | :-: |
| **[1]** | **Policy-Optimized RL** | *Evolutionary / Policy Search* | Cross-Entropy Method optimizing topological graph features (Junctions, traps, BFS). | µs-scale |
| **[2]** | **Deep Q-Network (DQN)** | *Deep Reinforcement Learning* | Identity-preserving spatial state with motion, mode, stall, and horizon features $\rightarrow$ ConvNet $\rightarrow$ Dueling Double-DQN. | sub-ms, CPU-only |
| **[3]** | **System 1 (LLM)** | *Foundation Model Zero-Shot* | Structured JSON / spatial reasoning via Ollama `POST /v1/systemone`. | network round-trip (live) |
| **[4]** | **Greedy Heuristic** | *Symbolic / Expert Rules* | Hand-crafted priority rules balancing BFS food seeking and ghost evasion. | µs-scale |
| **[5]** | **Textbook Q-Learning** | *Classical TD-Learning* | Bellman equation updates on classic linear feature approximations. | µs-scale |
| **[6]** | **Random Baseline** | *Empirical Floor* | Uniform random distribution across legal corridors. | ~zero |

---

## 📊 Tournament Benchmark

All agents compete on the identical 19×21 maze across deterministic seeded **TEST** episodes (`core/seeds.py`, disjoint from the TRAIN/VAL ranges used for training and model selection).

> [!IMPORTANT]
> **Scores and run parameters are intentionally not recorded in this README.** The engine, policies, and episode horizon are still evolving, so any number pasted here would silently drift out of sync with the code. The single source of truth is the generated artifact:
> **[`results/tournament_results.json`](results/tournament_results.json)** — written by the runner below, and read back by both interactive arenas to render their score badges.

Reproduce the full benchmark on your machine:

```bash
# Seeded TEST episodes at the canonical horizon (writes results/tournament_results.json)
python compare_baselines.py

# Episode count, horizon, offline mode etc.: see --help (more episodes -> tighter CIs)
python compare_baselines.py --help
```

The report includes, per agent: mean score ±95% CI, mean moves/pellets, survival and win rates, blunder rate, illegal-move and error counts, and decision latency (System 1 latencies from the offline simulator are flagged as synthetic).

### What to look for in the numbers:
1. **The Latency Divide:** learned/heuristic policies decide in microseconds-to-sub-millisecond on CPU, while a live LLM pays a network/model round-trip on every move. Measure it with `benchmark.py` against your own model before drawing conclusions for real-time use.
2. **Feature Engineering vs. Search:** CEM policy search over graph features (dead-end traps, safe junctions, BFS distances), the textbook TD agent and the hand-tuned Greedy heuristic all encode human domain knowledge in different ways — compare their mean scores and CIs rather than assuming the optimized policy wins.
3. **End-to-End Visual Learning:** the DQN learns corridor navigation from raw spatial tensors with zero feature engineering; watch whether extended training closes the gap to the feature-based agents (see roadmap below).

---

## 🔬 Ongoing Research & DQN Roadmap

### Why DQN Has the Highest Theoretical Ceiling
In classical reinforcement learning, policy search methods (such as the Cross-Entropy Method) can quickly converge to high scores when supplied with **human-engineered features** (`dead_end_trap`, `safe_junction`, BFS distance maps). However, this introduces a hard limitation: **the policy's intelligence is strictly bounded by human domain knowledge and representation bias**.

**The Deep Q-Network (DQN) operates on a fundamentally purer principle:**
- **Spatial State Learning:** It receives symbolic maze maps with separate actor identities, motion, and environment-state features.
- **Autonomous Representation Learning:** The convolutional filters learn their own spatial kernels for corridor recognition, proximity gradients, and escape pathways directly from Bellman temporal difference errors.
- **Unbounded Potential:** Because its representation capacity is vast and unconstrained by linear assumptions, extended training is expected to surpass handcrafted heuristics. Current progress is tracked in the training artifacts (`rl/weights/dqn_training_metrics.json`, `dqn_training_figures.svg`), not in this README.

### Phase 2 DQN Roadmap
- [x] **Markov-Oriented State Encoding**: Separate current/history maps for each ghost, Pac-Man and ghost headings, scatter/chase phase, stall progress, and remaining episode horizon (30-channel encoder, see `dqn.md`).
- [x] **Global Receptive Field Pooling & Momentum Shaping**: MaxPool for global pellet perception, anti-stall loop cutoff, directional momentum preservation, and inference anti-orbit dynamic memory.
- [x] **Standardized Simulation Engine & Authentic Arcade Dynamics**: Unified single simulation engine (`core.environment.Environment`) across desktop Pygame, web arena, training, and tournaments with authentic Chase/Scatter cycling and seeded RNG.
- [x] **Automated Training Diagnostics**: Per-episode metrics logging (JSON/CSV) and 4-panel visual figure generation (vector SVG and raster PNG).
- [x] **Prioritized Experience Replay (PER)**: Replay transitions by TD-error priority with annealed importance-sampling correction.
- [x] **Dueling DQN Architecture**: Separate state value $V(s)$ and action advantages $A(s, a)$; inference remains compatible with legacy checkpoints.
- [ ] **Extended Training Run**: Longer runs with cosine learning-rate scheduling; compare best-checkpoint validation means against the feature-based agents on identical seeds before updating the shipped checkpoint.
- [x] **Training Resume Support**: full training-state checkpoints (periodic, `--stop-after`, Ctrl+C) and exact `--resume` (see Quickstart).

### Open Issues
- **Retrain all learned policies.** The collision rule was corrected (Pac-Man stepping onto a ghost's tile is a hit even if that ghost moves away). The shipped DQN checkpoint and the CEM / TD weights predate this, so their tournament scores are provisional until retrained and the tournament is regenerated.
- **Stall truncation is training-only.** DQN training ends an episode after `STALL_STEPS` steps without a pellet, so the stall plane is never seen past that value during evaluation or in the arenas.
- **Replay buffer structure.** Sampling still scans the whole buffer, and `next_state` is stored separately; a sum-tree and index-linked frame storage would cut both time and RAM.
- **Repository hygiene.** Large binary/generated artifacts (`.pt`, metrics JSON/CSV, tournament results) are committed directly; consider Git LFS or release assets.

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
> The default system Python usually lacks `torch` and `pygame-ce`; always run the scripts from the virtual environment created above.

### 2. Launch the Interactive Pygame Arena

```bash
python pacman_game.py
```

- **Keys `[1] - [6]`:** Instantly swap the live AI controller in real-time.
- **Spacebar:** Pause / Resume the game.
- **Live HUD:** Shows real-time action probability distributions, decision confidence, and telemetry latency.

### 3. Run the Head-to-Head Tournament Benchmark

Run a reproducible tournament across all agents (the six arena controllers, with DQN reported both raw and with inference heuristics):

```bash
python compare_baselines.py          # options: python compare_baselines.py --help
```

### 4. Train New Models

```bash
# Train the Deep Q-Network (PyTorch CNN)
python rl/train_dqn.py

# Run Direct Policy Search (Cross-Entropy Method)
python rl/optimize_policy.py

# Train Approximate TD Q-Learning
python rl/train_q_learning.py
```

Default hyperparameters live in each script (`--help` lists the overridable ones); they are working values, not tuned or final.

#### Resuming long DQN runs
DQN training writes a full training-state checkpoint (`rl/weights/dqn_training_state.pt`: networks, optimizer, LR schedule, replay buffer, exploration, counters, metrics and RNG streams) every `--checkpoint-every` episodes and when you press **Ctrl+C**.

```bash
python rl/train_dqn.py --stop-after 300   # train in sessions: run 300 episodes, checkpoint, exit
python rl/train_dqn.py --resume           # continue where the last session (or Ctrl+C) stopped
```

- `--resume` reuses the checkpoint's hyperparameters, so passing `--episodes`, `--lr`, etc. alongside it is rejected.
- Resuming from a periodic or `--stop-after` checkpoint reproduces the uninterrupted run exactly. A Ctrl+C checkpoint also keeps the interrupted episode's partial experience, so the result can differ slightly.
- The state file holds the whole replay buffer (up to roughly 2 GB). It is git-ignored and deleted automatically once the run completes; `dqn_pacman.pt` (best validated model) is what the agents load.
- For a run trained with a custom `--save-path`, pass the state file explicitly: `--resume --checkpoint-path <dir>/dqn_training_state.pt`.

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
│   ├── dqn_model.py             # PacmanDQN CNN & identity-preserving state encoder
│   ├── train_dqn.py             # Double-DQN training pipeline with ReplayBuffer & metrics
│   ├── plot_metrics.py          # Diagnostic figure generator (SVG vector & PNG raster)
│   ├── train_q_learning.py      # Approximate TD Q-learning trainer
│   ├── optimize_policy.py       # Direct policy search (Cross-Entropy Method / ES)
│   └── weights/                 # Checkpoints, learned weights & diagnostics
│       ├── dqn_pacman.pt        # Latest best-validation DQN checkpoint (see metrics JSON)
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
│   ├── features.py              # Shared linear-feature extractors (train/inference parity)
│   ├── paths.py                 # Checkpoint / weight path resolution
│   ├── registry.py              # Controller registry shared by both arenas
│   ├── random_agent.py          # Uniform random baseline agent
│   ├── greedy_agent.py          # Multi-objective heuristic planner
│   ├── q_learning_agent.py      # Linear feature Q-learning agents
│   ├── dqn_agent.py             # PyTorch CNN inference agent
│   └── system_one_agent.py      # Ollama / Jev System 1 wrapper agent
│
├── core/                        # Core Game Engine & Simulation
│   ├── __init__.py              # Core exports
│   ├── maze_data.py             # ASCII grid, dimensions, graph BFS analytics
│   ├── environment.py           # Standardized headless Pac-Man environment simulator
│   └── seeds.py                 # Disjoint TRAIN / VAL / TEST seed ranges & global seeding
│
├── llm/                         # System 1 / Ollama Client Subsystem
│   ├── __init__.py              # LLM client exports
│   ├── decision_client.py       # SystemOneAgent client & fallback heuristic simulator
│   └── check_client.py          # Manual live/offline smoke runner (python -m llm.check_client)
│
├── tests/                       # Automated Test Suite
│   ├── test_agents.py           # Multi-agent decision verification test
│   ├── test_environment.py      # Scatter/Chase clock, collision rule, tunnel wrap
│   ├── test_facades.py          # Root compatibility facades
│   ├── test_train_resume.py     # Exact DQN stop/resume, Ctrl+C checkpoints
│   ├── test_system_one.py       # Ollama schema & response verification test
│   └── test_review_regressions.py # Regression coverage for previously fixed bugs
│
├── pacman_game.py               # Interactive Pygame visualizer and arena
├── compare_baselines.py         # Head-to-Head tournament runner
├── benchmark.py                 # Latency / throughput profiler
├── web_arena.py                 # Zero-dependency browser visualizer (HTML5/Canvas)
├── LICENSE                      # MIT Open Source License
└── requirements.txt             # Runtime dependencies (minimum versions)
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). Contributions, experiments, and PRs are welcome!
