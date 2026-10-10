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
│   ├── human_agent.py           # Manual keyboard controller with turn buffering & continuous pacing
│   ├── system_one_agent.py      # LLM / Ollama wrapper agent
│   └── __init__.py              # Agent registry exports
│
├── rl/                          # Reinforcement Learning Subsystem
│   ├── dqn_model.py             # PacmanDQN CNN and the (grid, scalars) encode_state
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
│   ├── test_dqn_encoding.py     # DQN observation layout and history-freedom
│   ├── test_environment.py      # Scatter/Chase clock, collision rule, tunnel wrap
│   ├── test_facades.py          # Root compatibility facades
│   ├── test_train_resume.py     # Exact DQN stop/resume, Ctrl+C checkpoints
│   ├── test_system_one.py       # Schema and fallback verification
│   └── test_review_regressions.py # Regression coverage for previously fixed bugs
│
├── pacman_game.py               # Interactive visual Pygame arena (Live hot-swapping [1]-[7] & manual play)
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

### 1. Markov-Oriented, History-Free State Encoding (`encode_state`)
The observation is a pair `(grid, scalars)` computed from the current game state only (see `rl/dqn_model.py` and `dqn.md` for the authoritative layout):
* **Grid planes**: walls, pellets, Pac-Man position, one position map per ghost, Pac-Man heading and per-ghost headings (one-hot, set at each actor's own tile), Scatter flag and Scatter/Chase cycle phase.
* **Scalars** (fed to the dense layer): stall progress and remaining episode horizon — they only change the value of the future, not local dynamics.
* No previous-position maps: previous position = position − heading for every actor, so headings carry the motion information.

### 2. Inertia & Loop Handling
* **Momentum Regulation**: Reversal penalties during training (`R_REVERSAL` in `rl/train_dqn.py`) and inference (`REVERSAL_PENALTY` in `agents/dqn_agent.py`) discourage micro-oscillations between adjacent empty corridor cells.
* **Stall Penalty**: Once `STALL_STEPS` steps pass without a pellet, every further step costs `R_STALL_PER_STEP`. Stalls are penalized, not truncated, so training episodes end exactly where evaluation episodes do.
* **Score-Aligned Reward**: The reward is the tournament score delta scaled by `REWARD_SCALE` plus the small shaping terms above; see `dqn.md` for the rationale (including the long discount).
* **Anti-Orbit Dynamic Memory**: A rolling position buffer (`recent_positions`) tracks repeated tile visits during inference. When no pellets have been eaten and a candidate move leads into repeatedly visited empty corridors, a penalty proportional to the visit count (`ORBIT_PENALTY_PER_VISIT`) is subtracted, allowing the agent to exit local corridor limit cycles without retraining.

### 3. Full-Resolution Neural Architecture (`PacmanDQN`)
* **Input**: grid `(batch, NUM_CHANNELS, 21, 19)` and scalars `(batch, NUM_SCALARS)`
* **Convolution stack**: `CONV_LAYERS` x (`Conv2d(3x3, padding=1)` + ReLU) at full grid resolution, **no pooling** — exact tile offsets between actors are preserved while the receptive field grows by 2 tiles per layer.
* **Channel reduction**: `Conv2d(1x1)` to `REDUCED_CHANNELS` + ReLU, keeping the dense layer small.
* **Shared feature head**: `Linear(REDUCED_CHANNELS * 21 * 19 + NUM_SCALARS, HIDDEN_UNITS)` + ReLU (flattened features concatenated with the scalars)
* **Dueling heads**: `Linear(HIDDEN_UNITS, 1)` for state value $V(s)$ and `Linear(HIDDEN_UNITS, 4)` for action advantages $A(s,a)$.
* **Checkpoint compatibility**: checkpoints from earlier encodings/networks are rejected with a "retrain" message; the agent falls back to `[UNTRAINED]` and the tournament skips the DQN rows.
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
3. **Unbiased Final Evaluation**: When a run completes, the trainer re-evaluates the best checkpoint and the final weights on VAL seeds not used for selection and writes `rl/weights/dqn_final_eval.json`; compare runs on these numbers, not on the (optimistic) selection score.
4. **Artifact-Only Results**: Training conclusions (validation means, best-checkpoint episode, quartile analyses) and run parameters live in the artifacts above — **not** in this document, so this guide can never drift from the latest run.
5. **Checkpoint Status**: No full DQN training run has been performed since the latest DQN and environment changes (including the collision-rule fix); the shipped checkpoint predates them and must be retrained before its scores are meaningful.

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
python rl/train_dqn.py --resume      # continue an interrupted / --stop-after run

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
