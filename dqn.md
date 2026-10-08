# DQN observation state: completeness plan

## Current representation

The DQN encoder in `rl/dqn_model.py` produces 30 spatial channels: walls, pellets, Pac-Man's current and previous positions, separate current and previous maps for each of three ghosts, four Pac-Man heading planes, twelve ghost heading planes, and four global planes for scatter mode, cycle phase, stall progress, and remaining horizon. This remains a symbolic grid observation rather than rendered-frame input.

## Separate object channels versus one merged map

There are two common ways to encode the object types on this grid.

### Separate binary channels

Give each object type its own 0/1 spatial map. For example, use a wall map, a pellet map, one Pac-Man map, and one map per ghost identity. Add history or heading features separately where needed.

**Advantages**

- Object type is explicit; the network does not have to infer it from an arbitrary numeric code.
- Binary values have no artificial ordering. A ghost label of `4` is not treated as “twice as much” as a pellet label of `2`.
- Different objects can occupy the same cell without one overwriting the other.
- Separate ghost maps retain ghost identity, which matters because the environment gives ghosts different behaviors.
- Adding history is straightforward: provide separate time-indexed planes for the entities whose motion matters.

**Tradeoffs**

- The input has more channels, which increases the first convolution's parameter count and computation modestly.
- Position maps for Pac-Man and ghosts are sparse. This is expected for entities that occupy only a few cells; sparsity alone does not make the representation poor.
- Static walls and dynamic objects repeat across observations. This costs input bandwidth, though it is usually small for this maze.

### One merged categorical map

Encode each cell with one integer such as `0=empty`, `1=wall`, `2=pellet`, `3=Pac-Man`, and `4..6=ghost identity`.

**Advantages**

- Uses one spatial plane and is compact to store.
- Can be convenient for visualization or as input to a model designed for categorical tokens.

**Tradeoffs**

- Feeding these integers directly to a standard CNN imposes arbitrary numeric distances and ordering between categories. The model may treat category 6 as more similar to 5 than to 2 for reasons unrelated to the game.
- A single label per cell cannot represent overlapping objects unless an explicit priority or multi-label scheme is added. Priority can hide useful information, for example at a collision state.
- Current object identity and temporal information still need a representation; merging does not solve state completeness.
- A learned embedding for categories can avoid the numeric-order problem, but it adds model complexity and still needs a way to represent multiple simultaneous categories per cell.

### Recommendation

Prefer separate binary channels for this small symbolic grid. Keep static walls and remaining pellets separate, and give each ghost its own channel. The current concern about sparse Pac-Man and ghost maps is not, by itself, a reason to merge them. If reducing channels is important, compare alternatives empirically on identical seeds and training budgets; do not assume a single numeric label plane is equivalent.

## Current DQN versus the Atari paper's DQN

The Atari paper, [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602), learned from preprocessed visual observations. It converted emulator frames to a low-resolution grayscale representation and stacked four successive frames (commonly described as an `84 × 84 × 4` input) so a feed-forward network could infer motion from image changes.

| Aspect | This Pac-Man implementation | Atari paper setup |
| --- | --- | --- |
| Observation source | Structured game state: maze cells, pellets, and actor coordinates | Emulator image frames |
| Spatial representation | Binary semantic maps on a `21 × 19` grid | Preprocessed grayscale pixels at `84 × 84` per frame |
| Temporal input | Current and previous position maps per actor plus explicit headings | Four successive image frames |
| Object identity | Each ghost has its own current, history, and heading channels | Not supplied as labels; the network must infer objects from pixels |
| Network | Small custom CNN with `3 × 3` convolutions, max pooling, and a 128-unit head; default dueling heads | CNN designed for the larger pixel input, with convolutional layers followed by a fully connected value output |
| Learning details | Double-DQN targets, dueling architecture, prioritized replay, Huber loss, legal-action masks, and project-specific reward shaping | The paper's DQN uses replay memory and a target Q-network with its reported Atari preprocessing and training setup |

The representations serve different observation settings. Atari's frame stack compensates for the fact that a single image often does not reveal object velocity. This project has exact symbolic positions, so it can provide motion and identity directly without rendering pixels. A closer symbolic analogue to Atari's temporal stack would be several recent board observations, but that is not automatically better than explicit headings and per-ghost identity channels.

The paper is a useful reference for temporal information and end-to-end visual learning, not a requirement to imitate its pixel input for a simulator that already exposes structured state. The practical comparison should be between observation variants trained from scratch under the same seeds, reward, and compute budget.

## State variables that are missing or ambiguous

### Implemented state variables

The encoder now preserves ghost identity and headings, Pac-Man heading, the scatter/chase phase, the stall counter, and remaining episode horizon. Training, validation, tournaments, and interactive frontends pass the corresponding context into the same encoder. Time-limit and stall transitions are terminal for bootstrapping because they end the finite training episode.

The 30-channel encoder changes checkpoint input semantics. Legacy six-channel checkpoints are loadable through a first-layer weight expansion for compatibility, but should be retrained before relying on their policy quality.

## Variables that do not currently need encoding

- **Ghost RNG state:** junction jitter makes transitions stochastic, but the random draw is exogenous; the agent does not need the random generator's internal state.
- **Cumulative score:** it does not affect environment dynamics or the shaped DQN reward beyond pellet, collision, and win events.
- **Illegal-move count:** DQN action selection is masked to legal moves, so this counter does not affect its normal transitions.
- **Lives:** the training loop treats a collision as terminal. The interactive game can respawn and continue, so lives matter only if the intended learned objective spans multiple lives; in that case training and observation semantics would need to change together.

## Next step

Retrain checkpoints after changing the encoder, then compare against the current baseline on identical validation seeds. Legacy weights can be expanded to load but have not learned to use the added features, so their scores are not evidence for the new representation.

## Verification checklist

- Check that two states with identical encoded observations have the same legal actions and reward semantics for each action, except for intentional stochastic ghost choices.
- Check that the encoded state distinguishes different ghost identities, scatter/chase phases, and headings when those variables change future transitions or rewards.
- Check reset and respawn observations, including Pac-Man's initial heading and ghosts' initial headings.
- Verify that replay stores the same complete observation at training and inference time.
- Run regression checks and retrain/compare before relying on policy quality.

## Relevant implementation files

- `core/environment.py` — transition rules, ghost modes/headings, collision handling, and scoring constants.
- `rl/dqn_model.py` — observation encoder and network input shape.
- `rl/train_dqn.py` — reward shaping, stall logic, replay transitions, validation, and checkpointing.
- `agents/dqn_agent.py` — inference-time state history and model input.
