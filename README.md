# RSI-Bench

A benchmark for recursive self-improvement across six game environments.
RSI-Bench trains agent frameworks and evaluates their initial and final states
on shared held-out tasks, measuring score gains, retention, generalization,
and token/time costs.

- **Methods:** ACE, AHE, HyperAgents.
- **Games:** Farm, Combat, Sweepmine, Hero's Journey, Minecraft, Astray.
- **Extensible:** game environments and RSI methods use separate interfaces.

## Installation

Run the following from the repository root:

```bash
conda create -n rsi-bench python=3.13.12
conda activate rsi-bench
pip install -r requirements.txt
```

Additional dependencies depend on the method and games:

```bash
# HyperAgents
pip install -r rsi_methods/HyperAgents/requirements.txt

# Minecraft and Astray
pip install -e ./benches/gameworld/assets/core/code
playwright install chromium

# AHE (Linux; unprivileged user namespaces must also be enabled)
sudo apt-get install bubblewrap
```

AHE requires filesystem sandbox support and a successful Agent Debugger
diagnosis before each refinement. It stops if either is unavailable.
NexAU is bundled locally and installed by `requirements.txt`.

Configure model credentials through environment variables:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="YOUR_API_BASE_URL"
```

Set method-specific models and backend options in
`rsi_methods/<METHOD>/config.toml`. These settings are not overridden by
`agent.model` in benchmark configs.

## Run Experiments

The 18 configurations in [`configs/main/`](configs/main/) cover all six games
and three methods. For example:

```bash
# Train ACE on Farm and evaluate the final framework
python -m benchmark configs/main/farm_ace.json

# Evaluate the initial framework on the same held-out tasks
python -m benchmark configs/main/farm_ace.json --initial

# Resume an interrupted run
python -m benchmark configs/main/farm_ace.json --resume
```

Replace `ace` with `ahe` or `hyperagents` to use another method.
Results, trajectories, task manifests, method artifacts, and checkpoints are
saved under the configured `output_dir`.

See the [experiment guide](configs/main/README.md) for task groups and metrics,
and the [resume guide](benchmark/RESUME.md) for checkpoint recovery.

### Experiment Configuration

Experiment JSON files control tasks, execution, and outputs:

| Field | Meaning |
|---|---|
| `seed` | Seed for training task generation and method initialization. |
| `schedule` | Ordered training groups, each with `game`, `difficulty`, `count`, and optional `replay`. `count` excludes inserted replay runs. |
| `holdout` | Evaluation groups, each with `game`, `difficulty`, and `count`. These tasks do not trigger refinement. |
| `holdout_seed` | Separate seed for generating held-out tasks. |
| `holdout_tasks_per_difficulty` | Pool size per requested game/difficulty pair; must cover each holdout group's `count`. |
| `replay.every` / `replay.lag` | Insert a replay every N normal training tasks, revisiting the task L training rounds earlier. Both must be positive when a schedule group enables `replay`. |
| `agent.provider` / `agent.adapter` | Set `provider` to `rsi` and `adapter` to the method's Python import path. |
| `agent.adapter_options` | Method-specific options for external adapters. |
| `agent.timeout` / `agent.max_attempts` | Acting request timeout in seconds and maximum attempts. |
| `agent.model` | Model setting for external adapters; bundled methods use their own `config.toml`. |
| `evolution_window` | Number of early/late training scores used for evolution metrics, not the number of training rounds. |
| `output_dir` | Directory for results, trajectories, and checkpoints. Use a separate directory for each experiment. |
| `continue_on_error` | Continue after recoverable task failures; fatal method errors still stop the run. |
| `game_options` | Game-specific settings. Minecraft/Astray use `game_options.gameworld` for task lists and interaction settings. |

Game IDs are `farm`, `combat`, `sweepmine`, `heros_journey`, and `gameworld`
(Minecraft/Astray). For example, `schedule: [{"game": "farm", "difficulty": 3,
"count": 20, "replay": true}]` trains on 20 difficulty-3 Farm tasks.
With `replay: {"every": 5, "lag": 3}`, it also replays training tasks 2, 7,
12, and 17 after rounds 5, 10, 15, and 20, without refinement on those replays.

For Minecraft/Astray, `game_options.gameworld.train` and `.holdout` specify the
actual game/task IDs, and `max_steps` limits actions per episode. Start from a
matching config in `configs/main/` rather than changing only the top-level game ID.

## Analyze Results

```bash
python -m analyze ace outputs/main/farm/ace --model MODEL --output farm_ace_analysis.json
python -m analyze ahe outputs/main/farm/ahe --model MODEL --output farm_ahe_analysis.json
python -m analyze hyper outputs/main/farm/hyperagents --model MODEL --output farm_hyperagents_analysis.json
python -m benchmark.aggregate_results --input RUN_DIRS --output-dir OUT
```

Calculate HCI using a trained run and its matching initial evaluation:

```bash
python -m benchmark.hci --trained outputs/main/farm/ace --initial outputs/main/farm/ace_initial --output analysis/farm_ace_hci.json
```

Use `python -m analyze --help` for trajectory analysis and reporting commands.

## Add an RSI Method

Implement [`RSIMethod`](benchmark/interfaces.py) and configure its import path:

```json
{
  "agent": {
    "provider": "rsi",
    "adapter": "my_method.adapter:MyAdapter",
    "adapter_options": {},
    "model": "YOUR_MODEL"
  }
}
```

The adapter owns method initialization, acting, refinement, and state
persistence. Games own observations, actions, simulation, and scoring; adding
a method does not require changing game code.

See [`examples/minimal_rsi.py`](examples/minimal_rsi.py) for an offline lifecycle
example and [`benchmark/contracts.py`](benchmark/contracts.py) for shared data
types. Run the example with:

```bash
python -m benchmark configs/examples/minimal_rsi.json
```

## Repository Layout

| Directory | Contents |
|---|---|
| `benchmark/` | Experiment runner, interfaces, metrics, and checkpoints |
| `benches/` | Game environments and scoring |
| `rsi_methods/` | ACE, AHE, and HyperAgents adapters |
| `configs/` | Experiment configurations |
| `analyze/` | Trajectory analysis and reports |
| `tests/` | Regression tests |

## License

RSI-Bench's own code is released under the [MIT License](LICENSE).
Bundled third-party components retain their own licenses:

| Component | License / terms |
|---|---|
| ACE | [Apache-2.0](rsi_methods/ACE/ace/LICENSE.txt); see [modifications](rsi_methods/ACE/ace/MODIFICATIONS.md) |
| AHE | [MIT](rsi_methods/AHE/agentic-harness-engineering/LICENSE) |
| NexAU | Apache-2.0; see [bundled distribution details](third_party/nexau/README.md) |
| HyperAgents | [CC BY-NC-SA 4.0](rsi_methods/HyperAgents/hyperagents/LICENSE.md), non-commercial use only |
| Hero's Journey | [MIT](benches/heros_journey/HerosJourney/LICENSE) |
| GameWorld runtime and games | Adapted from [GameWorld](https://github.com/gameworld-project/gameworld); bundled game assets are for research and educational use only. See [attribution and terms](benches/gameworld/README.md). |

The repository's MIT License does not override third-party terms.
