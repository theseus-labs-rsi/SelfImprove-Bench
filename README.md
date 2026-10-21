# RSI-Bench

A benchmark for recursive self-improvement across six games: Farm, Combat,
Sweepmine, Hero's Journey, Minecraft, and Astray. Built-in methods are ACE, AHE,
and HyperAgents.

## Installation

Run from the repository root:

```bash
conda create -n SelfImprove-Bench python=3.13.12
conda activate SelfImprove-Bench
pip install -r requirements.txt
```

Install additional dependencies for the components you use:

```bash
# HyperAgents (also requires a running Docker daemon)
pip install -r rsi_methods/HyperAgents/requirements.txt

# Minecraft and Astray
pip install -e ./benches/gameworld/assets/core/code
playwright install chromium
```

AHE requires Linux, `bubblewrap` (`bwrap`), and enabled unprivileged user
namespaces. Missing sandbox support or a failed Debugger diagnosis stops the run.
Keep the pinned dependency versions. NexAU is supplied as a local wheel; see
[package provenance](third_party/nexau/README.md).

## Configuration

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="YOUR_API_BASE_URL"
```

Configure built-in models and reasoning effort in
`rsi_methods/{ACE,AHE,HyperAgents}/config.toml`. The benchmark's `agent.model`
does not override these method-specific settings.

The 18 configurations in `configs/main/` define seeds, training schedules,
holdout tasks, timeouts, and output directories. See the
[experiment guide](configs/main/README.md) for task groups and scoring.

## Running

Train and evaluate a framework, or evaluate its untrained initial state:

```bash
python -m benchmark configs/main/farm_ace.json
python -m benchmark configs/main/farm_ace.json --initial
```

Replace `farm` with another game and `ace` with `ahe` or `hyperagents`.
Training uses 20 evolution rounds plus scheduled replays. Replay and holdout
tasks do not trigger refinement. Final evaluation uses the framework retained
after the last training round, not the historical highest-scoring framework.

AHE advances valid candidates even when their score decreases. HyperAgents
uses archive-based parent sampling during training; the final round retains
its valid candidate, or its parent if the candidate is invalid or unchanged.

Resume without changing the run's configuration or dependencies:

```bash
python -m benchmark configs/main/farm_ace.json --resume
```

Outputs include `result.json`, per-task trajectories, checkpoints, and method
artifacts under `output_dir`. See [recovery rules](benchmark/RESUME.md) for pause
and branching options. Incompatible historical checkpoints require their
matching code or a new run.

## Analysis

```bash
# Trajectory classification (requires a model API); readers: ace, ahe, hyper
python -m analyze ace outputs/main/farm/ace --model MODEL --output farm_ace_analysis.json

# HCI from matching trained and initial evaluations (offline)
python -m benchmark.hci --trained outputs/main/farm/ace --initial outputs/main/farm/ace_initial --output farm_ace_hci.json
```

See the [experiment guide](configs/main/README.md) for reporting and aggregation.

## Adding a Method

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

The adapter receives `MethodContext` and implements `init`, `load`, `refine`,
`export_state`, `restore_state`, and `close`. Its acting agent implements
`act(prompt, *, image_paths=())`, returning task-facing action text in
`BackendResponse`. Types are defined in
[`benchmark/contracts.py`](benchmark/contracts.py).

Games own observations, legal actions, and scoring; methods own improvement
logic. New methods need no game or runner edits. Optional candidate interfaces
support same-task validation. See [`examples/minimal_rsi.py`](examples/minimal_rsi.py)
for an offline lifecycle example, not a learning baseline:

```bash
python -m benchmark configs/examples/minimal_rsi.json
```

## Licenses

RSI-Bench's own code uses the [MIT License](LICENSE). Bundled components retain
their original terms:

| Component | License |
|---|---|
| ACE | [Apache-2.0](rsi_methods/ACE/ace/LICENSE.txt); see [modifications](rsi_methods/ACE/ace/MODIFICATIONS.md) |
| AHE | [MIT](rsi_methods/AHE/agentic-harness-engineering/LICENSE) |
| NexAU | [Apache-2.0](third_party/nexau/LICENSE) |
| HyperAgents | [CC BY-NC-SA 4.0](rsi_methods/HyperAgents/hyperagents/LICENSE.md) |
| Hero's Journey | [MIT](benches/heros_journey/HerosJourney/LICENSE) |
| GameWorld and game assets | Research and educational use only; see [attribution and terms](benches/gameworld/README.md) |

The repository's MIT License does not override third-party restrictions.
