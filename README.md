# SelfImprove-Bench

A deterministic benchmark harness for evaluating Recursive Self-Improvement (RSI)
and self-evolving agent methods across six game environments.

## Project Overview

RSI-Bench runs a method through a training (evolution) phase and then a
held-out evaluation phase, and reports multi-dimensional RSI metrics such as
score gain, replay retention, generalization, and token/time cost.

- `benchmark/` — orchestration: config loading, deterministic task/holdout
  generation, agent lifecycle, refinement/selection loop, metrics, persistence.
  Entry point: `python -m benchmark`.
- `benches/` — six games (task generation, simulator, reference scoring):
  Farm, Combat, Sweepmine, Hero's Journey, Minecraft, and Astray.
  Minecraft and Astray share the GameWorld integration.
- `rsi_methods/` — method adaptors: `ACE`, `AHE`, `HyperAgents`.
- `analyze/` — inspects a method's evolution artifacts.

Training and evaluation support three frameworks: ACE (`ace`), AHE (`ahe`),
and HyperAgents (`hyperagents`). Each can be evaluated before and after training.

The Minecraft and Astray environments are adapted, with some modifications, from
[GameWorld](https://github.com/gameworld-project/gameworld); see
[benches/gameworld/README.md](benches/gameworld/README.md) for attribution, citation,
and the research-only disclaimer that applies to the bundled game assets.

## Environment Setup

Use the following command to install the conda environment.

```bash
conda create -n SelfImprove-Bench python=3.13.12
conda activate SelfImprove-Bench

pip install -r requirements.txt
```

Install only what your target method needs.
```bash
# HyperAgents:
pip install -r rsi_methods/HyperAgents/requirements.txt

# GameWorld
pip install -e ./benches/gameworld/assets/core/code
playwright install chromium
```

- Credentials are read from environment variables and are never written to configs:
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL` — the OpenAI-compatible backends configured
    in each method `config.toml`.
- Method models (backend, model, reasoning effort) live in
  `rsi_methods/<METHOD>/config.toml`; the benchmark config's `agent.model` does not
  override them.

## Running Guide

For the six-game main experiments (ACE, AHE, HyperAgents), use the 18 complete
configurations under `configs/main/`. Run commands, initial-framework evaluation,
task groupings, trajectory analysis, and historical-protocol limitations are in
[configs/main/README.md](configs/main/README.md). Run commands from this repository
root. No external task repository is required for Minecraft or Astray.

For example, train ACE on Farm and evaluate the trained framework, then evaluate
its untrained initial framework on the same holdout tasks:

```bash
python -m benchmark configs/main/farm_ace.json
python -m benchmark configs/main/farm_ace.json --initial
```

Replace `ace` with `ahe` or `hyperagents` to run the other frameworks.

Key config fields: `seed`, `schedule` (training items), `holdout`, `agent`
(`provider`/`backend`/`model`/`timeout`/`max_attempts`), `output_dir`,
`continue_on_error`, and `game_options` (GameWorld only).

Recovery flags:

- `--max-new-train-runs N` — pause after N new committed training runs.
- `--resume` — continue the same `output_dir`.
- `--target-dir NEW_DIR` — branch into a new workspace.

See `benchmark/RESUME.md` for the full recovery contract.

Outputs under `output_dir`: `result.json` (summary + metrics), `runs/train` and
`runs/holdout`, `checkpoints/`, `tasks/`, `rsi/` (method state),
`config.resolved.json`, `task_manifest.json`, plus `gameworld/` for GameWorld runs.

Analysis and aggregation:

```bash
python -m analyze ace outputs/main/farm/ace --model MODEL --output farm_ace_analysis.json
python -m analyze ahe outputs/main/farm/ahe --model MODEL --output farm_ahe_analysis.json
python -m analyze hyper outputs/main/farm/hyperagents --model MODEL --output farm_hyperagents_analysis.json
python -m benchmark.aggregate_results --input RUN_DIRS --output-dir OUT
```

Create a readable report for one game from its existing trajectory analyses:

```bash
python -m analyze report --game farm --input farm_ace_analysis.json farm_ahe_analysis.json farm_hyperagents_analysis.json --output-dir analysis/reports/farm --model MODEL
```

This writes `report.md`, `report.json`, `nodes.csv`, and `proposals.csv`.
The report contains descriptive statistics, evidence-linked findings, untested
improvement recommendations, and hypotheses about trajectory characteristics.
Omit `--model` for offline statistics only (no API calls). Game identity is
declared by `--game`; pass only that game's trajectories. Repeat for each game.
Model interpretations are not causal proof or replacements for raw scores.
Large inputs are interpreted in node-preserving batches and then synthesized;
this requires multiple API calls when `--model` is enabled.

Calculate G0--G3 HCI from a trained run and its matching untrained evaluation:

```bash
python -m benchmark.hci --trained outputs/main/farm/ace --initial outputs/main/farm/ace_initial --output analysis/farm_ace_hci.json
```

HCI is a ratio, not a percent. The command checks task identities, uses the
pre-refinement execution-score baseline for G0, and the target game's initial
scores for G1--G3. It uses shared group references: Farm Medium/Hard
1.179/1.297, Combat 1.433/1.314, and 1 for the other four games. G3 uses the
destination game's Medium reference. Scores retain their available precision
through calculation; rounding is only for display.
See [the main experiment guide](configs/main/README.md#hci-calculation)
for task groupings and missing-data handling.

## License and third-party components

SelfImprove-Bench's own code is released under the [MIT License](LICENSE).
The repository also bundles third-party components, some with modifications. Each
keeps its original license, which governs that directory:

| Component | Path | License / terms |
|---|---|---|
| ACE | `rsi_methods/ACE/ace/` | Apache-2.0 ([LICENSE.txt](rsi_methods/ACE/ace/LICENSE.txt)); modified for this benchmark, see [MODIFICATIONS.md](rsi_methods/ACE/ace/MODIFICATIONS.md) |
| AHE | `rsi_methods/AHE/agentic-harness-engineering/` | MIT ([LICENSE](rsi_methods/AHE/agentic-harness-engineering/LICENSE)) |
| HyperAgents | `rsi_methods/HyperAgents/hyperagents/` | **CC BY-NC-SA 4.0** ([LICENSE.md](rsi_methods/HyperAgents/hyperagents/LICENSE.md)): non-commercial use only; adaptations must be shared under the same license |
| Hero's Journey | `benches/heros_journey/HerosJourney/` | MIT ([LICENSE](benches/heros_journey/HerosJourney/LICENSE)) |
| GameWorld runtime and games | `benches/gameworld/assets/` | Adapted from [GameWorld](https://github.com/gameworld-project/gameworld) with some modifications; bundled game assets are for **research and educational use only**, see [benches/gameworld/README.md](benches/gameworld/README.md) |

The MIT License does not override these terms. In particular, the HyperAgents
component and the GameWorld game assets may not be used for commercial purposes.
