# Legacy Multimodal Game Experiments

> GameWorld 已接入仓库根目录的 `benchmark` / `rsi_methods`。
> 活跃资产与运行工具位于 `benches/gameworld/`，配置模板仍在本目录的 `configs/templates/`。
> 配置导入路径统一留待讨论，请使用自己准备的本地配置执行 `python -m benchmark CONFIG`；接入说明见
> [GameWorld README](../../benches/gameworld/README.md)，恢复说明见
> [RESUME](../../benchmark/RESUME.md)。下文是导入来源的历史记录，旧 `run.sh`、
> `tools/launch.py`、`prepare_checkout.py` 不再作为当前运行入口。
> 历史重复实现保留作比对资料；其中移动或删除的目录不应再被历史脚本调用。

Code-only distribution of the Minecraft and Astray experiments using ACE,
AHE and HyperAgents. The repository's original root-level implementation is
retained separately. No experiment results, trajectories, checkpoints,
generated candidates, API credentials or local Python environments are included.

## Layout

- `code/`: benchmark runner, framework adapters, multimodal evidence and retry logic.
- `code/benches/gameworld/assets/core/`: one shared GameWorld implementation.
- `code/benches/gameworld/assets/tasks/train/`: Astray and Minecraft training YAMLs.
- `code/benches/gameworld/assets/tasks/holdout/`: original evaluation task YAMLs.
- `configs/templates/`: six experiment configurations and framework settings.
- `code/benches/gameworld/utils/`: runtime step storage and browser recovery helpers.
- `tools/`: launcher, checkout initialization and the offline step-resume checker.
- `tests/`: offline regression tests.
- `reference/code/`: limited historical source fixtures used by regression tests.
- `provenance/`: source hashes, not experiment outcomes.

Bundled upstream source and game assets retain their original license files.
Game rights notices are retained; inclusion does not grant a blanket license
or relicense third-party assets. Consult each game's `RIGHTS.md` before reuse.
Unrelated upstream paper-review data and historical local run configurations
are excluded.

## Setup

From this directory, on Linux with Python 3.13 and Node.js available:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.runtime.txt
.venv/bin/python -m playwright install --with-deps chromium
.venv/bin/python tools/prepare_checkout.py
./run.sh preflight
./run.sh test
```

Alternatively, set `MM_PYTHON` to an existing compatible interpreter.
Initialization resolves `${RSI_ROOT}` in config templates and writes ignored
local configurations plus a release manifest. It refuses existing experiment
checkpoints and never rewrites a live lineage. Moving an initialized checkout
requires reinitialization before creating any runs.

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in the process environment or use
an untracked file referenced by `RSI_KEY_FILE`. Do not commit credentials.
The experiment model preset requests `gpt-5.6-sol`; access to a compatible
provider serving this model is required.

HyperAgents additionally requires Docker and its runtime image. Set
`DOCKER_HOST` or `RSI_HYPERAGENTS_DOCKER_SOCK` for your daemon. The inherited
HyperAgents build helpers manage the image; Docker socket access is required.

## Execution

Training and evaluation incur provider charges and require explicit opt-in:

```bash
./run.sh train minecraft ace --allow-paid
./run.sh train minecraft ace --rounds 19 --allow-paid
./run.sh holdout minecraft ace --allow-paid
./run.sh status minecraft ace
```

Replace `minecraft` with `astray` and `ace` with `ahe` or `hyperagents`.
Each configuration uses 20 training tasks followed by five official holdout
tasks, at most 100 game interactions per episode, low reasoning effort and
one candidate per refinement. Holdout requires a completed 20-round agent.
Start with a single training round and inspect it before committing to a full run.

## Boundaries

Training and holdout use the same `gameworld_root`. The configuration selects
their YAML directories independently through `game_options.gameworld.task_assets_roots`
(`train` and `holdout`); each generated task records its resolved `task_assets_root`.
The shared Astray core accepts an optional `train_seed` URL parameter. Without
it, seed 42 and the original evaluation rules remain the defaults. Minecraft
keeps its original fixed world seed and training quota variants.

Both task execution and step recovery load the same YAML. Step recovery binds
the task, game API and page source hashes; pre-reorganization checkpoints and
absolute paths are not automatically migrated. Task-level checkpoint behavior
is unchanged by this asset reorganization.

This packaging does not change framework algorithms or historical scores.
The common actor adapter supplies the same observation format, but method-owned
history, tools and internal call budgets differ. Minecraft's runtime defaults
to world seed 42; configured training seed metadata is not proof that different
worlds were executed. These limitations are not corrected by publishing code.

The launcher preserves unknown usage as unknown, and reconnect/restart attempts
can incur unrecorded costs. Step recovery is best-effort, not exactly-once API
execution. Offline tests are not a replacement for live browser/container/API
validation. No paid experiments are launched during publication.
