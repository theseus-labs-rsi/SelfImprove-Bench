# Legacy Multimodal Game Experiments

Code-only distribution of the Minecraft and Astray experiments using ACE,
AHE and HyperAgents. The repository's original root-level implementation is
retained separately. No experiment results, trajectories, checkpoints,
generated candidates, API credentials or local Python environments are included.

## Layout

- `code/`: benchmark runner, framework adapters, multimodal evidence and retry logic.
- `gameworld/`: official evaluation games and task definitions.
- `gameworld_train_minecraft/`, `gameworld_train_astray/`: training overlays.
- `configs/templates/`: six experiment configurations and framework settings.
- `tools/`: launcher, checkpoint helpers and checkout initialization.
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

This packaging does not change framework algorithms or historical scores.
The common actor adapter supplies the same observation format, but method-owned
history, tools and internal call budgets differ. Minecraft's runtime defaults
to world seed 42; configured training seed metadata is not proof that different
worlds were executed. These limitations are not corrected by publishing code.

The launcher preserves unknown usage as unknown, and reconnect/restart attempts
can incur unrecorded costs. Step recovery is best-effort, not exactly-once API
execution. Offline tests are not a replacement for live browser/container/API
validation. No paid experiments are launched during publication.
