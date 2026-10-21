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

Built-in frameworks are ACE (`ace`), AHE (`ahe`), and HyperAgents (`hyperagents`).
External RSI methods use the public adapter interface described below, without
changing game implementations or the runner. Each supports before/after evaluation.

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

Run installation from this repository root. `requirements.txt` installs the
bundled NexAU 0.3.9 wheel under `third_party/nexau/`; access to its upstream Git
repository is not required. This is an RSI-Bench repack of a locally installed
distribution, not an official upstream release artifact. Package files were
verified against the installed RECORD before repacking; see
[provenance and redistribution details](third_party/nexau/README.md).

For AHE, install Linux `bubblewrap` (`bwrap`, e.g. `apt-get install bubblewrap`)
and enable unprivileged user namespaces. Actor, Debugger, and Evolver processes
run inside an allowlisted filesystem namespace; missing sandbox support stops
the run rather than falling back to an unrestricted local process.
Python/system dependencies and method code are read-only. Actors receive only
their harness and their own writable action artifacts. Debuggers receive the
current training evidence read-only. Evolvers can persist changes only to the
candidate harness and explicitly mounted output files; training evidence and
prior change summaries are read-only. Game implementations, holdout results,
and unrelated host files are not mounted. Child shell commands inherit these
filesystem restrictions. Networking remains enabled for model APIs; this is
filesystem isolation, not a complete network or hostile-code security sandbox.

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

AHE requires a successful Agent Debugger diagnosis before every refinement.
The adapter runs the bundled `agent_debugger_core` CLI with the same Python
interpreter, rather than relying on an `adb` executable on `PATH`. It passes
the method's model and backend environment to the Debugger and sets `AHE_HOME`
to the bundled AHE source. In benchmark runs, the method-scoped LLM environment
takes precedence over interactive `adb config` / `QA_*` settings.
Missing dependencies or configuration, a timeout, or an
invalid/empty diagnosis stops training even with `continue_on_error: true`;
there is no raw-trace-only fallback. Historical runs are not repaired by this
change and must not be relabeled as Debugger-enabled runs.
Keep the pinned `ag-ui-protocol` version: version 1.0 removes event classes
imported by the bundled context-compaction middleware and prevents Debugger
construction.

Debugger provider usage is exported with the diagnosis and saved as
`input/analysis/debugger_usage_snapshot.json` in each training iteration.
`ahe_refine.json` sums that diagnostic usage once plus all candidate Evolver
usage. Missing provider usage leaves the total unknown, never zero or a
partial total labeled complete. This does not recover costs of historical runs.

AHE's active-lineage policy follows the upstream default evolution loop
(audited against upstream commit `8b2a55d97590363fe50c3cc6b5e833b020a4bb4c`):
valid candidates advance even if their paired score decreases; failed candidate
executions retain the parent. The default is one candidate. Multiple requested
variants use benchmark-owned selection, not a recreation of upstream's optional
full-dataset best-of-N mode. The final agent is the active committed workspace.
The original Evolve prompt and context-compaction middleware are retained,
with runtime adaptations for continuous scores, evidence paths, and screenshots.
Only the task harness is writable, not the Evolve Agent itself.

`evolution_history.{json,md}` links each round to its candidate manifests and
`change_evaluation.json`. Comparisons are same-task parent/candidate deltas,
not per-change causal estimates. Unlike upstream's fixed evaluation dataset,
RSI-Bench supplies a changing training task per round, so no global best-ever
ranking or cross-task pass/fail-flip attribution is inferred. Native NexAU traces
remain available beside the reconstructed whole-game conversation.
New checkpoints record this policy and history; old-policy checkpoints are
rejected to avoid mixing incompatible evolution rules in one lineage.
The filesystem-isolated policy also rejects older non-isolated checkpoints;
start a new run instead of silently changing the protocol during resume.

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

This writes a short `report.md` and the statistics-only model input
`statistics.json`. Detailed audit exports remain in `report.json`, `nodes.csv`,
and `proposals.csv`; these raw evidence exports are **not** sent to the report model.
The model interprets classification counts/proportions and score statistics only,
with findings and possible framework characteristics, not concrete agent changes.
Add `--score-stats summary.csv category_stats.csv farm_ace_hci.json` to include
adjacent-round summaries, category/score associations, or `benchmark.hci` outputs.
Imported scores and HCI are preserved, not recomputed by the report command.
Omit `--model` for offline statistics only (no API calls). Game identity is
declared by `--game`; pass only that game's trajectories. Repeat for each game.
Model interpretations are not causal proof or replacements for raw scores.
Interpretation uses one statistics-only API request; oversized summaries are
rejected before an API call. No proposal text, code or raw logs enter that request.

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

## Adding an RSI Method

**Integration point:** implement an adapter following
[`RSIMethod`](benchmark/interfaces.py), then set `agent.provider` to `rsi` and
`agent.adapter` to its `package.module:AdapterClass` import path. The package
must be importable in your Python environment. No registry or game edits are
required. Import paths execute Python code; use only adapters you trust.

Games consume task-facing action text, not method-specific response envelopes.
Adapters must unwrap their own envelopes before returning `BackendResponse`;
they can retain the original model output in `metadata["raw_response_text"]`.
All methods use the same GameWorld client for screenshot input, action parsing,
retry accounting and interaction logging. The old named clients remain only as
constructor/import compatibility aliases. AHE step recovery uses the shared
agent handle without changing the stored checkpoint format.

The three bundled methods use this same interface and configuration in all
18 main experiment configs, not a separate construction path:

| Method | `agent.adapter` |
|---|---|
| ACE | `rsi_methods.ACE.ACEAdaptor:ACEAdaptor` |
| AHE | `rsi_methods.AHE.AHEAdaptor:AHEAdaptor` |
| HyperAgents | `rsi_methods.HyperAgents.HyperAgentsAdaptor:HyperAgentsAdaptor` |

For example, ACE uses `"provider": "rsi"` together with the ACE import path
above. The old `ace`, `ahe`, and `hyperagents` provider names remain aliases
for the same adapter factory and lifecycle. Method names in result directories,
GameWorld task manifests and HCI reports remain `ace`, `ahe`, and `hyperagents`.
Bundled methods keep their existing model/prompt settings in their respective
`rsi_methods/<method>/config.toml`; they currently require empty
`adapter_options`. The generic `agent.model` is available to external adapters
but does not override the bundled methods' per-role TOML settings.
Seeds, task schedules and scoring are unchanged. Method-specific search policies
are described below.

### HyperAgents Restoration (2026-10-21)

Fresh candidates use the upstream one-line MetaAgent editing instruction and
the upstream text `<json>` tool protocol. The runtime overlay retains API
transport, usage accounting, and explicit budget controls, without adding a
tool-use system instruction. Evaluation manifests describe fields and paths,
not an optimization strategy.
The initial candidate README also restores the upstream description of improving
the system's own agent-generating mechanisms; only factual game evidence paths
are appended.

Game evidence remains an adaptation: up to eight raw screenshots by default
are attached directly to the editing conversation, labeled by case and step.
`RSI_MAX_REFINEMENT_IMAGES` controls sampling. There is no separate screenshot
analysis call. Game task prompts and scoring are not changed.

The active adapter retains every candidate returned for evaluation in a persistent archive,
including invalid candidates (not eligible as parents) and valid candidates whose
score regresses. The next parent is sampled from all valid evaluated nodes using
the upstream CLI default `score_child_prop`, including its child-count penalty.
`refine.parent_selection` also supports `random`, `latest`, `best`, and `score_prop`.
Training feedback, candidate evaluations and editing logs are available to the
MetaAgent as an allowlisted history; holdout results are not included. Checkpoints
save every branch, patch digests and the sampler's random state. Fresh archive
sampling uses the benchmark `seed`; resume restores the saved random state.

Runtime overlays are installed on the initial baseline before applying lineage
patches. Subsequent edits to `meta_agent.py`, `agent/llm_withtools.py` and
`agent/tools/bash.py` survive reconstruction. Benchmark transport and contracts
remain external fixed infrastructure.

Upstream reference: `facebookresearch/HyperAgents` commit
`59a68f672dfb92c74aeb7e61535d776fb36e172d` (`generate_loop.py` and
`utils/gl_utils.py`). This restores the default individual-agent archive search,
not the optional ensemble or evolved parent-selector modes. RSI-Bench still owns
the game loop, paired same-task candidate evaluation, score normalization, API
transport, multimodal evidence and resource limits. Archived scores come from
the observed training tasks, not a newly added fixed upstream validation suite.
Because the benchmark requires one final agent rather than an archive, the final
round exports that round's valid candidate, retaining its parent if the candidate
is invalid or unchanged; it does not select the historical highest score. Selection reports
distinguish the archived candidate from the selected next parent.
Trajectory analysis preserves that distinction: `accepted` means candidate
admission to the archive, while `selected` means selection as the next parent.
An admitted child can have `accepted: true` and `selected: false`. Its score
still comes from its own saved candidate evaluation, not the sampled parent's
score. Re-run analysis of existing artifacts to update previously generated
statistics; saved analysis files are not retroactively rewritten.
If the next parent is an older branch not evaluated on the current task,
`post_refinement_score` is unavailable, rather than borrowed from another task
or the newly evaluated child. Holdout scores still come from actual evaluation.

Use a new run to evaluate the restored seed. Historical checkpoints with the old
single-lineage search policy are rejected rather than silently migrated;
historical scores are not reclassified. No paid training was run for this change.

The adapter factory receives one `MethodContext` containing the run-local
`artifact_dir`, `task_count`, `options`, `model`, `timeout`, `seed`, and `resuming` flag.
Construction must not initialize/train/reset the method: the harness calls
`init()` for fresh runs or `restore_state()` for checkpoint resumes.

| Interface | Responsibility |
|---|---|
| `init()` | Initialize the untrained method state. |
| `load()` | Return a fresh acting session for the currently committed method state. Do not train or change committed state. |
| `agent.act(prompt, *, image_paths=())` | Return `BackendResponse` containing action text, optional reasoning and token usage. Minecraft/Astray supply screenshot paths; other games supply text. |
| `refine(feedback, *, output_dir)` | Improve from `list[FeedbackResult]`. Commit internally and return `None`, or return `CandidateBatch` for harness evaluation. |
| `export_state()` | Return a JSON object fully describing the committed method state. |
| `restore_state(state)` | Restore precisely that state, without training or making model calls. |
| `close()` | Release resources; it must be safe to call repeatedly, including between tasks. |

`BackendResponse` and `FeedbackResult` are defined in
[`benchmark/contracts.py`](benchmark/contracts.py). Feedback includes the task,
step trajectory, numeric score, execution status, and diagnostic metrics.
Use `evaluation.score` for numeric performance; success and execution status are
not interchangeable with score. Action JSON remains game-specific and is
specified in the game-built prompt: Farm uses a full plan, Combat stage
decisions, Sweepmine a single action, Hero's Journey its configured interaction
protocol, and Minecraft/Astray registered GameWorld actions.

The separation is:

```text
Your method -> RSIMethod / ActingAgent -> runner -> game bench -> score + trajectory
```

Games own observation construction, legal actions, simulation, and scoring.
[`benchmark/game_adapters.py`](benchmark/game_adapters.py) is the shared game-side
evaluation entry point; it does not initialize or improve methods. The runner
calls `refine` only on normal training tasks, never on replay or holdout tasks.
Method internals (prompts, memory, code, model weights, selection strategy) belong
to the adapter. Avoid leaking per-episode execution state across `load()` calls;
if parallel calls are supported, agents may declare `supports_parallel_calls=True`.

Minimal configuration:

```json
{
  "agent": {
    "provider": "rsi",
    "adapter": "my_method.adapter:MyAdapter",
    "adapter_options": {"my_setting": 1},
    "model": "YOUR_MODEL"
  }
}
```

`adapter_options` is passed unchanged as `context.options`; credentials should
stay in environment variables, not options or exported state. Keep code and
dependencies fixed when resuming. For checkpoint files that reference large
artifacts, write immutable/versioned files under `context.artifact_dir`, not a
mutable "latest" file: interrupted refinement must not overwrite committed
snapshots. `export_state` must return finite JSON values (no NaN/Infinity).

For an executable, **offline lifecycle example, not a learning baseline**, see
[`examples/minimal_rsi.py`](examples/minimal_rsi.py):

```bash
python -m benchmark configs/examples/minimal_rsi.json
python -m benchmark configs/examples/minimal_rsi.json --initial
```

It exercises five training tasks, a round-2 replay after round 5, and held-out
evaluation using a fixed no-op Farm plan. To test checkpoint recovery instead,
start with a fresh output directory:

```bash
python -m benchmark configs/examples/minimal_rsi.json --max-new-train-runs 2
python -m benchmark configs/examples/minimal_rsi.json --resume
```

External adapters support initial evaluation, training, holdout, task replay,
task-level checkpoint resume, and the existing G0--G3 HCI calculator when the
chosen task manifest contains the required groups. For Minecraft/Astray, use
`game_options.gameworld.framework: "rsi"` (or omit it to infer the method);
the standard screenshot/action client is selected automatically. Browser-step
recovery (`resumable_steps`/`RSI_STEP_RESUME`) and legacy copied-workspace
`--target-dir` recovery remain built-in-only; external methods use task-level
`--resume`. Bundled adapters use the shared GameWorld client while retaining
their built-in recovery support even when configured with `provider: "rsi"`.
Method-specific response compatibility, error handling and recovery rules live
in each adapter's optional `runtime_policy` (`MethodRuntimePolicy`); the runner
calls generic hooks rather than branching on method names. Existing run
configurations must not be changed during an in-place resume: keep an old
provider alias for an old run; use the migrated configs for fresh runs.

### Optional Candidate Evaluation

A method that wants harness-managed same-task validation returns public
`CandidateBatch`/`Candidate` objects from `refine`, and implements the optional
`CandidateSelectingMethod.select` interface. The runner evaluates candidates on
the same training task, not on holdout tasks, and passes their
`SelectionEvaluation` values back to the method. `select` owns acceptance and
state commitment; return `SelectionResult` with status `committed`, `completed`,
or `unchanged`, or `needs_validation` with another candidate batch. Provide the
committed candidate ID when applicable. Candidate IDs must be unique within a
batch; artifact directories must be writable and isolated from committed state.

Methods may instead perform their own improvement/selection internally and
return `None`. Such updates are recorded as completed, but a same-task
`post_refinement_score` stays unavailable unless evaluated through the candidate
interface; the harness never invents a post-update score. G0 still compares the
original pre-refinement execution score with a later replay score.

### Optional Trajectory Classification

Training/evaluation logs and numeric results are emitted for every adapter.
The LLM trajectory classifiers under `analyze/` currently have artifact readers
for ACE, AHE, and HyperAgents. A new method's native modification artifacts need
their own reader for that classification pipeline; classification is **not** a
prerequisite for training, evaluation, replay, or HCI calculation.

## License and third-party components

SelfImprove-Bench's own code is released under the [MIT License](LICENSE).
The repository also bundles third-party components, some with modifications. Each
keeps its original license, which governs that directory:

| Component | Path | License / terms |
|---|---|---|
| ACE | `rsi_methods/ACE/ace/` | Apache-2.0 ([LICENSE.txt](rsi_methods/ACE/ace/LICENSE.txt)); modified for this benchmark, see [MODIFICATIONS.md](rsi_methods/ACE/ace/MODIFICATIONS.md) |
| AHE | `rsi_methods/AHE/agentic-harness-engineering/` | MIT ([LICENSE](rsi_methods/AHE/agentic-harness-engineering/LICENSE)) |
| NexAU 0.3.9 runtime | `third_party/nexau/` | Apache-2.0; original license retained inside the bundled wheel |
| HyperAgents | `rsi_methods/HyperAgents/hyperagents/` | **CC BY-NC-SA 4.0** ([LICENSE.md](rsi_methods/HyperAgents/hyperagents/LICENSE.md)): non-commercial use only; adaptations must be shared under the same license |
| Hero's Journey | `benches/heros_journey/HerosJourney/` | MIT ([LICENSE](benches/heros_journey/HerosJourney/LICENSE)) |
| GameWorld runtime and games | `benches/gameworld/assets/` | Adapted from [GameWorld](https://github.com/gameworld-project/gameworld) with some modifications; bundled game assets are for **research and educational use only**, see [benches/gameworld/README.md](benches/gameworld/README.md) |

The MIT License does not override these terms. In particular, the HyperAgents
component and the GameWorld game assets may not be used for commercial purposes.
