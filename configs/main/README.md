# Main Experiment Configurations

Run from the repository root after installing the dependencies in the root README.
The 18 JSON files cover `farm`, `combat`, `sweepmine`, `hero`, `minecraft`, and
`astray`, each with `ace`, `ahe`, and `hyperagents`.
All use `agent.provider: "rsi"` and an explicit `agent.adapter` import path,
the same public interface used by additional RSI methods. The old method-named
providers remain compatibility aliases. Output method names and experiment
settings are unchanged. Do not switch config formats during an existing run;
checkpoint resume requires its original configuration.

## Models and Credentials

Set `OPENAI_API_KEY` in your shell. Set `OPENAI_BASE_URL` when using an
OpenAI-compatible gateway. Do not put credentials in JSON or TOML files.
The three method TOMLs use model `gpt-5.6-sol` with reasoning effort `low`.
This model identifier must be supported by your endpoint; changing it constitutes
a different model configuration. Set each method's role models in
`rsi_methods/{ACE,AHE,HyperAgents}/config.toml` if needed. The GameWorld
`model_spec` refers to the bundled catalog profile, not an API credential.

## Training and Initial Evaluation

For example, to collect the trained and untrained Minecraft ACE scores:

```bash
python -m benchmark configs/main/minecraft_ace.json
python -m benchmark configs/main/minecraft_ace.json --initial
```

The normal run has 20 training tasks, four read-only replay evaluations, and
final holdouts. `--initial` creates a fresh instance of the same method, performs
no training/refinement/replay, and evaluates identical holdouts. Initial outputs
have the `_initial` suffix. This is not a static CLI baseline. Existing output
directories are not overwritten; use `--resume` for interrupted runs.
`--initial --target-dir` is rejected so that evolved state cannot become the
initial baseline accidentally.

To run all configurations sequentially:

```bash
for config in configs/main/*.json; do
  python -m benchmark "$config" --initial || break
  python -m benchmark "$config" || break
done
```

Independent configurations can run in separate terminals. Training and analysis
make model API calls and may incur charges.

## Tasks and Score Records

| Games | Training | Own-game holdouts | Paired-game holdouts |
| --- | --- | --- | --- |
| Farm / Combat | D3, 20 tasks | D3: 20; D5: 20 | Paired game D3: 20 |
| Sweepmine / Hero | D3, 20 tasks | D3: 20; D4: 20 | Paired game D3: 20 |
| Minecraft / Astray | 20 explicit training tasks | Five official tasks | Paired game's five official tasks |

Pairs are Farm/Combat, Sweepmine/Hero, and Minecraft/Astray. `hero` uses internal
game name `heros_journey`. GameWorld assets are bundled under
`benches/gameworld/assets/tasks/{train,holdout}`. For Minecraft/Astray, the outer
`difficulty: 1` is an orchestration placeholder, not the task's difficulty.
Use `game_id` and `game_task_id` to distinguish games/tasks, including transfers.
Minecraft Medium is the mean of `18_02,18_03`; Hard is the mean of `18_04,18_05`.
Astray Medium is `03_03` and Hard is `03_05` for the reported task grouping.

Each output contains `config.resolved.json`, `task_manifest.json`, `result.json`,
and per-task execution records in `runs/`. These retain individual scores,
reference scores, status, token/time use, replay identity, and method artifacts.
Compute G1/G2 against the corresponding method's initial scores; G3 uses the
destination game's initial Medium score and its reference score, not the source
game baseline. For an aggregate, first average each set of task scores, then use
`(final - initial) / (reference - initial)`. A zero denominator is undefined,
not a zero gain. Treat failed executions as zero for the paper's convention,
while retaining the original error status in the records.
Use the matching configuration's `--initial` output for transfer baselines too:
holdout pool membership/order can affect generated task seeds across different
training-game configurations. Match the destination task identities, not just
the game/difficulty label. Keep scores on one scale: the HCI references below are
on the `relative_score` scale, not raw-profit or raw-objective solver references.

The replay schedule uses rounds 2/7/12/17 as originals and evaluates those exact
tasks after rounds 5/10/15/20. For G0, pair the original task's execution score
before that round's refinement (`relative_score`) with its later replay execution
score via `replay_of`. For example, compare the score before round 2's refinement
with the score of the framework retained after round 5 on that same round-2 task.
Do not use `post_refinement_score`, selected-candidate scores, or the maximum
candidate score as the G0 baseline. Selection evidence is not required for G0;
post-refinement scores remain available as diagnostic records. Initial and trained holdout
manifests have identical task identities even though their config hashes differ.

## HCI Calculation

After both the normal run and its matching `--initial` run finish:

```bash
python -m benchmark.hci --trained outputs/main/farm/ace --initial outputs/main/farm/ace_initial --output analysis/farm_ace_hci.json
```

The command supports all six games and all three frameworks. It reads
`result.json` and `task_manifest.json` in each directory without making model
calls or modifying experimental records. Replace the game and method names to
calculate another experiment. It produces scores and HCI data, not paper tables.

- Formula: `(mean(final) - mean(initial)) / (reference - mean(initial))`, using
  `relative_score` throughout. Output HCI values are ratios, not percentages.
  Average scores before dividing, retain their stored precision throughout, and
  round only for display. Scores and HCI values are not clipped, including
  scores above the reference.
- G0: the pre-refinement execution score on each original instance
  versus the later replay execution score on that exact instance.
- G1: initial versus trained scores on the source game's Medium holdouts.
- G2: initial versus trained scores on the source game's Hard holdouts.
- G3: initial versus trained scores on the paired destination game's Medium
  holdouts, both read from this experiment's matching holdout manifests.
  The source game's Medium baseline is never substituted.

HCI uses the following fixed, shared references from the reported experiment
groups. Farm/Combat use the highest saved group-level scores across baselines
and trained methods; the other four games use 1. These are not individual-task
maxima, method-specific maxima, or values recomputed from one run's scores.

| Game | Medium reference | Hard reference |
| --- | --- | --- |
| Farm | 1.179 | 1.297 |
| Combat | 1.433 | 1.314 |
| Sweepmine / Hero / Minecraft / Astray | 1 | 1 |

G0 and G1 use the source game's Medium reference; G2 uses its Hard reference;
G3 uses the destination game's Medium reference. Farm's Hard maximum includes
the saved Ds baseline (1.297), not just the three evolving methods. The fixed
references live in `benchmark/hci.py:HCI_REFERENCES`; a new comparison cohort
requiring different maxima must use a consistently updated reference table for
all methods, rather than silently changing the denominator per run. Each output
group records its `reference_score` and `reference_group`. Per-execution
`reference_score` fields are raw solver references and are not HCI denominators.
Historical tables that rounded input aggregates before computing HCI may differ
slightly from a calculation that retains the full available score precision.

Medium/Hard are D3/D5 for Farm/Combat and D3/D4 for Sweepmine/Hero. Minecraft
uses `18_02,18_03` / `18_04,18_05`; Astray uses `03_03` / `03_05`, matching the
reported grouping above. G3 uses the destination game's Medium definition.
The command verifies manifest fingerprints, complete result coverage, matching
initial/trained task identities, identical agent configurations, and an untrained
initial baseline. Use the same method TOML/model settings for both runs: the
benchmark JSON does not capture changes to external method TOMLs.

Execution records with `status=error` or `status=skipped` count as zero while
preserving their statuses in the output. Missing or invalid successful scores
are not silently converted to zero. Missing G0 pre-refinement execution scores
are not replaced by candidate or post-refinement scores. A missing group or score makes that HCI
`unavailable`; partial groups are not averaged into a reported HCI. The report
is saved with status `incomplete` and exit code 1. Invalid/mismatched inputs are
rejected with exit code 2. Zero headroom yields `hci: null`, status `undefined`,
reason `zero_headroom`; it is not a missing experiment or a zero gain.

## Trajectory Analysis

After training, run the existing analyzers on the full output directory:

```bash
mkdir -p analysis
python -m analyze ace outputs/main/farm/ace --model gpt-5.6-sol --output analysis/farm_ace.json
python -m analyze ahe outputs/main/farm/ahe --model gpt-5.6-sol --output analysis/farm_ahe.json
python -m analyze hyper outputs/main/farm/hyperagents --model gpt-5.6-sol --output analysis/farm_hyperagents.json
```

Replace `farm` with any of the other five configuration names. The HyperAgents
analysis subcommand is `hyper`, not `hyperagents`. Outputs contain per-node
proposals, classification dimensions, and available scores. Classification uses
a model; it is not a deterministic score calculation. ACE reads operations from
each task's candidate subdirectory; missing artifacts make the analysis status
`incomplete`, not `completed`, and the CLI exits with an error after saving the
partial results. AHE analyzes rejected proposals too, but only
accepted candidates become parents or contribute to the final retained snapshot.
Missing AHE selection reports are an error rather than an assumed linear chain.
For the adjacent-round score analysis, use the first 20 ordinary training
execution scores in `runs/`,
exclude replay/holdout/candidate-validation runs, and pair rounds 1-2 through
19-20. This yields 19 pairs; it does not establish same-task causal improvement.
Do not mix these differences with parent/candidate scores from analyzer nodes.
No paper-table generator is needed to retain these underlying experimental data.

### Readable Per-Game Reports

After classifying the trajectories above, combine the frameworks for one game:

```bash
python -m analyze report --game farm --input analysis/farm_ace.json analysis/farm_ahe.json analysis/farm_hyperagents.json --output-dir analysis/reports/farm --model gpt-5.6-sol
```

Use any of the six game names (`farm`, `combat`, `sweepmine`, `hero`, `minecraft`,
`astray`), with only the matching game's input files. The game name is explicitly
declared and checked against an embedded `game` field when present; otherwise
it is not independently verified. Trajectory schemas 5, 6, and 7 are supported;
old files without node scores can still be analyzed, with scores left missing.
One or multiple framework/run files are accepted; runs remain separate rather
than being treated as independent seeds or silently pooled. Duplicate input
contents and duplicate node IDs within one input are rejected.

Outputs in the requested directory:

- `report.md`: short classification/score tables, findings and possible framework
  characteristics. No proposal text or concrete agent-change recommendations.
  Reports are generated in English only, including model-written interpretations.
- `statistics.json`: complete statistics-only model input: counts, proportions,
  numeric score summaries and evidence IDs. No proposal text, code or raw logs.
- `report.json`: full-precision statistics, complete source evidence, source-file
  hashes, evidence IDs, model interpretation, limitations, and API usage metadata.
- `nodes.csv`: node-level scores, acceptance/selection status and proposal counts.
- `proposals.csv`: every extracted proposal occurrence with labels and evidence.

Without `--model`, the command performs no API calls, even if `OPENAI_MODEL` is
set. With `--model`, one structured interpretation call summarizes observations
and possible framework characteristics from aggregate statistics only.
Every statement must reference existing aggregate-statistic evidence IDs;
invalid references fail instead of silently entering the report. The statistical
files remain available if model interpretation fails, with an explicit failure
status. The command does not modify trajectory files or train/evaluate agents.

Statistics are recomputed from the node proposals, not cached summary counters.
Node deltas require finite parent/current scores and must agree with any stored
delta. Missing scores and unknown acceptance remain missing/unknown, not zero or
rejection. Category associations count each node once per label; overlapping
labels and multiple proposals at a node preclude causal attribution to one
proposal. The node score semantics stay those of each framework's analyzer:
these are **not** the adjacent-training-round score analysis or HCI. Incomplete
analyses are flagged and unclassified nodes are not zero-proposal observations.

Full evidence is retained in the audit JSON/CSV, but is never sent to the report
model. `statistics.json` is the entire data payload for interpretation; the model
does not reclassify proposals. Oversized summaries are rejected before API calls.
Hypotheses require controlled experiments before being treated as established results.

Optionally add `--score-stats FILE [FILE ...]` to attach score statistics:

- `benchmark.hci` JSON outputs (schema 1), importing G0--G3 summaries, not raw pairs.
- Adjacent-round CSV: `game,method,valid_pairs,positive,zero,negative,mean_delta`.
- Category association CSV: `game,method,category,pairs,positive,zero,negative,mean_delta`.
- Evaluation CSV: `game,method,metric,target_game,score_group,initial_score,final_score,hci_ratio,hci_status`.

Extra source/evidence text columns are discarded from the model input. Only rows
for the declared game are imported; methods must match a supplied trajectory.
Duplicate summaries are rejected. Missing scores stay missing; imported HCI is
not recalculated. Native node deltas, adjacent-round differences and final
evaluation gains are separate statistics, never interchangeable. Imported scores
are matched by game/method, not independently certified as the same run lineage.

## Provenance and Limits

- Farm's 18-crop catalog and numeric crop attributes are recovered from archived
  task records. The previous 13-crop catalog allowed only 13 distinct D5 crop
  subsets, insufficient for a 20-task unique holdout pool. Existing English
  display names are retained. Catalog ordering was recovered and validated
  against 80 archived task seeds/crop subsets. Display language differs from
  those historical tasks; use saved task records when comparing old scores.
- Sweepmine/Hero settings follow their archived resolved configurations; their
  historical Hard configuration is D4, not D5. Farm/Combat AHE settings follow
  its archived resolved configurations. Farm/Combat ACE/HyperAgents share those
  settings here as reference configurations; their exact original top-level
  configuration files were not recovered, so they are not certified originals.
- Minecraft/Astray task order and method settings follow the available main
  configurations, with a bundled model profile and relative output paths.
  The paired-game holdouts and four read-only replays are additions for collecting
  transfer and replay data; their original configurations had only five own-game
  holdouts and no scheduled replay. Own-game holdouts remain first in the pool.
- Historical Farm/Combat ACE/HyperAgents traces include refinement on replay
  tasks. This runner intentionally evaluates replays without refining, so those
  historical 24-event training trajectories cannot be reproduced exactly by
  these 20-refinement configurations. This supplement does not change selection
  logic, prompts, or scoring code to conceal that difference.
- This package provides rerun configurations and raw-data/analysis entry points,
  not a guarantee of numerically identical stochastic model outputs. No ablation
  configurations or generated tables are added.

## Offline Checks

No model calls are made by:

```bash
python -m unittest discover -s tests -p 'test_main_configs.py' -v
```

These checks generate real task manifests; Combat's local reference solver can
take several minutes. They do not execute model agents or collect new scores.
