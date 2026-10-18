# Main Experiment Configurations

Run from the repository root after installing the dependencies in the root README.
The 18 JSON files cover `farm`, `combat`, `sweepmine`, `hero`, `minecraft`, and
`astray`, each with `ace`, `ahe`, and `hyperagents`.

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
the game/difficulty label. Keep scores on one scale; when using `relative_score`,
use its normalized reference (1), not a raw-profit or raw-objective reference.

The replay schedule uses rounds 2/7/12/17 as originals and evaluates those exact
tasks after rounds 5/10/15/20. For G0, pair the score of the harness retained
after the original task's refinement with its replay execution score via
`replay_of`. Training records save this baseline as `post_refinement_score`.
For older records, the metrics reader recovers it from the final committed
candidate's same-task validation result; when selection retains the parent,
use the original execution score. Never use a rejected candidate or the maximum
candidate score. Missing selection evidence is reported as an unavailable pair,
not silently replaced with the pre-refinement score. Initial and trained holdout
manifests have identical task identities even though their config hashes differ.

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
