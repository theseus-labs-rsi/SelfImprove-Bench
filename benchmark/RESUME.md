# RSI recovery entry points (ACE / AHE / HyperAgents)

Run from the repository root; use an empty `output_dir` for a new experiment:

```bash
python -m benchmark CONFIG --max-new-train-runs 1
python -m benchmark CONFIG --resume
```

`--max-new-train-runs N` pauses after N new training tasks have committed a checkpoint.
`--resume` continues the same output directory; it requires an unchanged config and
task manifest, and after validating the consecutive rounds, result hashes, and method
state it skips already-committed tasks. An orphaned `runs/` file does not mean the
state was committed. A process lock guards the directory, so two instances cannot run
at the same time.

Recovery into a new directory is still supported:

```bash
python -m benchmark CONFIG --target-dir NEW_OUTPUT_DIR
```

`CONFIG.output_dir` is the original directory and `--target-dir` is the new working
directory; the existing adaptor recovery rules apply. It cannot be combined with
`--resume` or `--max-new-train-runs`. An older experiment without
`checkpoints/initial.json` cannot use the new `--resume`.

Ordinary errors keep the `continue_on_error` semantics; fatal AHE errors stop
the experiment. Continuing after a task that was not successfully committed stops
appending to the checkpoint chain, keeps the ordinary results, and marks
`checkpoint_chain_complete=false` in the summary. Original-directory recovery refuses to
skip executed records that follow such uncommitted tasks, so an evolution state after a
failure is not mistakenly treated as committed.

Before an uncommitted task is re-executed, its old results and part of its evolution
evidence are moved to `abandoned_attempts/`. `attempt.json` marks that the cost may be
unknown; the known usage in run records does not include every interrupted cost. The
current metrics algorithm follows CSD, so such archives need a separate cost check and
the sum of records must not be treated as the true total cost.

GameWorld step-level recovery is enabled separately by `resumable_steps=true` or
`RSI_STEP_RESUME=1`. When enabled, it preserves the browser checkpoint, already-received
model replies, and session under `output_dir/gameworld/`, avoiding re-requesting steps
whose evidence already exists. If the post-training refinement has started but not
committed, automatic retry of that evolution stage is refused. Requests that were sent
without receiving usage may still incur extra cost.

`--resume` is task-state recovery while a GameWorld checkpoint is in-game step recovery;
the two cannot substitute for each other. After changing code, dependencies, or asset
paths, start a new experiment directory; old checkpoints are not migrated automatically.
