# Package entry points and upstream import bridges

Run commands from the repository root. The supported benchmark entry point is:

```bash
python -m benchmark configs/main/farm_ace.json
python -m benchmark configs/main/combat_ace.json
```

The broken, deprecated `benches/farm/launcher.py`,
`benches/farm/batch_evaluate.py`, and `benches/combat/batch_evaluate.py` have been
removed. Their Python APIs and direct-script commands are no longer supported.
Move the old task count, difficulty, provider, and output settings into a
benchmark configuration and use `python -m benchmark CONFIG`; the main
configurations provide examples. This also avoids a local `benchmark.py`
shadowing the top-level benchmark package.

The task-pool generators use package-relative imports and module entry points:

```bash
python -m benches.farm.batch_generate_tasks --output-dir generated/farm --seed 100
python -m benches.combat.batch_generate_tasks --output-dir generated/combat --seed 100
```

Use `--help` for difficulty and pool-size options. Direct execution of these
generator files is not supported. Task generation is local and makes no model
API calls; Combat also runs its local reference solver.

## Why import bridges remain

Own runtime modules use package imports. `benchmark/imports.py` contains the
explicit exceptions needed to preserve bundled upstream code:

- Hero loads the bundled `herosjourney` package by its package spec, retaining
  upstream absolute imports without adding the vendor source root to
  `sys.path`. A cached package from another location is rejected rather than
  mixed with the bundled copy; failed initialization cleans partial modules.
- HyperAgents host code imports its nested vendor package directly. Its old
  global source-path injection is removed. Candidate scripts keep their
  upstream import layout inside their own container.
- GameWorld still has flat upstream modules such as `main` and `utils`; their
  source root is exposed through the central path bridge. This does not make
  GameWorld a fully namespaced distribution or allow switching cached roots
  within one process.
- AHE uses the same path bridge only in dedicated actor/evolver workers.
  Candidate tools and middleware precede upstream fallback modules. These
  paths and the worker's module cache remain separate from the parent process.

The bridge is not a workaround for executing ordinary benchmark files as
scripts. Upstream/vendor bootstrap code is preserved rather than rewritten as
part of the host package cleanup.

## AHE checkpoint compatibility

AHE checkpoint policy **v5** binds the import bridge's source digest in addition
to its existing method/configuration fingerprints. A changed bridge is a
different execution environment, so restoring the previous checkpoint is
rejected. This also means v4 checkpoints are not automatically accepted or
converted: resume them with their matching historical code, or start a new v5
run. Do not mix the two policies within one lineage.
