# Combat game module

This directory only maintains the Combat task generation, reference ground truth,
per-level simulator, and unified `game.py` adapter. For cross-game orchestration,
agent lifecycle, and RSI metrics, see the repository root [README](../README.md) and
run via `python -m benchmark`.

Holdout tasks are generated deterministically from the benchmark configuration's
`holdout_seed` and saved in each run's `task_manifest.json`. The benchmark does
not load a bundled fixed task pool. `batch_generate_tasks.py` can export tasks
for standalone inspection; those exports are not automatically used by the runner.
