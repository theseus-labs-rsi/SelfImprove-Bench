# Farm game module

This directory only maintains the Farm task generation, reference ground truth,
simulator, and unified `game.py` adapter. For cross-game orchestration, agent
lifecycle, and RSI metrics, see the repository root [README](../README.md) and run
via `python -m benchmark`.

The 25 tasks with seed=100 under `task/` form the fixed holdout pool; use
`batch_generate_tasks.py` to regenerate that pool.
