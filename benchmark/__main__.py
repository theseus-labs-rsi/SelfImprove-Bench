"""Command-line entry point: ``python -m benchmark CONFIG``."""

from __future__ import annotations

import argparse

from .config import load_config
from .runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-game RSI benchmark")
    parser.add_argument("config", help="JSON or TOML benchmark configuration")
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--target-dir", help="New workspace for resuming from config.output_dir")
    recovery.add_argument("--resume", action="store_true", help="Resume checkpoints in config.output_dir")
    parser.add_argument("--max-new-train-runs", type=int, help="Pause after N new committed training runs")
    args = parser.parse_args()
    try:
        run_benchmark(load_config(args.config), target_dir=args.target_dir,
                      resume=args.resume, max_new_train_runs=args.max_new_train_runs)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
