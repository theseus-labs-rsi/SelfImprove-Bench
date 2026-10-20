"""Command-line entry point: ``python -m benchmark CONFIG``."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import ReplayConfig, load_config
from .runner import run_benchmark
from .methods import RSI_PROVIDERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-game RSI benchmark")
    parser.add_argument("config", help="JSON or TOML benchmark configuration")
    parser.add_argument(
        "--initial", action="store_true",
        help="Evaluate a fresh RSI framework on the same holdouts, without training",
    )
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--target-dir", help="New workspace for resuming from config.output_dir")
    recovery.add_argument("--resume", action="store_true", help="Resume checkpoints in config.output_dir")
    parser.add_argument("--max-new-train-runs", type=int, help="Pause after N new committed training runs")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.initial:
            if config.agent.provider not in RSI_PROVIDERS:
                raise ValueError("--initial requires an RSI method")
            if args.target_dir:
                raise ValueError("--initial cannot load an evolved --target-dir workspace")
            # Keep the holdout pool and seeds identical to the training config.
            config = replace(config, schedule=(), replay=ReplayConfig(),
                             output_dir=config.output_dir + "_initial")
        run_benchmark(config, target_dir=args.target_dir,
                      resume=args.resume, max_new_train_runs=args.max_new_train_runs)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
