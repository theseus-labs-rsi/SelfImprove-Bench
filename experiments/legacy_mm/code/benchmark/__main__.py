"""Command-line entry point: ``python -m benchmark CONFIG``."""

from __future__ import annotations

import argparse

from .config import load_config
from .runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-game RSI benchmark")
    parser.add_argument("config", help="JSON or TOML benchmark configuration")
    parser.add_argument("--resume", action="store_true", help="Resume a checkpointed experiment without repeating committed runs")
    parser.add_argument("--max-new-train-runs", type=int, metavar="N",
                        help="Safely pause after N new committed training rounds (leave the full manifest unchanged)")
    args = parser.parse_args()
    if args.max_new_train_runs is not None and args.max_new_train_runs < 1:
        parser.error("--max-new-train-runs must be a positive integer")
    try:
        run_benchmark(load_config(args.config), resume=args.resume,
                      max_new_train_runs=args.max_new_train_runs)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
