"""Command-line entry point: ``python -m benchmark CONFIG``."""

from __future__ import annotations

import argparse

from .config import load_config
from .runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-game RSI benchmark")
    parser.add_argument("config", help="JSON or TOML benchmark configuration")
    args = parser.parse_args()
    try:
        run_benchmark(load_config(args.config))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
