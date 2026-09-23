"""Deprecated launcher that forwards to the unified benchmark runner."""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark.legacy import run_single_game  # noqa: E402

@dataclass(frozen=True)
class LauncherConfig:
    difficulty: int = 2
    provider: str = "codex"
    rounds: int = 1
    model: str | None = None
    seed: int | None = None
    timeout: float = 300
    prompt_file: str | None = None
    max_attempts: int = 3
    continue_on_error: bool = True
    verbose: bool = False
    rsi_artifact_root: str = "benchmark_results/legacy_farm"

def run_benchmark(config: LauncherConfig, **provider_options: Any) -> dict[str, Any]:
    del provider_options
    return run_single_game("farm", provider=config.provider, model=config.model, timeout=config.timeout, max_attempts=config.max_attempts, difficulty=config.difficulty, count=config.rounds, result_dir=config.rsi_artifact_root, seed=config.seed or 0, continue_on_error=config.continue_on_error, show_agent_output=config.verbose)

def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated: use python -m benchmark CONFIG")
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark_results/legacy_farm")
    args = parser.parse_args()
    run_benchmark(LauncherConfig(difficulty=args.difficulty, provider=args.provider, rounds=args.rounds, seed=args.seed, rsi_artifact_root=args.output_dir))

if __name__ == "__main__":
    main()
