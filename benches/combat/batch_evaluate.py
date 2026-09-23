"""Deprecated combat-only wrapper around the shared benchmark runner."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmark.legacy import run_single_game  # noqa: E402

DEFAULT_TASK_DIR = Path(__file__).parent / "task"
DEFAULT_PLAN_DIR = Path(__file__).parent / "plans"
DEFAULT_RESULT_DIR = Path(__file__).parent / "results"

def evaluate_batch(*, task_dir: str | Path = DEFAULT_TASK_DIR, plan_dir: str | Path = DEFAULT_PLAN_DIR, result_dir: str | Path = DEFAULT_RESULT_DIR, provider: str = "codex", model: str | None = None, timeout: float = 300, max_attempts: int = 3, difficulty: int | None = None, limit_per_difficulty: int | None = 1, continue_on_error: bool = True, prompt_file: str | Path | None = None, show_agent_output: bool = False) -> dict[str, Any]:
    del task_dir, plan_dir, prompt_file
    return run_single_game("combat", provider=provider, model=model, timeout=timeout, max_attempts=max_attempts, difficulty=difficulty, count=limit_per_difficulty, result_dir=result_dir, continue_on_error=continue_on_error, show_agent_output=show_agent_output)

def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated: use python -m benchmark CONFIG")
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--difficulty", type=int)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()
    evaluate_batch(provider=args.provider, difficulty=args.difficulty, limit_per_difficulty=args.count, result_dir=args.result_dir)

if __name__ == "__main__":
    main()
