"""Generate reproducible, solver-verified combat task files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .combat_planner import CombatError, generate_task, save_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_DIR = PROJECT_DIR / "task"


def generate_batch(
    *,
    output_dir: str | Path = DEFAULT_TASK_DIR,
    min_difficulty: int = 1,
    max_difficulty: int = 5,
    per_difficulty: int = 5,
    seed: int | None = 100,
) -> list[Path]:
    if not 1 <= min_difficulty <= max_difficulty <= 5:
        raise ValueError("difficulty range must be within 1..5")
    if per_difficulty < 1:
        raise ValueError("per_difficulty must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for difficulty in range(min_difficulty, max_difficulty + 1):
        for index in range(1, per_difficulty + 1):
            sequence = (difficulty - min_difficulty) * per_difficulty + index - 1
            task_seed = None if seed is None else seed + sequence
            task = generate_task(
                difficulty,
                seed=task_seed,
                task_id=f"combat-d{difficulty}-{index:02d}",
            )
            path = destination / f"difficulty_{difficulty}_task_{index:02d}.json"
            save_json(task, path)
            paths.append(path)
            print(
                f"[generate] difficulty={difficulty} index={index} "
                f"reference={task['reference_score']} verified=true",
                flush=True,
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic combat benchmark tasks")
    parser.add_argument("--min-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=5)
    parser.add_argument("--per-difficulty", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_TASK_DIR))
    args = parser.parse_args()
    try:
        paths = generate_batch(
            output_dir=args.output_dir,
            min_difficulty=args.min_difficulty,
            max_difficulty=args.max_difficulty,
            per_difficulty=args.per_difficulty,
            seed=None if args.random_seed else args.seed,
        )
    except (CombatError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(paths)} tasks in {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
