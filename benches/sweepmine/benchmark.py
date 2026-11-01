"""Common Benchmark adaptor for batches of mutable Sweepmine tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmark.contracts import (
    ExecutionStatus,
    FeedbackResult,
    FeedbackResults,
    to_jsonable,
)
from benchmark.token_ledger import aggregate_feedback_usage

from .core import SweepmineCore, SweepmineState
from .play import build_error_feedback, play_feedback, snapshot_task
from .task_factory import (
    SweepmineDifficulty,
    build_task_descriptor,
    load_difficulty_manifest,
)


class SweepmineBenchmark:
    """Construct and execute independent Sweepmine episode states."""

    def __init__(
        self,
        difficulties: Mapping[int, SweepmineDifficulty | Mapping[str, Any]] | None = None,
        seeds: Sequence[int] = (),
        *,
        repeats: int = 1,
        max_workers: int = 1,
    ) -> None:
        difficulty_source = load_difficulty_manifest() if difficulties is None else difficulties
        if not difficulty_source:
            raise ValueError("At least one Sweepmine difficulty is required")
        if not seeds:
            raise ValueError("At least one Sweepmine task seed is required")
        if repeats <= 0 or max_workers <= 0:
            raise ValueError("repeats and max_workers must be positive")

        self._difficulties = {
            int(level): self._normalize_difficulty(value)
            for level, value in difficulty_source.items()
        }
        self._seeds = tuple(int(seed) for seed in seeds)
        self._repeats = repeats
        self._max_workers = max_workers

    def construct(
        self,
        difficulty: int,
        seed: int | None = None,
    ) -> list[SweepmineCore]:
        """Create fresh mutable tasks for the selected difficulty.

        Without ``seed``, the configured seed batch is used. Supplying a seed
        constructs the configured number of repeats for that one seed.
        """
        try:
            selected = self._difficulties[difficulty]
        except KeyError as exc:
            available = ", ".join(str(level) for level in sorted(self._difficulties))
            raise ValueError(
                f"Unknown Sweepmine difficulty {difficulty}; available: {available}"
            ) from exc

        task_seeds = self._seeds if seed is None else (int(seed),)
        tasks: list[SweepmineCore] = []
        for case_index, task_seed in enumerate(task_seeds):
            for repeat_index in range(self._repeats):
                task_id = (
                    f"sweepmine-d{difficulty}-c{case_index}-"
                    f"s{task_seed}-r{repeat_index}"
                )
                tasks.append(
                    self._build_task(task_id, difficulty, task_seed, selected)
                )
        return tasks

    def play(
        self,
        tasks: Sequence[SweepmineCore],
        agent: Any,
    ) -> FeedbackResults:
        """Execute a complete task batch and restore input ordering."""
        task_list = list(tasks)
        self._validate_tasks(task_list)
        if not task_list:
            return []
        if self._max_workers == 1:
            return [self._play_safely(task, agent) for task in task_list]

        results: list[FeedbackResult | None] = [None] * len(task_list)
        worker_count = min(self._max_workers, len(task_list))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self._play_safely, task, agent): index
                for index, task in enumerate(task_list)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        if any(result is None for result in results):
            raise RuntimeError("Sweepmine worker returned no feedback result")
        return [result for result in results if result is not None]

    def aggregate(self, feedback_results: FeedbackResults) -> dict[str, Any]:
        """Build a benchmark-owned summary for one task batch."""
        status_counts = {status.value: 0 for status in ExecutionStatus}
        for result in feedback_results:
            status_counts[result.evaluation.execution_status.value] += 1
        usage_report = aggregate_feedback_usage(feedback_results)

        task_count = len(feedback_results)
        success_count = sum(result.evaluation.success for result in feedback_results)
        score_total = sum(result.evaluation.score for result in feedback_results)
        return {
            "task_count": task_count,
            "success_count": success_count,
            "success_rate": success_count / task_count if task_count else 0.0,
            "score_total": score_total,
            "score_mean": score_total / task_count if task_count else 0.0,
            "execution_status_counts": status_counts,
            **usage_report,
        }

    def persist(
        self,
        feedback_results: FeedbackResults,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Persist benchmark feedback and its aggregate summary."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        feedback_path = output_path / "feedback_results.json"
        summary_path = output_path / "summary.json"
        self._write_json(feedback_path, feedback_results)
        self._write_json(summary_path, self.aggregate(feedback_results))
        return {"feedback_results": feedback_path, "summary": summary_path}

    @staticmethod
    def _normalize_difficulty(
        value: SweepmineDifficulty | Mapping[str, Any],
    ) -> SweepmineDifficulty:
        if isinstance(value, SweepmineDifficulty):
            return value
        return SweepmineDifficulty.from_mapping(value)

    @staticmethod
    def _build_task(
        task_id: str,
        difficulty: int,
        seed: int,
        config: SweepmineDifficulty,
    ) -> SweepmineCore:
        return SweepmineCore(
            build_task_descriptor(
                difficulty, seed=seed, task_id=task_id, options=config
            )
        )

    @staticmethod
    def _validate_tasks(tasks: Sequence[SweepmineCore]) -> None:
        if any(not isinstance(task, SweepmineCore) for task in tasks):
            raise TypeError("Every Sweepmine task must be a SweepmineCore")
        if len({id(task) for task in tasks}) != len(tasks):
            raise ValueError("The same mutable Sweepmine task cannot be submitted twice")
        if any(
            task.status != SweepmineState.ONGOING
            or task.steps != task.max_steps
            or task.restart_count != 0
            for task in tasks
        ):
            raise ValueError("Sweepmine tasks must be fresh and unplayed")

    @staticmethod
    def _play_safely(task: SweepmineCore, agent: Any) -> FeedbackResult:
        try:
            return play_feedback(task, agent)
        except Exception as exc:
            return build_error_feedback(task, snapshot_task(task), [], exc)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(to_jsonable(value), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = ["SweepmineBenchmark", "SweepmineDifficulty"]
