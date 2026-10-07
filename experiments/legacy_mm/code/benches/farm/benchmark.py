"""Farm task construction for the common benchmark interface."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmark.contracts import (
    AgentCallRecord,
    BackendResponse,
    ExecutionStatus,
    FeedbackResult,
    FeedbackResults,
    StepFeedback,
    TaskEvaluation,
    to_jsonable,
)

from . import game


class FarmBenchmark:
    """Hold the configured task seeds and generate Farm tasks on demand."""

    def __init__(self, seeds: Sequence[int], *, max_workers: int = 1) -> None:
        if not seeds:
            raise ValueError("At least one Farm task seed is required")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._seeds = tuple(int(seed) for seed in seeds)
        self._max_workers = max_workers

    def construct(self, difficulty: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Generate a fresh deterministic task for each selected seed."""
        level = int(difficulty)
        selected_seeds = self._seeds if seed is None else (int(seed),)
        return [
            game.generate(
                level,
                seed=task_seed,
                task_id=f"farm-d{level}-c{index}-s{task_seed}",
            )
            for index, task_seed in enumerate(selected_seeds)
        ]

    def play(self, tasks: Sequence[Mapping[str, Any]], agent: Any) -> FeedbackResults:
        """Run independent Farm tasks and restore their input ordering."""
        task_list = list(tasks)
        self._validate_tasks(task_list)
        if not task_list:
            return []
        if self._max_workers == 1:
            return [self._play_safely(task, agent) for task in task_list]
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(task_list))) as pool:
            return list(pool.map(lambda task: self._play_safely(task, agent), task_list))

    def aggregate(self, feedback_results: FeedbackResults) -> dict[str, Any]:
        """Summarize reference-score success, relative score, and model usage."""
        status_counts = {status.value: 0 for status in ExecutionStatus}
        usage_fields = (
            "input_tokens", "output_tokens", "total_tokens",
            "cached_input_tokens", "reasoning_tokens",
        )
        usage_totals = {field: 0 for field in usage_fields}
        for result in feedback_results:
            status_counts[result.evaluation.execution_status.value] += 1
            for step in result.trajectory:
                usage = step.agent_call.usage if step.agent_call else None
                if usage is not None:
                    for field in usage_fields:
                        usage_totals[field] += getattr(usage, field) or 0
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
            "usage": usage_totals,
        }

    def persist(
        self,
        feedback_results: FeedbackResults,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Save generated task snapshots, trajectories, and aggregate results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        feedback_path = output_path / "feedback_results.json"
        summary_path = output_path / "summary.json"
        feedback_path.write_text(
            json.dumps(to_jsonable(feedback_results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(self.aggregate(feedback_results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"feedback_results": feedback_path, "summary": summary_path}

    @staticmethod
    def _validate_tasks(tasks: Sequence[Mapping[str, Any]]) -> None:
        if any(not isinstance(task, Mapping) for task in tasks):
            raise TypeError("Every Farm task must be a mapping")
        task_ids = [task.get("task_id") for task in tasks]
        if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
            raise ValueError("Every Farm task must have a non-empty task_id")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Farm tasks must have distinct task IDs")

    @staticmethod
    def _parse_action(response_text: str) -> dict[str, Any]:
        """Extract the requested JSON plan without changing its fields."""
        decoder = json.JSONDecoder()
        for index, character in enumerate(response_text):
            if character != "{":
                continue
            try:
                action, _ = decoder.raw_decode(response_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(action, dict):
                return action
        raise ValueError("Farm agent response does not contain a JSON plan")

    @staticmethod
    def _agent_call(response: BackendResponse) -> AgentCallRecord:
        return AgentCallRecord(
            response_text=response.text,
            reasoning=response.reasoning,
            usage=response.usage,
            provider=response.provider,
            model=response.model,
            status=response.status,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
        )

    def _play_safely(self, task: Mapping[str, Any], agent: Any) -> FeedbackResult:
        """Preserve model calls even when parsing or simulation fails."""
        prompts: list[str] = []
        responses: list[str] = []
        calls: list[AgentCallRecord | None] = []

        def request(
            current_task: Mapping[str, Any],
            *,
            prompt: str,
            validator: Any,
            metrics: dict[str, Any],
        ) -> dict[str, Any]:
            prompts.append(prompt)
            started = time.perf_counter()
            try:
                response = agent.act(prompt)
                if isinstance(response, BackendResponse):
                    responses.append(response.text)
                    calls.append(self._agent_call(response))
                    metrics["usage"] = to_jsonable(response.usage)
                elif isinstance(response, str):
                    responses.append(response)
                    calls.append(None)
                else:
                    raise TypeError("Agent act() must return str or BackendResponse")
                return validator(self._parse_action(responses[-1]), current_task)
            finally:
                metrics["duration_seconds"] = round(time.perf_counter() - started, 6)

        native_result = game.evaluate(task, request)
        interaction = native_result["interactions"][0]
        raw_score = native_result.get("raw_score")
        reference_score = native_result.get("reference_score")
        reached_reference = (
            native_result["status"] == "ok"
            and self._numeric(raw_score)
            and self._numeric(reference_score)
            and raw_score >= reference_score
        )
        trajectory = (self._step_feedback(interaction, prompts, responses, calls),)
        evaluation = self._task_evaluation(native_result, reached_reference)
        return FeedbackResult(
            task_id=str(task["task_id"]),
            task=dict(task),
            trajectory=trajectory,
            evaluation=evaluation,
        )

    @staticmethod
    def _numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _step_feedback(
        interaction: Mapping[str, Any],
        prompts: Sequence[str],
        responses: Sequence[str],
        calls: Sequence[AgentCallRecord | None],
    ) -> StepFeedback:
        action = interaction.get("action")
        action_text = (
            json.dumps(action, ensure_ascii=False)
            if action is not None
            else (responses[0] if responses else "")
        )
        return StepFeedback(
            step_index=0,
            prompt=prompts[0] if prompts else "",
            image_records=(),
            action=action_text,
            environment_response=interaction.get("environment_response"),
            success=interaction.get("success"),
            score=interaction.get("relative_score"),
            feedback=interaction.get("feedback"),
            agent_call=calls[0] if calls else None,
        )

    @staticmethod
    def _task_evaluation(native_result: Mapping[str, Any], success: bool) -> TaskEvaluation:
        native_score = native_result.get("relative_score")
        error_type = str(native_result.get("error_type") or "")
        status = (
            ExecutionStatus.COMPLETED
            if native_result["status"] == "ok"
            else ExecutionStatus.TIMEOUT
            if "timeout" in error_type.lower()
            else ExecutionStatus.ERROR
        )
        feedback = (
            "Reference profit reached."
            if success
            else "Valid plan scored below the reference profit."
            if status == ExecutionStatus.COMPLETED
            else str(native_result.get("error") or "Farm execution failed.")
        )
        return TaskEvaluation(
            execution_status=status,
            success=success,
            score=float(native_score) if FarmBenchmark._numeric(native_score) else 0.0,
            feedback=feedback,
            metrics={
                "game_status": native_result.get("status"),
                "simulation_success": native_result.get("success"),
                "raw_score": native_result.get("raw_score"),
                "reference_score": native_result.get("reference_score"),
                "relative_score": native_score,
                "llm_metrics": native_result.get("llm_metrics"),
            },
        )


__all__ = ["FarmBenchmark"]
