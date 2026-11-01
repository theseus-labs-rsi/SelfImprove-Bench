"""Combat task construction for the common benchmark interface."""

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
from benchmark.token_ledger import aggregate_feedback_usage

from . import game


class CombatBenchmark:
    """Hold the configured task seeds and generate Combat tasks on demand."""

    def __init__(self, seeds: Sequence[int], *, max_workers: int = 1) -> None:
        if not seeds:
            raise ValueError("At least one Combat task seed is required")
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
                task_id=f"combat-d{level}-c{index}-s{task_seed}",
            )
            for index, task_seed in enumerate(selected_seeds)
        ]

    def play(self, tasks: Sequence[Mapping[str, Any]], agent: Any) -> FeedbackResults:
        """Run each sequential Combat session as one benchmark task."""
        task_list = list(tasks)
        self._validate_tasks(task_list)
        if not task_list:
            return []
        if self._max_workers == 1:
            return [self._play_safely(task, agent) for task in task_list]
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(task_list))) as pool:
            return list(pool.map(lambda task: self._play_safely(task, agent), task_list))

    def aggregate(self, feedback_results: FeedbackResults) -> dict[str, Any]:
        """Summarize reference-score success, relative score, and all stage calls."""
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
        """Save generated task snapshots, reached stages, and aggregate results."""
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
            raise TypeError("Every Combat task must be a mapping")
        task_ids = [task.get("task_id") for task in tasks]
        if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
            raise ValueError("Every Combat task must have a non-empty task_id")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Combat tasks must have distinct task IDs")

    @staticmethod
    def _parse_action(response_text: str) -> dict[str, Any]:
        """Extract one current-stage JSON decision from the agent response."""
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
        raise ValueError("Combat agent response does not contain a JSON decision")

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

    @staticmethod
    def _record_tokens(metrics: dict[str, Any], response: BackendResponse) -> None:
        usage = response.usage
        metrics["usage"] = to_jsonable(usage)
        metrics["tokens"] = {
            "prompt_tokens": usage.input_tokens if usage else None,
            "reasoning_tokens": usage.reasoning_tokens if usage else None,
            "answer_tokens": usage.output_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }

    def _play_safely(self, task: Mapping[str, Any], agent: Any) -> FeedbackResult:
        """Keep a separate model call record for every reached stage."""
        prompts: list[str] = []
        responses: list[str] = []
        calls: list[AgentCallRecord | None] = []

        def request(
            stage: Mapping[str, Any],
            *,
            prompt: str,
            validator: Any,
            metrics: dict[str, Any],
        ) -> dict[str, Any]:
            prompts.append(prompt)
            responses.append("")
            calls.append(None)
            index = len(prompts) - 1
            started = time.perf_counter()
            try:
                response = agent.act(prompt)
                if isinstance(response, BackendResponse):
                    responses[index] = response.text
                    calls[index] = self._agent_call(response)
                    self._record_tokens(metrics, response)
                elif isinstance(response, str):
                    responses[index] = response
                else:
                    raise TypeError("Agent act() must return str or BackendResponse")
                return validator(self._parse_action(responses[index]), stage)
            finally:
                metrics["duration_seconds"] = round(time.perf_counter() - started, 6)

        native_result = game.evaluate(task, request)
        raw_score = native_result.get("raw_score")
        reference_score = native_result.get("reference_score")
        reached_reference = (
            native_result["status"] == "ok"
            and self._numeric(raw_score)
            and self._numeric(reference_score)
            and raw_score >= reference_score
        )
        trajectory = tuple(
            self._step_feedback(interaction, prompts, responses, calls)
            for interaction in native_result["interactions"]
        )
        return FeedbackResult(
            task_id=str(task["task_id"]),
            task=dict(task),
            trajectory=trajectory,
            evaluation=self._task_evaluation(native_result, reached_reference),
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
        index = int(interaction["step_index"])
        action = interaction.get("action")
        action_text = (
            json.dumps(action, ensure_ascii=False)
            if action is not None
            else (responses[index] if index < len(responses) else "")
        )
        return StepFeedback(
            step_index=index,
            prompt=prompts[index] if index < len(prompts) else "",
            image_records=(),
            action=action_text,
            environment_response=interaction.get("environment_response"),
            success=interaction.get("success"),
            score=interaction.get("relative_score"),
            feedback=interaction.get("feedback"),
            agent_call=calls[index] if index < len(calls) else None,
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
            "Reference objective score reached."
            if success
            else "Completed session scored below the reference objective."
            if status == ExecutionStatus.COMPLETED
            else str(native_result.get("error") or "Combat execution failed.")
        )
        return TaskEvaluation(
            execution_status=status,
            success=success,
            score=float(native_score) if CombatBenchmark._numeric(native_score) else 0.0,
            feedback=feedback,
            metrics={
                "game_status": native_result.get("status"),
                "simulation_success": native_result.get("success"),
                "victory": (native_result.get("simulation") or {}).get("victory"),
                "raw_score": native_result.get("raw_score"),
                "reference_score": native_result.get("reference_score"),
                "relative_score": native_score,
                "llm_metrics": native_result.get("llm_metrics"),
            },
        )


__all__ = ["CombatBenchmark"]
