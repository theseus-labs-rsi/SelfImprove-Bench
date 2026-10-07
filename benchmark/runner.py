"""Single execution interface for all games and agent methods."""

from __future__ import annotations

import json
from copy import deepcopy
import time
from pathlib import Path
from typing import Any

from . import rsi_session
from .client import load_rsi_integration, request_json

from .config import BenchmarkConfig
from .game_adapters import EvaluationOutcome, evaluate_task
from .metrics import calculate_metrics, format_metrics
from .tasks import GAMES, PlannedTask, TaskManifest, build_manifest


RSI_METHODS = frozenset({"ace", "baseline", "hyperagents", "self_harness", "ahe"})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _suite_feedback(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rule-level records while retaining every detailed record."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("game")), int(record.get("difficulty", 0)))
        groups.setdefault(key, []).append(record)
    return [
        {
            "game": game,
            "difficulty": difficulty,
            "rule_count": len(items),
            "rule_success_count": sum(bool(item.get("success")) for item in items),
            "rule_success_rate": sum(bool(item.get("success")) for item in items) / len(items),
            "mean_score": sum(float(item.get("relative_score") or 0.0) for item in items) / len(items),
            "rule_results": items,
        }
        for (game, difficulty), items in sorted(groups.items())
    ]


def _token_total(metrics: dict[str, Any]) -> int:
    value = metrics.get("tokens", {}).get("total_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class AgentController:
    """Own either a stateless baseline or one evolving RSI lineage."""

    def __init__(
        self, config: BenchmarkConfig, output_dir: Path, training_count: int,
        *, resume_from: Path | None = None,
    ) -> None:
        self.config = config
        self.provider = config.agent.provider
        self.bridge: Any | None = None
        self.session: Any | None = None
        self.active_agent: Any | None = None
        if self.provider in RSI_METHODS:
            self.bridge = load_rsi_integration()
            resume_args = (
                {"resume_from": resume_from, "target_dir": output_dir}
                if resume_from is not None else {}
            )
            self.session = self.bridge.create_rsi_batch(
                self.provider,
                output_dir / "rsi",
                task_count=max(1, training_count),
                agent=config.agent,
                **resume_args,
            )

    def begin(self, task: PlannedTask) -> None:
        if self.session is None:
            self.active_agent = None
            return
        if task.phase == "train" and not task.replay:
            self.active_agent = self.session.begin_task(task.run_id)
        else:
            # Replay and holdout are read-only evaluations of committed state.
            final_agent = self.session.adaptor.load()
            self.active_agent = final_agent

    def request(
        self,
        task: dict[str, Any],
        *,
        prompt: str,
        validator: Any,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request_json(task, self.active_agent, prompt=prompt, validator=validator, metrics=metrics)

    def _request_json(self, task: dict[str, Any], rsi_agent: Any, **kwargs: Any) -> dict[str, Any]:
        agent = self.config.agent
        return request_json(
            task,
            **kwargs,
            provider=agent.provider,
            model=agent.model,
            timeout=agent.timeout,
            max_attempts=agent.max_attempts,
            show_agent_output=agent.show_output,
            rsi_agent=rsi_agent,
        )

    def request_for_agent(self, agent: Any):
        def request(task: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            return self._request_json(task, agent, **kwargs)

        return request

    def make_feedback(self, task: PlannedTask, record: dict[str, Any]) -> Any:
        if (
            self.session is None
            or self.bridge is None
            or task.phase != "train"
            or task.replay
        ):
            raise RuntimeError("FeedbackResult requires an RSI training task")
        steps = []
        for index, item in enumerate(record.get("interactions", [])):
            metrics = item.get("llm_metrics") or {}
            if item.get("agent_call") is not None:
                metrics = {"attempts": [{"rsi_agent_call": {
                    **item["agent_call"], "prompt": item.get("prompt", ""),
                }}]}
            steps.append(self.bridge.make_step_feedback(
                step_index=int(item.get("step_index", index)),
                metrics=metrics,
                action=item.get("action"),
                environment_response=item.get("environment_response"),
                success=item.get("success"),
                score=item.get("relative_score"),
                feedback=item.get("feedback"),
            ))
        execution_status = (
            self.bridge.ExecutionStatus.COMPLETED
            if record.get("status") == "ok"
            else self.bridge.execution_status_for_error(
                RuntimeError(str(record.get("error", "game task failed")))
            )
        )
        return self.bridge.make_feedback_result(
            task_id=task.run_id,
            task=task.task,
            trajectory=steps,
            execution_status=execution_status,
            success=record.get("status") == "ok",
            score=record.get("relative_score"),
            feedback=record.get("error") or "Task completed.",
            metrics={
                "game": task.game,
                "difficulty": task.difficulty,
                "relative_score": record.get("relative_score"),
                "replay": task.replay,
            },
        )
    def refine(
        self, task: PlannedTask, record: dict[str, Any], task_index: int,
        feedback: Any | None = None,
    ) -> Any | None:
        if (
            self.session is None
            or self.bridge is None
            or task.phase != "train"
            or task.replay
        ):
            return None
        return self.session.refine(
            feedback if feedback is not None else self.make_feedback(task, record), task_index
        )

    def close(self) -> None:
        if self.session is not None:
            self.session.close()


def _persist_tasks(output_dir: Path, manifest: TaskManifest) -> None:
    for item in manifest.holdout_pool:
        _write_json(
            output_dir
            / "tasks"
            / "holdout"
            / item.game
            / f"difficulty_{item.difficulty}"
            / f"{item.run_id}.json",
            item.task,
        )
    for item in manifest.training:
        if not item.replay:
            _write_json(output_dir / "tasks" / "train" / item.game / f"{item.run_id}.json", item.task)


def _error_result(task: PlannedTask, exc: Exception) -> dict[str, Any]:
    error_text = str(exc)
    is_timeout = isinstance(exc, TimeoutError) or "timeout" in error_text.lower()
    record = {
        "status": "error", "game": task.game, "task_id": task.task.get("task_id"),
        "difficulty": task.difficulty, "relative_score": None, "raw_score": None,
        "reference_score": task.task.get("reference_profit", task.task.get("reference_score")),
        "success": False, "error_type": "timeout" if is_timeout else type(exc).__name__,
        "error": f"无响应：{error_text}" if is_timeout else error_text,
        "agent_output": None, "simulation": None, "interactions": [], "llm_metrics": {},
    }
    if task.game == "heros_journey":
        record["interaction_protocol"] = task.task.get("interaction_protocol", "legacy_unlabeled")
        record["score_metric"] = "reference_steps_over_executed_steps_v1"
        record["completion_rate"] = None
    return record


def _ahe_skipped_result(result: dict[str, Any], controller: AgentController) -> dict[str, Any]:
    """Score an AHE empty response as zero while preserving its evidence."""
    error = str(result.get("error") or "").lower()
    markers = ("no response content or tool calls from llm", "empty response from api")
    if controller.provider != "ahe" or result.get("status") != "error":
        return result
    if not any(marker in error for marker in markers):
        return result
    return {
        **result,
        "status": "skipped",
        "relative_score": 0.0,
        "raw_score": 0.0,
        "success": False,
        "observed_relative_score": result.get("relative_score"),
        "skip_reason": "ahe_empty_response",
    }


def _selection_evaluation(
    record: dict[str, Any], result_path: Path
) -> rsi_session.SelectionEvaluation:
    score = record.get("relative_score")
    normalized_score = (
        float(score)
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else None
    )
    return rsi_session.SelectionEvaluation(
        status=str(record.get("status", "")),
        relative_score=normalized_score,
        success=bool(record.get("success", False)),
        result_path=str(result_path.resolve()),
        error_type=(str(record["error_type"]) if record.get("error_type") else None),
        error=(str(record["error"]) if record.get("error") else None),
    )


def _evaluate(
    task: PlannedTask, request: Any | None = None, agent: Any | None = None,
    reload_agent: Any | None = None,
) -> EvaluationOutcome:
    try:
        return evaluate_task(
            task.game, task.task, request=request, agent=agent, reload_agent=reload_agent
        )
    except Exception as exc:
        return EvaluationOutcome(_error_result(task, exc))


def _evaluate_candidates(
    task: PlannedTask,
    batch: Any,
    parent_evaluation: rsi_session.SelectionEvaluation,
    *,
    controller: AgentController,
    output_dir: Path,
    ordinal: int,
    stage_index: int,
    selection_mode: str = "paired",
) -> tuple[
    Any,
    list[dict[str, Any]],
    dict[str, tuple[EvaluationOutcome, float]],
    int,
    float,
]:
    candidate_evaluations: dict[str, rsi_session.SelectionEvaluation] = {}
    candidate_outcomes: dict[str, tuple[EvaluationOutcome, float]] = {}
    summaries: list[dict[str, Any]] = []
    validation_tokens = 0
    validation_seconds = 0.0
    candidate_ids: set[str] = set()
    for candidate in batch.candidates:
        candidate_id = str(candidate.candidate_id)
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("candidate ids must be non-empty and unique")
        candidate_ids.add(candidate_id)
        candidate_task = PlannedTask(
            run_id=task.run_id,
            phase=task.phase,
            game=task.game,
            difficulty=task.difficulty,
            task=deepcopy(task.task),
            task_seed=task.task_seed,
            group_index=task.group_index,
            replay=task.replay,
            replay_of=task.replay_of,
            replay_gap=task.replay_gap,
            generation_index=task.generation_index,
        )
        # Candidate validation uses the same task, separate from holdout evaluation.
        validation_started = time.perf_counter()
        outcome = _evaluate(
            candidate_task,
            request=controller.request_for_agent(candidate.agent),
            agent=candidate.agent,
        )
        result = outcome.record
        candidate_seconds = time.perf_counter() - validation_started
        validation_seconds += candidate_seconds
        validation_tokens += _token_total(result.get("llm_metrics", {}))
        artifact_dir = Path(candidate.artifact_dir).resolve()
        path = artifact_dir / "evaluation" / "result.json"
        _write_json(path, {
            **result, "candidate_id": candidate_id, "parent_run_id": task.run_id,
            "selection_stage_index": stage_index,
        })
        candidate_evaluations[candidate_id] = _selection_evaluation(result, path)
        candidate_outcomes[candidate_id] = (outcome, candidate_seconds)
        summaries.append({
            "candidate_id": candidate_id,
            "stage_index": stage_index,
            "result_path": str(path),
            "status": result.get("status"),
            "relative_score": result.get("relative_score"),
            "execution_seconds": round(candidate_seconds, 6),
        })
    selection = controller.session.select(
        parent_evaluation,
        candidate_evaluations,
        selection_mode=selection_mode,
    )
    return selection, summaries, candidate_outcomes, validation_tokens, validation_seconds


def _run_one(
    task: PlannedTask,
    *,
    controller: AgentController,
    ordinal: int,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    began = False
    try:
        controller.begin(task)
        began = True
        def reload_active_agent() -> Any:
            controller.begin(task)
            return controller.active_agent

        reload_agent = (
            reload_active_agent
            if task.phase == "holdout" and task.game == "heros_journey"
            and controller.provider == "ahe" and controller.session is not None
            else None
        )
        outcome = _evaluate(
            task,
            request=controller.request,
            agent=controller.active_agent,
            reload_agent=reload_agent,
        )
        result = _ahe_skipped_result(outcome.record, controller)
    except Exception as exc:
        result = _error_result(task, exc)
    execution_seconds = time.perf_counter() - started
    refinement = None
    selection_summary: dict[str, Any] | None = None
    candidate_validation_tokens = 0
    candidate_validation_seconds = 0.0
    if began and result["status"] == "skipped" and task.phase == "train" and not task.replay:
        controller.session.skip_task()
        selection_summary = {"status": "skipped", "reason": result["skip_reason"]}
    if (
        began
        and result["status"] != "skipped"
        and task.phase == "train"
        and not task.replay
        and controller.session is not None
    ):
        try:
            controller.refine(task, result, ordinal, feedback=outcome.feedback)
            refinement = controller.session.refinement_telemetry
            batch = controller.session.pending_batch
            parent_path = (
                output_dir / "runs" / "train" / f"{ordinal:04d}_{task.run_id}.json"
            )
            parent_evaluation = _selection_evaluation(result, parent_path)
            stages: list[dict[str, Any]] = []
            seen_batches: set[tuple[str, ...]] = set()
            while batch is not None:
                ids = tuple(str(item.candidate_id) for item in batch.candidates)
                if not ids or ids in seen_batches:
                    raise ValueError("candidate batch is empty or repeated")
                seen_batches.add(ids)
                selection, evaluations, _, stage_tokens, stage_seconds = _evaluate_candidates(
                    task, batch, parent_evaluation, controller=controller, output_dir=output_dir,
                    ordinal=ordinal, stage_index=len(stages)
                )
                candidate_validation_tokens += stage_tokens
                candidate_validation_seconds += stage_seconds
                status = str(getattr(selection, "status", "")).lower()
                stages.append({
                    "stage_index": len(stages),
                    "candidate_ids": list(ids),
                    "evaluations": evaluations,
                    "status": status,
                    "committed_candidate_id": getattr(selection, "committed_candidate_id", None),
                    "candidate_validation_tokens": stage_tokens,
                    "candidate_validation_seconds": round(stage_seconds, 6),
                })
                if status == "needs_validation":
                    batch = getattr(selection, "batch", None)
                    if batch is None:
                        raise ValueError("needs_validation selection must provide a batch")
                elif status in {"committed", "completed", "unchanged"}:
                    batch = None
                else:
                    raise ValueError(f"Unsupported RSI selection status: {status or '<missing>'}")
            selection_summary = {
                "stages": stages,
                "status": stages[-1]["status"] if stages else "unchanged",
                "committed_candidate_id": (
                    stages[-1]["committed_candidate_id"] if stages else None
                ),
            }
        except Exception as exc:
            selection_summary = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
            result = {
                **result,
                "parent_status": result.get("status"),
                "parent_error_type": result.get("error_type"),
                "parent_error": result.get("error"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    refinement_seconds = float(refinement.get("duration_seconds", 0.0)) if isinstance(refinement, dict) else 0.0
    refinement_tokens = _token_total(refinement) if isinstance(refinement, dict) else 0
    record = {
        **result, "run_id": task.run_id, "phase": task.phase, "group_index": task.group_index,
        "task_seed": task.task_seed, "replay": task.replay, "replay_of": task.replay_of,
        "replay_gap": task.replay_gap, "execution_seconds": round(execution_seconds, 6),
        "refinement": refinement, "selection": selection_summary,
        "candidate_validation_tokens": candidate_validation_tokens,
        "candidate_validation_seconds": round(candidate_validation_seconds, 6),
        "refinement_seconds": refinement_seconds,
        "total_seconds": round(execution_seconds + refinement_seconds + candidate_validation_seconds, 6),
        "inference_tokens": _token_total(result.get("llm_metrics", {})),
        "refinement_tokens": refinement_tokens,
        "total_tokens": _token_total(result.get("llm_metrics", {})) + refinement_tokens + candidate_validation_tokens,
    }
    _write_json(output_dir / "runs" / task.phase / f"{ordinal:04d}_{task.run_id}.json", record)
    score = record.get("relative_score")
    score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
    print(
        f"[{task.phase}] {ordinal:04d} {task.game} d{task.difficulty} "
        f"replay={task.replay} status={record['status']} relative_score={score_text}",
        flush=True,
    )
    return record


def _persist_online_record(
    task: PlannedTask,
    result: dict[str, Any],
    *,
    ordinal: int,
    output_dir: Path,
    execution_seconds: float,
    refinement: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
    candidate_validation_tokens: int = 0,
    candidate_validation_seconds: float = 0.0,
    extra_tokens: int = 0,
    extra_seconds: float = 0.0,
) -> dict[str, Any]:
    """Persist one first-seen online task without double-counting candidate cost."""

    refinement_seconds = (
        float(refinement.get("duration_seconds", 0.0))
        if isinstance(refinement, dict) else 0.0
    )
    refinement_tokens = _token_total(refinement) if isinstance(refinement, dict) else 0
    inference_tokens = _token_total(result.get("llm_metrics", {}))
    record = {
        **result,
        "run_id": task.run_id,
        "phase": task.phase,
        "group_index": task.group_index,
        "generation_index": task.generation_index,
        "task_seed": task.task_seed,
        "replay": task.replay,
        "replay_of": task.replay_of,
        "replay_gap": task.replay_gap,
        "execution_seconds": round(execution_seconds, 6),
        "refinement": refinement,
        "selection": selection,
        "candidate_validation_tokens": candidate_validation_tokens,
        "candidate_validation_seconds": round(candidate_validation_seconds, 6),
        "refinement_seconds": refinement_seconds,
        "inference_tokens": inference_tokens,
        "refinement_tokens": refinement_tokens,
        "total_tokens": refinement_tokens + candidate_validation_tokens + extra_tokens,
        "total_seconds": round(
            refinement_seconds + candidate_validation_seconds + extra_seconds, 6
        ),
    }
    if refinement is None:
        record["total_tokens"] = inference_tokens
        record["total_seconds"] = round(execution_seconds, 6)
    path = output_dir / "runs" / task.phase / f"{ordinal:04d}_{task.run_id}.json"
    _write_json(path, record)
    score = record.get("relative_score")
    score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
    print(
        f"[{task.phase}] {ordinal:04d} {task.game} d{task.difficulty} "
        f"generation={task.generation_index} replay={task.replay} "
        f"status={record['status']} relative_score={score_text}",
        flush=True,
    )
    return record


def _online_candidate_stage(
    current_task: PlannedTask,
    current_outcome: EvaluationOutcome,
    current_record_path: Path,
    target_task: PlannedTask,
    *,
    controller: AgentController,
    output_dir: Path,
    generation_index: int,
) -> tuple[
    tuple[EvaluationOutcome, float] | None,
    dict[str, Any],
    dict[str, Any] | None,
    int,
    float,
]:
    controller.refine(
        current_task,
        current_outcome.record,
        generation_index,
        feedback=current_outcome.feedback,
    )
    refinement = controller.session.refinement_telemetry
    parent_evaluation = _selection_evaluation(
        current_outcome.record, current_record_path
    )
    batch = controller.session.pending_batch
    stages: list[dict[str, Any]] = []
    seen_batches: set[tuple[str, ...]] = set()
    validation_tokens = 0
    validation_seconds = 0.0
    selected: tuple[EvaluationOutcome, float] | None = None

    while batch is not None:
        ids = tuple(str(item.candidate_id) for item in batch.candidates)
        if not ids or ids in seen_batches:
            raise ValueError("candidate batch is empty or repeated")
        seen_batches.add(ids)
        selection, evaluations, outcomes, stage_tokens, stage_seconds = _evaluate_candidates(
            target_task,
            batch,
            parent_evaluation,
            controller=controller,
            output_dir=output_dir,
            ordinal=generation_index,
            stage_index=len(stages),
            selection_mode="online",
        )
        validation_tokens += stage_tokens
        validation_seconds += stage_seconds
        status = str(getattr(selection, "status", "")).lower()
        committed_id = getattr(selection, "committed_candidate_id", None)
        stages.append({
            "stage_index": len(stages),
            "candidate_ids": list(ids),
            "evaluations": evaluations,
            "status": status,
            "committed_candidate_id": committed_id,
            "candidate_validation_tokens": stage_tokens,
            "candidate_validation_seconds": round(stage_seconds, 6),
        })
        if status == "needs_validation":
            batch = getattr(selection, "batch", None)
            if batch is None:
                raise ValueError("needs_validation selection must provide a batch")
            continue
        if status not in {"committed", "completed", "unchanged"}:
            raise ValueError(f"Unsupported RSI selection status: {status or '<missing>'}")
        if committed_id is not None:
            selected = outcomes.get(str(committed_id))
            if selected is None:
                raise ValueError("committed candidate has no evaluation outcome")
        batch = None

    summary = {
        "mode": "online",
        "stages": stages,
        "status": stages[-1]["status"] if stages else "unchanged",
        "committed_candidate_id": (
            stages[-1]["committed_candidate_id"] if stages else None
        ),
    }
    return selected, summary, refinement, validation_tokens, validation_seconds


def _run_online_training(
    manifest: TaskManifest,
    *,
    controller: AgentController,
    output_dir: Path,
    prior_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tasks = list(manifest.training)
    if not tasks or tasks[0].replay or tasks[0].generation_index != 0:
        raise ValueError("online evolution manifest must start with generation 0")

    records = list(prior_records or [])
    if records:
        position = len(records)
        current_ordinal = max(index for index in range(1, position + 1)
                              if not tasks[index - 1].replay)
        current_task = tasks[current_ordinal - 1]
        current_record = records[current_ordinal - 1]
        current_outcome = EvaluationOutcome(
            current_record, controller.make_feedback(current_task, current_record)
        )
        current_session_active = False
    else:
        baseline = tasks[0]
        baseline_started = time.perf_counter()
        controller.begin(baseline)
        current_outcome = _evaluate(
            baseline, request=controller.request, agent=controller.active_agent
        )
        baseline_result = _ahe_skipped_result(current_outcome.record, controller)
        if baseline_result is not current_outcome.record:
            current_outcome = EvaluationOutcome(baseline_result, current_outcome.feedback)
            controller.session.skip_task()
        baseline_seconds = time.perf_counter() - baseline_started
        current_record = _persist_online_record(
            baseline, current_outcome.record, ordinal=1, output_dir=output_dir,
            execution_seconds=baseline_seconds,
        )
        records = [current_record]
        current_task = baseline
        current_ordinal = 1
        current_session_active = current_record["status"] == "ok"
        position = 1

    while position < len(tasks):
        target = tasks[position]
        target_ordinal = position + 1
        if target.replay:
            raise ValueError("online replay must follow a completed generation")
        if target.generation_index is None or target.generation_index < 1:
            raise ValueError("online target must have a positive generation index")
        feedback_ordinal = next((index for index in range(position, 0, -1)
                                 if not tasks[index - 1].replay
                                 and records[index - 1]["status"] == "ok"), None)
        if feedback_ordinal is not None and not current_session_active:
            controller.begin(tasks[feedback_ordinal - 1])

        try:
            if feedback_ordinal is None:
                selected, refinement, validation_tokens, validation_seconds = None, None, 0, 0.0
                selection = {"mode": "online", "status": "skipped", "reason": "no_valid_feedback"}
            else:
                feedback_task = tasks[feedback_ordinal - 1]
                feedback_record = records[feedback_ordinal - 1]
                feedback_outcome = (
                    current_outcome if feedback_ordinal == current_ordinal else
                    EvaluationOutcome(feedback_record, controller.make_feedback(feedback_task, feedback_record))
                )
                selected, selection, refinement, validation_tokens, validation_seconds = _online_candidate_stage(
                    feedback_task,
                    feedback_outcome,
                    output_dir / "runs" / "train" / f"{feedback_ordinal:04d}_{feedback_task.run_id}.json",
                    target,
                    controller=controller,
                    output_dir=output_dir,
                    generation_index=target.generation_index,
                )
            fallback_tokens = 0
            fallback_seconds = 0.0
            if selected is None:
                fallback_started = time.perf_counter()
                controller.begin(target)
                next_outcome = _evaluate(
                    target, request=controller.request, agent=controller.active_agent
                )
                fallback_seconds = time.perf_counter() - fallback_started
                fallback_tokens = _token_total(next_outcome.record.get("llm_metrics", {}))
                skipped_result = _ahe_skipped_result(next_outcome.record, controller)
                if skipped_result is not next_outcome.record:
                    next_outcome = EvaluationOutcome(skipped_result, next_outcome.feedback)
                    selection = {**selection, "status": "skipped", "reason": skipped_result["skip_reason"]}
                    controller.session.skip_task()
                    controller.active_agent = None
                selection["fallback_parent_evaluation"] = True
                execution_seconds = fallback_seconds
                next_session_active = next_outcome.record["status"] == "ok"
            else:
                next_outcome, execution_seconds = selected
                selection["fallback_parent_evaluation"] = False
                next_session_active = False
            next_record = _persist_online_record(
                target,
                next_outcome.record,
                ordinal=target_ordinal,
                output_dir=output_dir,
                execution_seconds=execution_seconds,
                refinement=refinement,
                selection=selection,
                candidate_validation_tokens=validation_tokens,
                candidate_validation_seconds=validation_seconds,
                extra_tokens=fallback_tokens,
                extra_seconds=fallback_seconds,
            )
        except Exception as exc:
            refinement = controller.session.refinement_telemetry
            next_outcome = EvaluationOutcome(_error_result(target, exc))
            next_record = _persist_online_record(
                target,
                next_outcome.record,
                ordinal=target_ordinal,
                output_dir=output_dir,
                execution_seconds=0.0,
                refinement=refinement,
                selection={
                    "mode": "online",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            records.append(next_record)
            break

        records.append(next_record)
        current_task = target
        current_outcome = next_outcome
        current_record = next_record
        current_ordinal = target_ordinal
        current_session_active = next_session_active
        position += 1
        while position < len(tasks) and tasks[position].replay:
            replay = tasks[position]
            replay_record = _run_one(
                replay,
                controller=controller,
                ordinal=position + 1,
                output_dir=output_dir,
            )
            records.append(replay_record)
            position += 1
            if replay_record["status"] not in {"ok", "skipped"} and not controller.config.continue_on_error:
                return records
        if next_record["status"] not in {"ok", "skipped"} and not controller.config.continue_on_error:
            break
    return records


def _move_to_discarded(output_dir: Path, source: Path) -> None:
    """Preserve earlier discarded artifacts when resuming the same lineage."""
    destination = output_dir / "discarded" / source.relative_to(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        index = 1
        while destination.with_name(f"{destination.stem}_resume_{index:02d}{destination.suffix}").exists():
            index += 1
        destination = destination.with_name(f"{destination.stem}_resume_{index:02d}{destination.suffix}")
    source.rename(destination)


def _discard_copied_latest(output_dir: Path, controller: AgentController) -> None:
    """Move the untrusted task record and artifact inside the copied workspace."""
    session = controller.session
    if session is None or session.resumed_batch_dir is None:
        raise RuntimeError("Resume requires a restored RSI session")
    holdout = sorted((output_dir / "runs" / "holdout").glob("*.json"))
    training = sorted((output_dir / "runs" / "train").glob("*.json"))
    if session.resume_phase == "holdout":
        newest = holdout[-1] if holdout else None
    else:
        newest = training[-1] if training else None
    if newest is not None and session.resume_phase == "train":
        record = json.loads(newest.read_text(encoding="utf-8"))
        index = record.get("generation_index") if controller.config.online_evolution else int(newest.name[:4])
        if index is None or int(index) != session.resume_before_task_index:
            newest = None
    if newest is not None:
        _move_to_discarded(output_dir, newest)
    if session.resume_phase == "train":
        index = session.resume_before_task_index
        for artifact in session.resumed_batch_dir.glob(f"task_{index:04d}_*"):
            _move_to_discarded(output_dir, artifact)
    result = output_dir / "result.json"
    if result.exists():
        _move_to_discarded(output_dir, result)


def _existing_records(output_dir: Path, tasks: tuple[PlannedTask, ...], phase: str) -> list[dict[str, Any]]:
    """Read a contiguous prefix of already completed benchmark records."""
    records: list[dict[str, Any]] = []
    for ordinal, task in enumerate(tasks, start=1):
        path = output_dir / "runs" / phase / f"{ordinal:04d}_{task.run_id}.json"
        if not path.is_file():
            break
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("run_id") != task.run_id or record.get("phase") != phase:
            raise ValueError(f"Resume record does not match manifest: {path}")
        records.append(record)
    remaining = list((output_dir / "runs" / phase).glob("*.json"))
    if len(remaining) != len(records):
        raise ValueError(f"Non-contiguous {phase} records in resume workspace")
    return records


def run_benchmark(
    config: BenchmarkConfig, *, target_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate, orchestrate, evaluate and persist one complete benchmark."""
    source_dir = Path(config.output_dir).resolve()
    output_dir = Path(target_dir).resolve() if target_dir is not None else source_dir
    resuming_holdout = False
    if target_dir is None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if target_dir is not None:
        saved_config = json.loads((source_dir / "config.resolved.json").read_text(encoding="utf-8"))
        current_config = json.loads(json.dumps(config.as_dict()))
        saved_config["output_dir"] = current_config["output_dir"]
        saved_games = saved_config.get("game_options", {})
        current_games = current_config.get("game_options", {})
        if "heros_journey" in saved_games and "heros_journey" in current_games:
            saved_games["heros_journey"] = current_games["heros_journey"]
        if saved_config != current_config:
            raise ValueError("Resume configuration differs beyond output_dir and Hero difficulty defaults")
        previous = json.loads((source_dir / "task_manifest.json").read_text(encoding="utf-8"))
        saved_manifest = TaskManifest.from_mapping(previous)
        manifest = build_manifest(config)
        if len(saved_manifest.training) != len(manifest.training) or any(
            old.manifest_record() != new.manifest_record()
            for old, new in zip(saved_manifest.training, manifest.training)
        ):
            raise ValueError("Resume training tasks differ from completed lineage")
        if len(saved_manifest.holdout) != len(manifest.holdout) or any(
            (old.run_id, old.game, old.difficulty, old.task_seed, old.group_index)
            != (new.run_id, new.game, new.difficulty, new.task_seed, new.group_index)
            for old, new in zip(saved_manifest.holdout, manifest.holdout)
        ):
            raise ValueError("Resume holdout schedule differs from saved lineage")
        completed_training = _existing_records(source_dir, saved_manifest.training, "train")
        completed_holdout = _existing_records(source_dir, saved_manifest.holdout, "holdout")
        if completed_holdout and len(completed_training) != len(saved_manifest.training):
            raise ValueError("Holdout records exist before training is complete")
        resuming_holdout = bool(completed_holdout)
        if any(
            old.manifest_record() != new.manifest_record()
            for old, new in zip(
                saved_manifest.holdout[:len(completed_holdout)],
                manifest.holdout[:len(completed_holdout)],
            )
        ):
            raise ValueError("Completed holdout tasks differ from current configuration")
        if config.agent.provider not in {"ace", "ahe", "hyperagents"}:
            raise ValueError(f"Resume is not implemented for {config.agent.provider}")
    else:
        manifest = build_manifest(config)
        _write_json(output_dir / "config.resolved.json", config.as_dict())
        _write_json(output_dir / "task_manifest.json", manifest.as_dict())
        _persist_tasks(output_dir, manifest)

    refinement_count = (
        sum(item.count for item in config.schedule)
        if config.online_evolution
        else sum(not task.replay for task in manifest.training)
    )
    controller = AgentController(
        config, output_dir, refinement_count,
        resume_from=source_dir if target_dir is not None else None,
    )
    if target_dir is not None:
        _discard_copied_latest(output_dir, controller)
        _write_json(output_dir / "config.resolved.json", config.as_dict())
        _write_json(output_dir / "task_manifest.json", manifest.as_dict())
        _persist_tasks(output_dir, manifest)
    training_records = _existing_records(output_dir, manifest.training, "train") if target_dir is not None else []
    holdout_records = _existing_records(output_dir, manifest.holdout, "holdout") if target_dir is not None else []
    try:
        if config.online_evolution:
            training_records = _run_online_training(
                manifest, controller=controller, output_dir=output_dir,
                prior_records=training_records,
            )
        else:
            for index, task in enumerate(manifest.training, start=1):
                if index <= len(training_records):
                    continue
                record = _run_one(
                    task, controller=controller, ordinal=index, output_dir=output_dir
                )
                training_records.append(record)
                if record["status"] not in {"ok", "skipped"} and not config.continue_on_error:
                    break

        # Holdout starts only after all scheduled evolution/refinement has
        # finished. AgentController.begin loads but never refines the final RSI version.
        if len(training_records) == len(manifest.training) and (
            resuming_holdout or config.continue_on_error
            or all(item["status"] in {"ok", "skipped"} for item in training_records)
        ):
            for index, task in enumerate(manifest.holdout, start=1):
                if index <= len(holdout_records):
                    continue
                record = _run_one(task, controller=controller, ordinal=index, output_dir=output_dir)
                holdout_records.append(record)
                if record["status"] not in {"ok", "skipped"} and not config.continue_on_error:
                    break
    finally:
        controller.close()

    metrics = calculate_metrics(
        training_records,
        holdout_records,
        evolution_window=config.evolution_window,
    )
    report = {
        "schema_version": 1,
        "config_hash": manifest.config_hash,
        "config": config.as_dict(),
        "summary": {
            "scheduled_training_runs": len(manifest.training),
            "scheduled_refinements": (
                refinement_count // controller.session.adaptor.refinement_every
                if config.agent.provider == "ahe" and controller.session is not None
                else refinement_count
            ),
            "completed_training_runs": len(training_records),
            "scheduled_holdout_runs": len(manifest.holdout),
            "completed_holdout_runs": len(holdout_records),
            "training_errors": sum(item["status"] == "error" for item in training_records),
            "holdout_errors": sum(item["status"] == "error" for item in holdout_records),
            "training_skipped": sum(item["status"] == "skipped" for item in training_records),
            "holdout_skipped": sum(item["status"] == "skipped" for item in holdout_records),
        },
        "metrics": metrics,
        "suite_feedback": {
            "training": _suite_feedback(training_records),
            "holdout": _suite_feedback(holdout_records),
        },
        "training_runs": training_records,
        "holdout_runs": holdout_records,
    }
    _write_json(output_dir / "result.json", report)
    print(format_metrics(metrics), flush=True)
    print(f"Saved benchmark result to {output_dir / 'result.json'}", flush=True)
    return report
