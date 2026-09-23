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
            "mean_score": sum(float(item.get("relative_score", 0.0)) for item in items) / len(items),
            "rule_results": items,
        }
        for (game, difficulty), items in sorted(groups.items())
    ]


def _token_total(metrics: dict[str, Any]) -> int:
    value = metrics.get("tokens", {}).get("total_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class AgentController:
    """Own either a stateless baseline or one evolving RSI lineage."""

    def __init__(self, config: BenchmarkConfig, output_dir: Path, training_count: int) -> None:
        self.config = config
        self.provider = config.agent.provider
        self.bridge: Any | None = None
        self.session: Any | None = None
        self.active_agent: Any | None = None
        if self.provider in RSI_METHODS:
            self.bridge = load_rsi_integration()
            self.session = self.bridge.create_rsi_batch(
                self.provider,
                output_dir / "rsi",
                task_count=max(1, training_count),
                agent=config.agent,
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
        steps = [
            self.bridge.make_step_feedback(
                step_index=int(item.get("step_index", index)),
                metrics=item.get("llm_metrics", {}),
                action=item.get("action"),
                environment_response=item.get("environment_response"),
                success=item.get("success"),
                score=item.get("relative_score"),
                feedback=item.get("feedback"),
            )
            for index, item in enumerate(record.get("interactions", []))
        ]
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
    return {
        "status": "error", "game": task.game, "task_id": task.task.get("task_id"),
        "difficulty": task.difficulty, "relative_score": 0.0, "raw_score": None,
        "reference_score": task.task.get("reference_profit", task.task.get("reference_score")),
        "success": False, "error_type": type(exc).__name__, "error": str(exc),
        "agent_output": None, "simulation": None, "interactions": [], "llm_metrics": {},
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
    task: PlannedTask, request: Any | None = None, agent: Any | None = None
) -> EvaluationOutcome:
    try:
        return evaluate_task(task.game, task.task, request=request, agent=agent)
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
) -> tuple[Any, list[dict[str, Any]], int, float]:
    candidate_evaluations: dict[str, rsi_session.SelectionEvaluation] = {}
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
        )
        # Candidate validation uses the same task, separate from holdout evaluation.
        validation_started = time.perf_counter()
        outcome = _evaluate(
            candidate_task,
            request=controller.request_for_agent(candidate.agent),
            agent=candidate.agent,
        )
        result = outcome.record
        validation_seconds += time.perf_counter() - validation_started
        validation_tokens += _token_total(result.get("llm_metrics", {}))
        artifact_dir = Path(candidate.artifact_dir).resolve()
        path = artifact_dir / "evaluation" / "result.json"
        _write_json(path, {
            **result, "candidate_id": candidate_id, "parent_run_id": task.run_id,
            "selection_stage_index": stage_index,
        })
        candidate_evaluations[candidate_id] = _selection_evaluation(result, path)
        summaries.append({
            "candidate_id": candidate_id,
            "stage_index": stage_index,
            "result_path": str(path),
            "status": result.get("status"),
            "relative_score": result.get("relative_score"),
        })
    selection = controller.session.select(parent_evaluation, candidate_evaluations)
    return selection, summaries, validation_tokens, validation_seconds


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
        outcome = _evaluate(
            task,
            request=controller.request,
            agent=controller.active_agent,
        )
        result = outcome.record
    except Exception as exc:
        result = _error_result(task, exc)
    execution_seconds = time.perf_counter() - started
    refinement = None
    selection_summary: dict[str, Any] | None = None
    candidate_validation_tokens = 0
    candidate_validation_seconds = 0.0
    if (
        began
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
                selection, evaluations, stage_tokens, stage_seconds = _evaluate_candidates(
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
            result = {**result, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
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


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Generate, orchestrate, evaluate and persist one complete benchmark."""
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config)
    _write_json(output_dir / "config.resolved.json", config.as_dict())
    _write_json(output_dir / "task_manifest.json", manifest.as_dict())
    _persist_tasks(output_dir, manifest)

    refinement_count = sum(not task.replay for task in manifest.training)
    controller = AgentController(config, output_dir, refinement_count)
    training_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    try:
        for index, task in enumerate(manifest.training, start=1):
            record = _run_one(task, controller=controller, ordinal=index, output_dir=output_dir)
            training_records.append(record)
            if record["status"] != "ok" and not config.continue_on_error:
                break

        # Holdout starts only after all scheduled evolution/refinement has
        # finished. AgentController.begin loads but never refines the final RSI version.
        if len(training_records) == len(manifest.training) and (
            config.continue_on_error or all(item["status"] == "ok" for item in training_records)
        ):
            for index, task in enumerate(manifest.holdout, start=1):
                record = _run_one(task, controller=controller, ordinal=index, output_dir=output_dir)
                holdout_records.append(record)
                if record["status"] != "ok" and not config.continue_on_error:
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
            "completed_training_runs": len(training_records),
            "scheduled_holdout_runs": len(manifest.holdout),
            "completed_holdout_runs": len(holdout_records),
            "training_errors": sum(item["status"] != "ok" for item in training_records),
            "holdout_errors": sum(item["status"] != "ok" for item in holdout_records),
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
