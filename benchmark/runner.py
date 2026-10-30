"""Single execution interface for all games and agent methods."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from . import resume_state, rsi_session
from .checkpoint_io import lock_file, unlock_file
from .client import load_rsi_integration, request_json

from .config import BenchmarkConfig
from .game_adapters import EvaluationOutcome, evaluate_task
from .metrics import calculate_metrics, format_metrics, post_refinement_score
from .tasks import GAMES, PlannedTask, TaskManifest, build_manifest
from .methods import BUILTIN_METHODS, RSI_PROVIDERS, method_identity, runtime_policy


RSI_METHODS = frozenset(RSI_PROVIDERS)


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


def _is_fatal_method_error(config: BenchmarkConfig, record: Mapping[str, Any]) -> bool:
    if record.get("status") in {"ok", "skipped"}:
        return False
    return runtime_policy(config.agent).is_fatal_error(record.get("error"))


def _is_provider_failure(record: Mapping[str, Any]) -> bool:
    """Keep transport failures separate from scored invalid model actions."""
    if record.get("status") != "error":
        return False
    metrics = record.get("llm_metrics") or {}
    calls = [metrics, *(metrics.get("stages") or [])]
    attempts = [attempt for call in calls for attempt in call.get("attempts", [])]
    if attempts:
        return attempts[-1].get("status") == "provider_error"
    # GameWorld and method workers can wrap the transport exception. Preserve
    # its type across result serialization instead of refining a failed request.
    if any(_is_provider_failure({"status": "error", **cause})
           for cause in record.get("error_chain", [])):
        return True
    error_type = str(record.get("error_type") or "").lower()
    if error_type in {
        "timeout", "timeouterror", "apitimeouterror", "apiconnectionerror",
        "connectionerror", "connecttimeout", "readtimeout", "ratelimiterror",
        "authenticationerror", "permissiondeniederror", "internalservererror",
        "serviceunavailableerror",
    }:
        return True
    error = str(record.get("error") or "").lower()
    return any(marker in error for marker in (
        "api request timed out", "request timed out", "openai request failed",
        "connection refused", "connection reset by peer",
        "apiconnectionerror:", "apitimeouterror:", "ratelimiterror:",
        "authenticationerror:", "internalservererror:",
        "error in agent execution: connection error.",
    ))


def _must_stop_after_error(config: BenchmarkConfig, record: Mapping[str, Any]) -> bool:
    """Stop on fatal method errors even when ordinary errors may be tolerated."""
    if record.get("status") in {"ok", "skipped"}:
        return False
    return (not config.continue_on_error or _is_fatal_method_error(config, record)
            or _is_provider_failure(record))


def _checkpoint_is_stable(config: BenchmarkConfig, task: PlannedTask,
                          record: Mapping[str, Any]) -> bool:
    """A failed task attempt can still have a valid post-refinement RSI state."""
    if _is_provider_failure(record):
        return False
    if task.phase == "train" and not task.replay:
        status = (record.get("selection") or {}).get("status")
        if status not in {"committed", "completed", "unchanged", "skipped"}:
            return False
    if record.get("status") in {"ok", "skipped"}:
        return True
    return record.get("status") == "error" and not _is_fatal_method_error(config, record)


class AgentController:
    """Own one method lineage through the shared RSI lifecycle."""

    def __init__(
        self, config: BenchmarkConfig, output_dir: Path, training_count: int,
        *, resume_from: Path | None = None,
        batch_id: str | None = None, resume: bool = False,
    ) -> None:
        self.config = config
        self.provider = method_identity(config.agent)
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
                config.agent.provider,
                output_dir / "rsi",
                task_count=max(1, training_count),
                seed=config.seed,
                agent=config.agent,
                batch_id=batch_id, resume=resume,
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
            provider=self.provider,
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
            success=bool(record.get("success", False)),
            score=record.get("relative_score"),
            feedback=record.get("error") or (
                "Reference objective score reached." if record.get("success")
                else "Completed session scored below the reference objective."
            ),
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
        "error": f"No response: {error_text}" if is_timeout else error_text,
        "agent_output": None, "simulation": None, "interactions": [], "llm_metrics": {},
    }
    causes = []
    seen = {id(exc)}
    cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        causes.append({"error_type": type(cause).__name__, "error": str(cause)})
        cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
    if causes:
        record["error_chain"] = causes
    if task.game == "heros_journey":
        record["interaction_protocol"] = task.task.get("interaction_protocol", "legacy_unlabeled")
        record["score_metric"] = "reference_steps_over_executed_steps_v1"
        record["completion_rate"] = None
    return record


def _method_result(result: dict[str, Any], controller: AgentController) -> dict[str, Any]:
    return runtime_policy(controller.config.agent).normalize_result(result)


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
    output_dir: Path | None = None,
) -> EvaluationOutcome:
    try:
        task_data = task.task
        if task.game == "gameworld":
            task_data = {**task.task, "task_id": task.run_id}
            if output_dir is not None:
                task_data["gameworld_results_root"] = str(output_dir / "gameworld")
        return evaluate_task(
            task.game, task_data, request=request, agent=agent, reload_agent=reload_agent
        )
    except Exception as exc:
        return EvaluationOutcome(_error_result(task, exc))


@dataclass
class _ValidationCost:
    tokens: int = 0
    seconds: float = 0.0


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
    costs: _ValidationCost | None = None,
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
    costs = costs if costs is not None else _ValidationCost()
    candidate_ids: set[str] = set()
    for candidate in batch.candidates:
        candidate_id = str(candidate.candidate_id)
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("candidate ids must be non-empty and unique")
        candidate_ids.add(candidate_id)
        candidate_run_id = task.run_id
        if task.game == "gameworld":
            identity = resume_state.digest([candidate_id, stage_index])[:16]
            candidate_run_id = f"{task.run_id}__candidate_{identity}"
        candidate_task = PlannedTask(
            run_id=candidate_run_id,
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
        try:
            outcome = _evaluate(
                candidate_task,
                request=controller.request_for_agent(candidate.agent),
                agent=candidate.agent,
                output_dir=output_dir,
            )
        finally:
            candidate_seconds = time.perf_counter() - validation_started
            costs.seconds += candidate_seconds
        result = outcome.record
        costs.tokens += _token_total(result.get("llm_metrics", {}))
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
    return selection, summaries, candidate_outcomes, costs.tokens, costs.seconds


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
            and runtime_policy(controller.config.agent).reload_holdout_agent
            and controller.session is not None
            else None
        )
        outcome = _evaluate(
            task,
            request=controller.request,
            agent=controller.active_agent,
            reload_agent=reload_agent,
            output_dir=output_dir,
        )
        result = _method_result(outcome.record, controller)
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
        and not _is_fatal_method_error(controller.config, result)
        and not _is_provider_failure(result)
    ):
        try:
            previous_telemetry = controller.session.refinement_telemetry
            refinement_started = time.perf_counter()
            try:
                controller.refine(task, result, ordinal, feedback=outcome.feedback)
            except Exception as exc:
                telemetry = controller.session.refinement_telemetry
                if (
                    isinstance(telemetry, Mapping)
                    and telemetry is not previous_telemetry
                    and telemetry.get("task_index", ordinal) == ordinal
                ):
                    refinement = dict(telemetry)
                else:
                    # Failure before the session publishes this round's telemetry
                    # must not reuse the previous round or imply free model calls.
                    refinement = {
                        "status": "error", "method": controller.provider,
                        "task_index": ordinal,
                        "duration_seconds": round(time.perf_counter() - refinement_started, 6),
                        "duration_method": "runner_wall_clock",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "tokens": rsi_session.refinement_token_metrics(None),
                    }
                raise
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
                stage_costs = _ValidationCost()
                try:
                    selection, evaluations, _, stage_tokens, stage_seconds = _evaluate_candidates(
                        task, batch, parent_evaluation, controller=controller, output_dir=output_dir,
                        ordinal=ordinal, stage_index=len(stages), costs=stage_costs,
                    )
                    stage_costs.tokens = stage_tokens
                    stage_costs.seconds = stage_seconds
                finally:
                    # Completed validation remains billable even if selection or
                    # a later candidate fails before the stage returns.
                    candidate_validation_tokens += stage_costs.tokens
                    candidate_validation_seconds += stage_costs.seconds
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
                "status": (stages[-1]["status"] if stages else
                           "completed" if controller.provider == "rsi" else "unchanged"),
                "committed_candidate_id": (
                    stages[-1]["committed_candidate_id"] if stages else None
                ),
            }
        except Exception as exc:
            error = runtime_policy(controller.config.agent).selection_error(exc)
            selection_summary = {"status": "error", "error_type": type(exc).__name__, "error": error}
            result = {
                **result,
                "parent_status": result.get("status"),
                "parent_error_type": result.get("error_type"),
                "parent_error": result.get("error"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": error,
            }
    refinement_seconds = float(refinement.get("duration_seconds", 0.0)) if isinstance(refinement, dict) else 0.0
    refinement_tokens: int | None = 0
    if isinstance(refinement, dict):
        value = (refinement.get("tokens") or {}).get("total_tokens")
        refinement_tokens = (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )
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
        "total_tokens": (
            _token_total(result.get("llm_metrics", {})) + refinement_tokens + candidate_validation_tokens
            if refinement_tokens is not None else None
        ),
    }
    if task.phase == "train" and not task.replay:
        record["post_refinement_score"] = post_refinement_score(record)
    resume_state.atomic_json(output_dir / "runs" / task.phase / f"{ordinal:04d}_{task.run_id}.json", record)
    score = record.get("relative_score")
    score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
    print(
        f"[{task.phase}] {ordinal:04d} {task.game} d{task.difficulty} "
        f"replay={task.replay} status={record['status']} relative_score={score_text}",
        flush=True,
    )
    return record


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
        index = int(newest.name[:4])
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


def _run_target_benchmark(
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
        if method_identity(config.agent) not in BUILTIN_METHODS:
            raise ValueError(f"Resume is not implemented for {config.agent.provider}")
    else:
        manifest = build_manifest(config)
        _write_json(output_dir / "config.resolved.json", config.as_dict())
        _write_json(output_dir / "task_manifest.json", manifest.as_dict())
        _persist_tasks(output_dir, manifest)

    refinement_count = sum(not task.replay for task in manifest.training)
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
        for index, task in enumerate(manifest.training, start=1):
            if index <= len(training_records):
                continue
            record = _run_one(
                task, controller=controller, ordinal=index, output_dir=output_dir
            )
            training_records.append(record)
            if _must_stop_after_error(config, record):
                break

        # Holdout starts only after all scheduled evolution/refinement has
        # finished. AgentController.begin loads but never refines the final RSI version.
        if len(training_records) == len(manifest.training) and (
            resuming_holdout or config.continue_on_error
            or all(item["status"] in {"ok", "skipped"} for item in training_records)
        ) and not any(_is_fatal_method_error(config, item) for item in training_records):
            for index, task in enumerate(manifest.holdout, start=1):
                if index <= len(holdout_records):
                    continue
                record = _run_one(task, controller=controller, ordinal=index, output_dir=output_dir)
                holdout_records.append(record)
                if _must_stop_after_error(config, record):
                    break
    finally:
        controller.close()

    return _finish_benchmark(config, output_dir, manifest, controller,
                             training_records, holdout_records)


def _finish_benchmark(config, output_dir, manifest, controller,
                      training_records, holdout_records, extra_summary=None):
    """Keep the existing metric contract for both recovery modes."""
    refinement_count = sum(not task.replay for task in manifest.training)
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
                runtime_policy(config.agent).scheduled_refinements(refinement_count, controller.session.adaptor)
                if controller.session is not None
                else refinement_count
            ),
            "completed_training_runs": len(training_records),
            "scheduled_holdout_runs": len(manifest.holdout),
            "completed_holdout_runs": len(holdout_records),
            "training_errors": sum(item["status"] == "error" for item in training_records),
            "holdout_errors": sum(item["status"] == "error" for item in holdout_records),
            "training_skipped": sum(item["status"] == "skipped" for item in training_records),
            "holdout_skipped": sum(item["status"] == "skipped" for item in holdout_records),
            **(extra_summary or {}),
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


def _checkpoint_path(output_dir: Path, phase: str, ordinal: int) -> Path:
    return output_dir / "checkpoints" / f"{phase}_{ordinal:04d}.json"


def _run_path(output_dir: Path, phase: str, ordinal: int, task: PlannedTask) -> Path:
    return output_dir / "runs" / phase / f"{ordinal:04d}_{task.run_id}.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_tasks(manifest):
    return [(phase, i, task) for phase, tasks in (
        ("train", manifest.training), ("holdout", manifest.holdout)
    ) for i, task in enumerate(tasks, 1)]


def _restore_checkpoint_chain(output_dir, manifest, initial, controller, records):
    """Trust only contiguous, hashed run/state pairs."""
    previous, committed = initial, 0
    all_tasks = _checkpoint_tasks(manifest)
    for phase, index, task in all_tasks:
        path = _checkpoint_path(output_dir, phase, index)
        if not path.is_file():
            break
        checkpoint = _read_json(path)
        record = _read_json(_run_path(output_dir, phase, index, task))
        if (checkpoint.get("config_hash") != manifest.config_hash
                or checkpoint.get("phase") != phase or checkpoint.get("ordinal") != index
                or checkpoint.get("run_id") != task.run_id
                or not _checkpoint_is_stable(controller.config, task, record)
                or record.get("run_id") != task.run_id or record.get("phase") != phase
                or checkpoint.get("record_digest") != resume_state.digest(record)
                or checkpoint.get("parent_digest") != resume_state.digest(previous)):
            raise RuntimeError(f"Corrupt or mismatched checkpoint for {phase} {index}")
        records[phase].append(record)
        previous, committed = checkpoint, committed + 1
    if any(_checkpoint_path(output_dir, p, i).exists() for p, i, _ in all_tasks[committed:]):
        raise RuntimeError("Non-contiguous checkpoints; refusing resume")
    if any(_run_path(output_dir, p, i, t).exists() for p, i, t in all_tasks[committed + 1:]):
        raise RuntimeError("Runs continued beyond an uncommitted task; use --target-dir after manual review")
    resume_state.restore(controller.session, previous["state"])
    _clear_committed_marker(output_dir)
    return previous


def _clear_committed_marker(output_dir):
    path = output_dir / "inflight.json"
    if not path.is_file():
        return
    marker = _read_json(path)
    checkpoint_path = _checkpoint_path(output_dir, marker["phase"], marker["ordinal"])
    if checkpoint_path.is_file():
        checkpoint = _read_json(checkpoint_path)
        if any(checkpoint.get(key) != marker.get(key) for key in ("run_id", "parent_digest")):
            raise RuntimeError("Stale in-flight marker differs from committed checkpoint")
        path.unlink()


def _archive_interrupted(output_dir, controller, task, ordinal):
    """Preserve uncommitted evidence and mark its possible cost explicitly."""
    paths = [_run_path(output_dir, task.phase, ordinal, task), output_dir / "inflight.json"]
    if task.phase == "train" and not task.replay:
        session = controller.session
        paths.append(session.artifact_dir / f"task_{ordinal:04d}_{rsi_session._safe_name(task.run_id)}")
        paths.extend(runtime_policy(controller.config.agent).interrupted_paths(session.adaptor))
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    archive = output_dir / "abandoned_attempts" / f"{task.phase}_{ordinal:04d}_{uuid4().hex}"
    archive.mkdir(parents=True, exist_ok=False)
    resume_state.atomic_json(archive / "attempt.json", {
        "phase": task.phase, "ordinal": ordinal, "run_id": task.run_id,
        "token_coverage": "unknown_possible_billed_requests",
    })
    for index, path in enumerate(existing):
        path.rename(archive / f"{index:02d}_{path.name}")


def _prepare_interrupted(output_dir, controller, task, ordinal, previous):
    path = output_dir / "inflight.json"
    if path.is_file():
        marker = _read_json(path)
        expected = {"phase": task.phase, "ordinal": ordinal, "run_id": task.run_id,
                    "parent_digest": resume_state.digest(previous)}
        if any(marker.get(key) != value for key, value in expected.items()):
            raise RuntimeError("In-flight identity differs; manual review required")
    if task.game == "gameworld":
        from benches.gameworld.benchmark import step_resume_enabled
        if step_resume_enabled(task.task):
            # Keep durable browser/model receipts, but never repeat paid evolution.
            refine_path = controller.session.artifact_dir / (
                f"task_{ordinal:04d}_{rsi_session._safe_name(task.run_id)}")
            if task.phase == "train" and not task.replay and refine_path.exists():
                raise RuntimeError("Evolution refinement was interrupted; no automatic paid retry")
            return
    _archive_interrupted(output_dir, controller, task, ordinal)


def _commit_checkpoint(output_dir, manifest, controller, task, index, record, previous):
    checkpoint = {"config_hash": manifest.config_hash, "phase": task.phase,
                  "ordinal": index, "run_id": task.run_id,
                  "record_digest": resume_state.digest(record),
                  "parent_digest": resume_state.digest(previous),
                  "refined": task.phase == "train" and not task.replay,
                  "state": resume_state.export(controller.session)}
    resume_state.atomic_json(_checkpoint_path(output_dir, task.phase, index), checkpoint)
    (output_dir / "inflight.json").unlink(missing_ok=True)
    return checkpoint


def _run_checkpoint_task(config, output_dir, manifest, controller, task, index, cursor):
    if cursor["resume"] and cursor["committable"]:
        _prepare_interrupted(output_dir, controller, task, index, cursor["previous"])
    if cursor["committable"]:
        resume_state.atomic_json(output_dir / "inflight.json", {
            "phase": task.phase, "ordinal": index, "run_id": task.run_id,
            "parent_digest": resume_state.digest(cursor["previous"]), "cost_if_interrupted": "unknown",
        })
    record = _run_one(task, controller=controller, ordinal=index, output_dir=output_dir)
    stable = _checkpoint_is_stable(config, task, record)
    if cursor["committable"] and stable:
        cursor["previous"] = _commit_checkpoint(
            output_dir, manifest, controller, task, index, record, cursor["previous"])
        if task.phase == "train" and not task.replay:
            cursor["new_train_runs"] += 1
    elif not stable:
        cursor["committable"] = False
    # Never run beyond an uncommitted state: resume must retry this task first.
    cursor["stopped"] = not stable or _must_stop_after_error(config, record)
    return record


def _execute_checkpoint_tasks(config, output_dir, manifest, controller, records, cursor, limit):
    for phase, tasks in (("train", manifest.training), ("holdout", manifest.holdout)):
        for index in range(len(records[phase]) + 1, len(tasks) + 1):
            record = _run_checkpoint_task(config, output_dir, manifest, controller,
                                          tasks[index - 1], index, cursor)
            records[phase].append(record)
            if limit is not None and cursor["new_train_runs"] >= limit:
                cursor["paused"] = True
            if cursor["stopped"] or cursor["paused"]:
                return


def _run_checkpoint_benchmark(config, output_dir, *, resume, max_new_train_runs):
    manifest = build_manifest(config)
    initial_path = output_dir / "checkpoints" / "initial.json"
    if resume:
        if (resume_state.digest(_read_json(output_dir / "config.resolved.json")) != resume_state.digest(config.as_dict())
                or resume_state.digest(_read_json(output_dir / "task_manifest.json")) != resume_state.digest(manifest.as_dict())):
            raise RuntimeError("Config/task manifest changed; refusing checkpoint resume")
        initial = _read_json(initial_path)
        if initial.get("config_hash") != manifest.config_hash:
            raise RuntimeError("Initial checkpoint config hash differs")
    else:
        _write_json(output_dir / "config.resolved.json", config.as_dict())
        _write_json(output_dir / "task_manifest.json", manifest.as_dict())
        _persist_tasks(output_dir, manifest)
        initial = None
    controller = AgentController(config, output_dir, sum(not t.replay for t in manifest.training),
        batch_id=initial["batch_id"] if resume else None, resume=resume)
    return _run_checkpoint_controller(config, output_dir, manifest, controller,
                                      initial, resume, max_new_train_runs)


def _run_checkpoint_controller(config, output_dir, manifest, controller, initial, resume, limit):
    records = {"train": [], "holdout": []}
    try:
        if resume:
            previous = _restore_checkpoint_chain(output_dir, manifest, initial, controller, records)
        else:
            previous = {"config_hash": manifest.config_hash, "batch_id": controller.session.batch_id,
                        "state": resume_state.export(controller.session)}
            resume_state.atomic_json(output_dir / "checkpoints" / "initial.json", previous)
        cursor = {"previous": previous, "resume": resume, "committable": True,
                  "new_train_runs": 0, "paused": False, "stopped": False}
        _execute_checkpoint_tasks(config, output_dir, manifest, controller, records, cursor, limit)
    finally:
        controller.close()
    return _finish_benchmark(config, output_dir, manifest, controller, records["train"], records["holdout"], {
        "paused_after_checkpoint": cursor["paused"], "checkpoint_chain_complete": cursor["committable"],
        "committed_training_runs": len(list((output_dir / "checkpoints").glob("train_*.json"))),
        "committed_holdout_runs": len(list((output_dir / "checkpoints").glob("holdout_*.json"))),
        "abandoned_attempts": len(list((output_dir / "abandoned_attempts").glob("*/attempt.json"))),
    })


def run_benchmark(config: BenchmarkConfig, *, target_dir: str | Path | None = None,
                  resume: bool = False, max_new_train_runs: int | None = None) -> dict[str, Any]:
    """Use committed checkpoints in place, or the existing target-dir recovery."""
    if max_new_train_runs is not None and (isinstance(max_new_train_runs, bool)
            or not isinstance(max_new_train_runs, int) or max_new_train_runs < 1):
        raise ValueError("max_new_train_runs must be a positive integer")
    if target_dir is not None and (resume or max_new_train_runs is not None):
        raise ValueError("--target-dir cannot be combined with --resume or --max-new-train-runs")
    supported = config.agent.provider in RSI_METHODS
    if not supported and (resume or max_new_train_runs is not None):
        raise ValueError("Checkpoint resume/pause requires an RSI method")
    if target_dir is not None and method_identity(config.agent) == "rsi":
        raise ValueError("External RSI adapters use --resume in place; --target-dir is built-in only")
    if target_dir is not None or not supported:
        return _run_target_benchmark(config, target_dir=target_dir)
    output_dir = Path(config.output_dir).resolve()
    if resume and not (output_dir / "checkpoints" / "initial.json").is_file():
        raise RuntimeError("Missing initial checkpoint; use --target-dir for an older lineage")
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty; choose a fresh directory or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".benchmark.lock").open("a+b") as stream:
        try:
            lock_file(stream)
        except BlockingIOError as exc:
            raise RuntimeError(f"Benchmark already running in {output_dir}") from exc
        try:
            return _run_checkpoint_benchmark(config, output_dir, resume=resume,
                                              max_new_train_runs=max_new_train_runs)
        finally:
            unlock_file(stream)
