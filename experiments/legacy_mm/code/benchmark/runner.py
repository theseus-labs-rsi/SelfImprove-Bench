"""Single execution interface for all games and agent methods."""

from __future__ import annotations

import json
import fcntl
import shutil
from uuid import uuid4
from copy import deepcopy
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import rsi_session, resume_state
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
            "evaluated_count": sum(item.get("status") == "ok" for item in items),
            "rule_success_count": sum(bool(item.get("success")) for item in items if item.get("status") == "ok"),
            "rule_success_rate": (
                sum(bool(item.get("success")) for item in items if item.get("status") == "ok")
                / sum(item.get("status") == "ok" for item in items)
                if any(item.get("status") == "ok" for item in items) else None
            ),
            "mean_score": (
                sum(float(item["relative_score"]) for item in items if item.get("status") == "ok")
                / sum(item.get("status") == "ok" for item in items)
                if any(item.get("status") == "ok" for item in items) else None
            ),
            "rule_results": items,
        }
        for (game, difficulty), items in sorted(groups.items())
    ]


def _token_total(metrics: dict[str, Any]) -> int | None:
    value = metrics.get("tokens", {}).get("total_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class AgentController:
    """Own either a stateless baseline or one evolving RSI lineage."""

    def __init__(self, config: BenchmarkConfig, output_dir: Path, training_count: int, *, batch_id: str | None = None, resume: bool = False) -> None:
        self.config = config
        self.provider = config.agent.provider
        self.bridge: Any | None = None
        self.session: Any | None = None
        self.active_agent: Any | None = None
        if self.provider in RSI_METHODS:
            self.bridge = load_rsi_integration()
            session_agent = config.agent
            if self.provider == "ahe":
                # This is part of the hashed GameWorld config, unlike a global
                # default or an environment toggle. Existing lineages without
                # the explicit setting retain the historical every-five policy.
                gameworld = config.game_options.get("gameworld", {})
                frequency = gameworld.get("ahe_refinement_every", 5)
                if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 1:
                    raise ValueError("game_options.gameworld.ahe_refinement_every must be a positive integer")
                session_agent = SimpleNamespace(
                    **vars(config.agent), ahe_refinement_every=frequency,
                )
            self.session = self.bridge.create_rsi_batch(
                self.provider,
                output_dir / "rsi",
                task_count=max(1, training_count),
                batch_id=batch_id, resume=resume,
                agent=session_agent,
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
        "difficulty": task.difficulty, "relative_score": None, "raw_score": None,
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
) -> tuple[Any, list[dict[str, Any]], int | None, float]:
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
        candidate_task_data = deepcopy(task.task)
        if task.game == "gameworld" and candidate_task_data.get("resumable_steps") is True:
            # Candidate evaluation is a NEW episode, not a replay of the
            # already finished parent. Keep the maze/seed, isolate receipts.
            import hashlib
            candidate_tag = hashlib.sha256(candidate_id.encode()).hexdigest()[:16]
            candidate_task_data["task_id"] = (
                f"{task.task['task_id']}__candidate_{stage_index}_{candidate_tag}")
        candidate_task = PlannedTask(
            run_id=task.run_id,
            phase=task.phase,
            game=task.game,
            difficulty=task.difficulty,
            task=candidate_task_data,
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
        candidate_tokens = _token_total(result.get("llm_metrics", {}))
        validation_tokens = (validation_tokens + candidate_tokens
                             if validation_tokens is not None and candidate_tokens is not None else None)
        artifact_dir = Path(candidate.artifact_dir).resolve()
        path = artifact_dir / "evaluation" / "result.json"
        _write_json(path, {
            **result, "candidate_id": candidate_id, "parent_run_id": task.run_id,
            "selection_stage_index": stage_index,
        })
        if result.get("status") != "ok":
            raise RuntimeError(
                f"Candidate validation failed for {candidate_id}: "
                f"{result.get('error_type', 'error')}: {result.get('error', 'unknown')}"
            )
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
        and result.get("status") == "ok"
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
                candidate_validation_tokens = (
                    candidate_validation_tokens + stage_tokens
                    if candidate_validation_tokens is not None and stage_tokens is not None else None
                )
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
            # A failed refinement/selection may already have spent tokens without a report.
            candidate_validation_tokens = None
            selection_summary = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
            result = {**result, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    refinement_seconds = float(refinement.get("duration_seconds", 0.0)) if isinstance(refinement, dict) else 0.0
    refinement_tokens = (_token_total(refinement) if isinstance(refinement, dict)
                         else None if selection_summary and selection_summary.get("status") == "error" else 0)
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
            if result.get("status") == "ok" and _token_total(result.get("llm_metrics", {})) is not None
            and (refinement is None or refinement_tokens is not None)
            and candidate_validation_tokens is not None else None
        ),
    }
    record["token_coverage"] = {
        "complete": record["status"] == "ok" and record["total_tokens"] is not None,
        "inference": "reported" if record["inference_tokens"] is not None else "unknown",
        "refinement": "reported" if record["refinement_tokens"] is not None else "unknown",
        "candidate_validation": "reported" if record["candidate_validation_tokens"] is not None else "unknown",
        "unknown_cost_blocks_token_per_gain": record["total_tokens"] is None,
    }
    resume_state.atomic_json(output_dir / "runs" / task.phase / f"{ordinal:04d}_{task.run_id}.json", record)
    score = record.get("relative_score")
    score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
    print(
        f"[{task.phase}] {ordinal:04d} {task.game} d{task.difficulty} "
        f"replay={task.replay} status={record['status']} relative_score={score_text}",
        flush=True,
    )
    return record


def _checkpoint_path(output_dir: Path, phase: str, ordinal: int) -> Path:
    return output_dir / "checkpoints" / f"{phase}_{ordinal:04d}.json"


def _run_path(output_dir: Path, phase: str, ordinal: int, task: PlannedTask) -> Path:
    return output_dir / "runs" / phase / f"{ordinal:04d}_{task.run_id}.json"


def _archive_interrupted(output_dir: Path, controller: AgentController, task: PlannedTask, ordinal: int) -> dict[str, Any]:
    """Move uncommitted evidence aside; never treat its unknown cost as zero."""
    old = _run_path(output_dir, task.phase, ordinal, task)
    inflight = output_dir / "inflight.json"
    session = controller.session
    artifacts = [old]
    # Keep the previous partial report too; the new report is a different attempt.
    previous_report = output_dir / "result.json"
    if previous_report.exists():
        artifacts.append(previous_report)
    if session is not None and task.phase == "train" and not task.replay:
        artifacts.append(session.artifact_dir / f"task_{ordinal:04d}_{rsi_session._safe_name(task.run_id)}")
        if session.method == "ahe":
            iteration = sum(1 for p in output_dir.joinpath("checkpoints").glob("train_*.json")
                             if json.loads(p.read_text(encoding="utf-8")).get("refined")) + 1
            state = session.adaptor.run_dir
            artifacts += [state / "runs" / f"iteration_{iteration:03d}",
                          state / "candidates" / f"iteration_{iteration:03d}"]
    if not inflight.exists() and not any(p.exists() for p in artifacts if p != previous_report):
        return {}
    if inflight.exists():
        marker = json.loads(inflight.read_text(encoding="utf-8"))
        if marker.get("phase") != task.phase or marker.get("ordinal") != ordinal or marker.get("run_id") != task.run_id:
            raise RuntimeError("Unexpected in-flight run marker; refusing unsafe resume")
    archive = output_dir / "abandoned_attempts" / f"{task.phase}_{ordinal:04d}_{uuid4().hex}"
    archive.mkdir(parents=True, exist_ok=False)
    try:
        old_record = json.loads(old.read_text(encoding="utf-8")) if old.is_file() else None
    except (OSError, json.JSONDecodeError):
        old_record = None  # Preserve truncated/unreadable evidence, never count it as zero.
    summary = {"run_id": task.run_id, "phase": task.phase, "ordinal": ordinal,
               "evidence_dir": str(archive), "status": "uncommitted_retry",
               "reported_tokens": old_record.get("total_tokens") if old_record else None,
               "reported_components": ({key: old_record.get(key) for key in (
                   "inference_tokens", "refinement_tokens", "candidate_validation_tokens")}
                   if old_record else None),
               "token_coverage": "unknown_possible_billed_requests", "moved_artifacts": {}}
    # The attempt marker precedes the destructive renames. A crash in the
    # middle of archival still leaves an auditable, unknown-cost attempt.
    resume_state.atomic_json(archive / "attempt.json", summary)
    for path in artifacts:
        if path.exists():
            target = archive / ("run.json" if path == old else "previous_result.json" if path == previous_report
                                else f"artifact_{len(list(archive.iterdir()))}_{path.name}")
            shutil.move(str(path), str(target))
            summary["moved_artifacts"][str(path)] = str(target)
    if inflight.exists():
        shutil.move(str(inflight), str(archive / "inflight.json"))
        summary["moved_artifacts"][str(inflight)] = str(archive / "inflight.json")
    resume_state.atomic_json(archive / "attempt.json", summary)
    with (output_dir / "abandoned_attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return summary


def run_benchmark(config: BenchmarkConfig, *, resume: bool = False,
                  max_new_train_runs: int | None = None) -> dict[str, Any]:
    """Execute with per-run committed state. --resume never trusts loose run files."""
    if max_new_train_runs is not None and (isinstance(max_new_train_runs, bool)
                                           or not isinstance(max_new_train_runs, int)
                                           or max_new_train_runs < 1):
        raise ValueError("max_new_train_runs must be a positive integer")
    output_dir = Path(config.output_dir).resolve()
    if resume and config.agent.provider not in {"ace", "ahe", "hyperagents"}:
        raise RuntimeError("Checkpoint resume is supported for ACE, AHE and HyperAgents only")
    if resume and not (output_dir / "checkpoints" / "initial.json").is_file():
        raise RuntimeError(
            f"No initial checkpoint in {output_dir}; old or partially initialized runs cannot be resumed safely"
        )
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty; choose a fresh output_dir or use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".benchmark.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Benchmark already running in {output_dir}") from exc
        return _run_locked(config, output_dir, resume=resume,
                           max_new_train_runs=max_new_train_runs)


def _run_locked(config: BenchmarkConfig, output_dir: Path, *, resume: bool,
                max_new_train_runs: int | None = None) -> dict[str, Any]:
    manifest = build_manifest(config)
    if resume:
        if (resume_state.digest(json.loads((output_dir / "config.resolved.json").read_text(encoding="utf-8"))) != resume_state.digest(config.as_dict())
                or resume_state.digest(json.loads((output_dir / "task_manifest.json").read_text(encoding="utf-8"))) != resume_state.digest(manifest.as_dict())):
            raise RuntimeError("Config/task manifest changed since original run; refusing resume")
        initial = json.loads((output_dir / "checkpoints" / "initial.json").read_text(encoding="utf-8"))
        if initial.get("config_hash") != manifest.config_hash:
            raise RuntimeError("Initial checkpoint config hash mismatch")
    else:
        _write_json(output_dir / "config.resolved.json", config.as_dict())
        _write_json(output_dir / "task_manifest.json", manifest.as_dict())
        _persist_tasks(output_dir, manifest)

    refinement_count = sum(not task.replay for task in manifest.training)
    controller = AgentController(config, output_dir, refinement_count,
        batch_id=initial.get("batch_id") if resume else None, resume=resume)
    training_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    new_train_runs = 0
    paused_after_checkpoint = False
    try:
        if resume:
            if controller.session is None:
                raise RuntimeError("Resume currently requires an RSI method")
            previous = initial
            committed = 0
            all_tasks = [("train", i, t) for i, t in enumerate(manifest.training, 1)] + [
                ("holdout", i, t) for i, t in enumerate(manifest.holdout, 1)]
            for phase, index, task in all_tasks:
                checkpoint_path = _checkpoint_path(output_dir, phase, index)
                if not checkpoint_path.exists():
                    break
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                record = json.loads(_run_path(output_dir, phase, index, task).read_text(encoding="utf-8"))
                if (checkpoint.get("config_hash") != manifest.config_hash
                        or checkpoint.get("phase") != phase or checkpoint.get("ordinal") != index
                        or checkpoint.get("run_id") != task.run_id or record.get("status") != "ok"
                        or checkpoint.get("record_digest") != resume_state.digest(record)
                        or checkpoint.get("parent_digest") != resume_state.digest(previous)):
                    raise RuntimeError(f"Corrupt or mismatched checkpoint for {phase} {index}")
                (training_records if phase == "train" else holdout_records).append(record)
                previous = checkpoint
                committed += 1
            if any(_checkpoint_path(output_dir, phase, i).exists() for phase, i, _ in all_tasks[committed:]):
                raise RuntimeError("Non-contiguous committed checkpoints; refusing resume")
            resume_state.restore(controller.session, previous["state"])
            stale_marker = output_dir / "inflight.json"
            if stale_marker.is_file():
                marker = json.loads(stale_marker.read_text(encoding="utf-8"))
                committed_path = _checkpoint_path(output_dir, marker["phase"], marker["ordinal"])
                if committed_path.is_file():
                    committed_marker = json.loads(committed_path.read_text(encoding="utf-8"))
                    if (committed_marker.get("run_id") != marker.get("run_id")
                            or committed_marker.get("parent_digest") != marker.get("parent_digest")):
                        raise RuntimeError("Stale in-flight marker does not match committed checkpoint")
                    stale_marker.unlink()
            print(f"Resume: {len(training_records)} training and {len(holdout_records)} holdout runs committed", flush=True)
        else:
            initial = {"config_hash": manifest.config_hash,
                       "batch_id": controller.session.batch_id if controller.session else None,
                       "state": resume_state.export(controller.session) if controller.session and config.agent.provider in {"ace", "ahe", "hyperagents"} else None}
            resume_state.atomic_json(output_dir / "checkpoints" / "initial.json", initial)
            previous = initial

        for phase, tasks, records in (("train", manifest.training, training_records),
                                      ("holdout", manifest.holdout, holdout_records)):
            if phase == "holdout" and len(training_records) != len(manifest.training):
                break
            for index in range(len(records) + 1, len(tasks) + 1):
                task = tasks[index - 1]
                if resume:
                    if task.game == "gameworld" and task.task.get("resumable_steps") is True:
                        # Resuming a committed game step must NOT archive its live
                        # browser/model receipt and start another paid episode.
                        marker_path = output_dir / "inflight.json"
                        if marker_path.exists():
                            marker = json.loads(marker_path.read_text(encoding="utf-8"))
                            if (marker.get("phase") != phase or marker.get("ordinal") != index
                                    or marker.get("run_id") != task.run_id
                                    or marker.get("parent_digest") != resume_state.digest(previous)):
                                raise RuntimeError("Astray inflight identity differs; manual audit required")
                            # If refinement/selection already started, the model
                            # may have been billed after the game ended. Never
                            # restart that evolution stage automatically.
                            refine_path = output_dir / "rsi" / f"task_{index:04d}_{rsi_session._safe_name(task.run_id)}"
                            if refine_path.exists():
                                raise RuntimeError("Evolution refinement was interrupted; no automatic paid retry")
                    else:
                        _archive_interrupted(output_dir, controller, task, index)
                resume_state.atomic_json(output_dir / "inflight.json", {
                    "phase": phase, "ordinal": index, "run_id": task.run_id,
                    "parent_digest": resume_state.digest(previous),
                    "cost_if_interrupted": "unknown"})
                record = _run_one(task, controller=controller, ordinal=index, output_dir=output_dir)
                if record["status"] != "ok" or (phase == "train" and not task.replay and controller.session is not None
                     and (record.get("selection") or {}).get("status") not in {"committed", "completed", "unchanged"}):
                    records.append(record)
                    print("Run failed; stopping at this run for safe resume (no evolution advance).", flush=True)
                    break
                state = resume_state.export(controller.session) if controller.session and config.agent.provider in {"ace", "ahe", "hyperagents"} else None
                checkpoint = {"config_hash": manifest.config_hash, "phase": phase,
                              "ordinal": index, "run_id": task.run_id,
                              "record_digest": resume_state.digest(record),
                              "parent_digest": resume_state.digest(previous),
                              "refined": phase == "train" and not task.replay,
                              "state": state}
                resume_state.atomic_json(_checkpoint_path(output_dir, phase, index), checkpoint)
                (output_dir / "inflight.json").unlink(missing_ok=True)
                previous = checkpoint
                records.append(record)
                if phase == "train" and not task.replay:
                    new_train_runs += 1
                    if max_new_train_runs is not None and new_train_runs >= max_new_train_runs:
                        paused_after_checkpoint = True
                        print(f"Paused safely after {new_train_runs} newly committed training run(s).", flush=True)
                        break
            if paused_after_checkpoint:
                break
            if records and records[-1]["status"] != "ok":
                break
    finally:
        controller.close()

    metrics = calculate_metrics(
        training_records,
        holdout_records,
        evolution_window=config.evolution_window,
        expected_training_runs=len(manifest.training),
    )
    abandoned_root = output_dir / "abandoned_attempts"
    abandoned_attempts = (sum(p.is_dir() for p in abandoned_root.iterdir())
                          if abandoned_root.is_dir() else 0)
    if abandoned_attempts:
        # Retries can include billed calls with unavailable usage; successful
        # round totals alone cannot be reported as the experiment cost.
        metrics["evolution_cost"]["total_tokens"] = None
        metrics["evolution_cost"]["tokens_per_absolute_gain"] = None
    report = {
        "schema_version": 1,
        "config_hash": manifest.config_hash,
        "config": config.as_dict(),
        "summary": {
            "paused_after_checkpoint": paused_after_checkpoint,
            "abandoned_attempts": abandoned_attempts,
            "scheduled_training_runs": len(manifest.training),
            "attempted_training_runs": len(training_records),
            "completed_training_runs": sum(item["status"] == "ok" for item in training_records),
            "committed_training_runs": sum(_checkpoint_path(output_dir, "train", i).is_file()
                                           for i in range(1, len(manifest.training) + 1)),
            "scheduled_holdout_runs": len(manifest.holdout),
            "attempted_holdout_runs": len(holdout_records),
            "completed_holdout_runs": sum(item["status"] == "ok" for item in holdout_records),
            "committed_holdout_runs": sum(_checkpoint_path(output_dir, "holdout", i).is_file()
                                          for i in range(1, len(manifest.holdout) + 1)),
            "training_errors": sum(item["status"] != "ok" for item in training_records),
            "holdout_errors": sum(item["status"] != "ok" for item in holdout_records),
            "training_with_complete_tokens": sum(item.get("token_coverage", {}).get("complete") is True
                                                 for item in training_records),
            "holdout_with_complete_tokens": sum(item.get("token_coverage", {}).get("complete") is True
                                                for item in holdout_records),
            "full_token_coverage": (
                len(training_records) == len(manifest.training)
                and len(holdout_records) == len(manifest.holdout)
                and abandoned_attempts == 0
                and all(item.get("token_coverage", {}).get("complete") is True
                        for item in training_records + holdout_records)
            ),
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
