"""Manage one benchmark-facing lifecycle session for an RSI method."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .contracts import (
    AgentCallRecord,
    BackendResponse,
    ExecutionStatus,
    FeedbackResult,
    SelectionEvaluation,
    StepFeedback,
    TaskEvaluation,
    TokenUsage,
    normalized_score,
    to_jsonable,
)


from .methods import RSI_PROVIDERS, create_adaptor, method_identity, runtime_policy
def is_fatal_ahe_error(value: Any) -> bool:
    """Compatibility alias; the method owns its error classification."""
    from rsi_methods.AHE.runtime_policy import is_fatal_error
    return is_fatal_error(value)


def _copy_resume_workspace(source: Path, target: Path, method: str) -> tuple[Path, Path, int, str]:
    """Copy one complete benchmark output and locate its single RSI lineage."""
    if not source.is_dir() or not (source / "task_manifest.json").is_file():
        raise ValueError(f"Not a benchmark workspace: {source}")
    if target.exists() or source == target or source in target.parents or target in source.parents:
        raise ValueError(f"Resume target must be a new, separate directory: {target}")
    batches = [path for path in (source / "rsi" / method).iterdir()
               if path.is_dir() and (path / "batch_manifest.json").is_file()]
    if len(batches) != 1:
        raise ValueError(f"Expected one {method} batch under {source}, found {len(batches)}")
    source_batch = batches[0]
    task_numbers = {int(path.name[5:9]) for path in source_batch.glob("task_[0-9][0-9][0-9][0-9]_*")
                    if path.is_dir()}
    records = [json.loads(path.read_text(encoding="utf-8")) for path in
               sorted((source / "runs" / "train").glob("[0-9][0-9][0-9][0-9]_*.json"))]
    config = json.loads((source / "config.resolved.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "task_manifest.json").read_text(encoding="utf-8"))
    if config.get("online_evolution", False):
        raise ValueError("Resuming legacy online_evolution runs is no longer supported")
    indices = list(range(1, len(records) + 1))
    before = max([*task_numbers, *indices], default=1)
    has_holdout = any((source / "runs" / "holdout").glob("*.json"))
    training_complete = len(records) == len(manifest["training"])
    if has_holdout and not training_complete:
        raise ValueError(f"Holdout records exist before training is complete: {source}")
    can_enter_holdout = config.get("continue_on_error", True) or all(
        record.get("status") in {"ok", "skipped"} for record in records
    )
    phase = "holdout" if has_holdout or (training_complete and can_enter_holdout) else "train"
    if phase == "holdout":
        before += 1
    shutil.copytree(source, target)
    return source_batch, target / "rsi" / method / source_batch.name, before, phase


def extract_rsi_answer(method: str, response: BackendResponse) -> str:
    """Return the benchmark answer carried by an RSI backend response."""
    return runtime_policy(method).answer(response)


def backend_response_record(
    method: str,
    prompt: str,
    response: BackendResponse,
) -> dict[str, Any]:
    """Create a JSON-safe record used by telemetry and later refinement."""
    reasoning = runtime_policy(method).reasoning(response)
    return {
        "prompt": prompt,
        "response_text": response.text,
        "reasoning": reasoning,
        "usage": to_jsonable(response.usage),
        "provider": response.provider,
        "model": response.model,
        "status": response.status,
        "finish_reason": response.finish_reason,
        "metadata": to_jsonable(response.metadata),
    }


def rsi_token_metrics(response: BackendResponse) -> dict[str, Any]:
    usage = response.usage
    if usage is None:
        return {
            "prompt_tokens": None,
            "reasoning_tokens": None,
            "answer_tokens": None,
            "total_tokens": None,
            "method": "unavailable",
            "sources": {
                "prompt_tokens": "unavailable",
                "reasoning_tokens": "unavailable",
                "answer_tokens": "unavailable",
                "total_tokens": "unavailable",
            },
            "includes_all_attempts": True,
        }
    answer_tokens = usage.output_tokens
    if answer_tokens is not None and usage.reasoning_tokens is not None:
        answer_tokens = max(answer_tokens - usage.reasoning_tokens, 0)
    return {
        "prompt_tokens": usage.input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": usage.total_tokens,
        "method": "rsi_backend_reported",
        "sources": {
            "prompt_tokens": "provider_reported",
            "reasoning_tokens": "provider_reported" if usage.reasoning_tokens is not None else "unavailable",
            "answer_tokens": "derived_from_provider_output",
            "total_tokens": "provider_reported" if usage.total_tokens is not None else "unavailable",
        },
        "includes_all_attempts": True,
    }


def refinement_token_metrics(report: Any) -> dict[str, Any]:
    """Aggregate only model usage reported by an RSI refinement operation."""
    ledger = report.get("token_usage_report") if isinstance(report, Mapping) else None
    if isinstance(ledger, Mapping):
        answer = ledger.get("answer_tokens", ledger.get("response_num_tokens"))
        reasoning = ledger.get("reasoning_tokens", ledger.get("reasoning_num_tokens"))
        if "answer_tokens" not in ledger and isinstance(answer, int) and isinstance(reasoning, int):
            answer = max(answer - reasoning, 0)
        return {
            "prompt_tokens": ledger.get("prompt_tokens", ledger.get("prompt_num_tokens")),
            "reasoning_tokens": reasoning,
            "answer_tokens": answer,
            "total_tokens": ledger.get("total_tokens"),
            "method": "rsi_attempt_ledger",
            "complete": ledger.get("complete") is True,
            "no_model_calls": ledger.get("no_model_calls") is True,
        }
    records: list[Mapping[str, Any]] = []

    def collect(value: Any) -> None:
        if is_dataclass(value):
            value = to_jsonable(value)
        if isinstance(value, Mapping):
            if any(
                name in value
                for name in (
                    "prompt_num_tokens",
                    "response_num_tokens",
                    "total_num_tokens",
                    "reasoning_num_tokens",
                )
            ):
                records.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                collect(child)

    collect(report)
    if not records:
        return {
            "prompt_tokens": None,
            "reasoning_tokens": None,
            "answer_tokens": None,
            "total_tokens": None,
            "method": "unavailable",
        }

    def total(name: str) -> int | None:
        values = [record.get(name) for record in records]
        return (
            sum(values)
            if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            else None
        )

    output_tokens = total("response_num_tokens")
    reasoning_tokens = total("reasoning_num_tokens")
    answer_tokens = output_tokens
    if output_tokens is not None and reasoning_tokens is not None:
        answer_tokens = max(output_tokens - reasoning_tokens, 0)
    return {
        "prompt_tokens": total("prompt_num_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": total("total_num_tokens"),
        "method": "rsi_refinement_reported",
    }


class RSIBatchSession:
    """Own one clean RSI lineage and advance it once after every game task."""

    def __init__(
        self,
        method: str,
        artifact_root: str | Path,
        *,
        task_count: int,
        batch_id: str | None = None,
        adaptor: Any | None = None,
        agent: Any | None = None,
        resume_from: str | Path | None = None,
        target_dir: str | Path | None = None,
        resume: bool = False,
        seed: int = 0,
    ) -> None:
        normalized = method.strip().lower()
        if normalized not in RSI_PROVIDERS:
            raise ValueError(f"Unsupported RSI method: {method}")
        if task_count < 1:
            raise ValueError("task_count must be positive")
        if (resume_from is None) != (target_dir is None):
            raise ValueError("resume_from and target_dir must be provided together")
        if resume and resume_from is not None:
            raise ValueError("resume and resume_from are mutually exclusive")
        identity = (method_identity(agent) or normalized) if normalized == "rsi" else normalized
        if resume_from is not None and identity == "rsi":
            raise ValueError("External RSI adapters use --resume in place; --target-dir is built-in only")
        self.resuming = resume
        self.seed = seed
        self.method = identity
        self._provider = normalized
        source_batch = None
        previous_batch = None
        before_task = None
        resume_phase = None
        if resume_from is not None:
            source_batch, previous_batch, before_task, resume_phase = _copy_resume_workspace(
                Path(resume_from).resolve(), Path(target_dir).resolve(), identity
            )
            artifact_root = Path(target_dir).resolve() / "rsi"
        self.resume_before_task_index = before_task
        self.resume_phase = resume_phase
        self.resumed_batch_dir = previous_batch
        self.batch_id = batch_id or uuid4().hex
        self.artifact_dir = Path(artifact_root).resolve() / identity / self.batch_id
        if resume:
            manifest_path = self.artifact_dir / "batch_manifest.json"
            if batch_id is None or not manifest_path.is_file():
                raise RuntimeError("Missing existing RSI batch for checkpoint resume")
            saved_batch = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(saved_batch.get(key) != value for key, value in {
                "method": identity, "batch_id": batch_id, "task_count": task_count,
            }.items()):
                raise RuntimeError("Existing RSI batch identity or task count differs")
        else:
            self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self.agent_config = agent
        self.adaptor = adaptor if adaptor is not None else self._create_adaptor(task_count)
        self._agent: Any | None = None
        self._task_id: str | None = None
        self._pending_batch: Any | None = None
        self._pending_output_dir: Path | None = None
        self._selection_stage = 0
        self._last_refinement_telemetry: dict[str, Any] | None = None
        self._closed = False
        if resume:
            # State restoration follows checkpoint validation in the runner.
            return
        if resume_from is None:
            self.adaptor.init()
        else:
            resume_adaptor = getattr(self.adaptor, "resume", None)
            if not callable(resume_adaptor):
                raise RuntimeError(
                    f"RSI method {self.method!r} does not support resume"
                )
            resume_adaptor(previous_batch, source_batch=source_batch,
                   before_task_index=before_task, target_dir=self.artifact_dir / "state")
        config_path = getattr(self.adaptor, "config_path", None)
        if previous_batch is not None:
            snapshot = previous_batch / "method_config.toml"
            if snapshot.is_file():
                (self.artifact_dir / "method_config.toml").write_bytes(snapshot.read_bytes())
        elif config_path is not None:
            resolved_config = Path(config_path).resolve()
            if resolved_config.is_file():
                snapshot_path = self.artifact_dir / "method_config.toml"
                redact = getattr(self.adaptor, "_redacted_config_text", None)
                if callable(redact):
                    snapshot_path.write_text(redact(), encoding="utf-8")
                else:
                    snapshot_path.write_bytes(resolved_config.read_bytes())
        manifest = {
            "schema_version": 1,
            "method": self.method,
            "batch_id": self.batch_id,
            "task_count": task_count,
            "clean_base": resume_from is None,
            "adaptor": type(self.adaptor).__name__,
            "effective_state": {
                "run_dir": getattr(self.adaptor, "run_dir", None),
                "reuse_existing": getattr(self.adaptor, "reuse_existing", None),
                "max_refinement_rounds": getattr(
                    self.adaptor, "max_refinement_rounds", None
                ),
            },
        }
        if self._provider == "rsi":
            manifest["adapter"] = self.agent_config.adapter
        if source_batch is not None:
            manifest["resumed_from_batch_id"] = source_batch.name
            manifest["resume_before_task_index"] = before_task
        (self.artifact_dir / "batch_manifest.json").write_text(
            json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _create_adaptor(self, task_count: int) -> Any:
        """Construct a method without depending on any game's implementation."""
        return create_adaptor(self._provider, seed=self.seed, artifact_dir=self.artifact_dir,
                              task_count=task_count, agent=self.agent_config,
                              resuming=self.resuming)

    def begin_task(self, task_id: str) -> Any:
        if self._closed:
            raise RuntimeError("RSI batch session is closed")
        if self._pending_batch is not None:
            raise RuntimeError("Pending RSI candidate batch requires selection")
        if self._agent is not None:
            raise RuntimeError("Previous RSI task has not been refined")
        agent = self.adaptor.load()
        self._task_id = str(task_id)
        self._agent = agent
        return agent

    def skip_task(self) -> None:
        """Release a failed evaluation without treating it as refinement."""
        if self._pending_batch is not None:
            raise RuntimeError("Cannot skip a task with pending RSI candidates")
        try:
            self.adaptor.close()
        finally:
            self.end_task()

    def end_task(self) -> None:
        """Release transient task state while retaining committed artifacts."""
        self._agent = None
        self._task_id = None
        self._pending_batch = None
        self._pending_output_dir = None

    def act(self, prompt: str) -> BackendResponse:
        if self._agent is None:
            raise RuntimeError("begin_task() must be called before act()")
        return self._agent.act(prompt)

    @property
    def pending_batch(self) -> Any | None:
        return self._pending_batch

    @property
    def refinement_telemetry(self) -> Mapping[str, Any] | None:
        return self._last_refinement_telemetry

    def refine(self, feedback: FeedbackResult, task_index: int) -> Any | None:
        if self._agent is None:
            raise RuntimeError("begin_task() must be called before refine()")
        if feedback.task_id != self._task_id:
            raise ValueError("feedback task_id does not match the active RSI task")
        output_dir = self.artifact_dir / f"task_{task_index:04d}_{_safe_name(feedback.task_id)}"
        previous_report = getattr(self.adaptor, "last_refinement_report", None)
        export_state = getattr(self.adaptor, "export_state", None)
        state_snapshot = export_state() if callable(export_state) else None
        refinement_called = False
        started = time.perf_counter()
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "feedback.json").write_text(
                json.dumps(to_jsonable(feedback), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            refinement_called = True
            batch = self.adaptor.refine([feedback], output_dir=output_dir)
            if batch is not None:
                _validate_candidate_batch(batch)
                if not callable(getattr(self.adaptor, "select", None)):
                    raise TypeError("An RSI adapter returning candidates must implement select()")
                self._pending_batch = batch
                self._pending_output_dir = output_dir
        except Exception as exc:
            try:
                restore_state = getattr(self.adaptor, "restore_state", None)
                if state_snapshot is None or not callable(restore_state):
                    raise RuntimeError("adapter does not support state recovery")
                restore_state(state_snapshot)
                recovered = True
            except Exception as recovery_error:
                recovered = False
                exc.add_note(f"Could not restore pre-refinement state: {recovery_error}")
            report = getattr(self.adaptor, "last_refinement_report", None)
            if not refinement_called:
                report = {"token_usage_report": {"total_tokens": 0, "complete": True, "no_model_calls": True}}
            elif report is previous_report:
                report = None
            self._last_refinement_telemetry = {
                "status": "error",
                "method": self.method,
                "task_index": task_index,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "artifact_dir": str(output_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "state_recovered": recovered,
                "tokens": refinement_token_metrics(report),
            }
            try:
                (output_dir / "refinement_telemetry.json").write_text(
                    json.dumps(self._last_refinement_telemetry, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as telemetry_error:
                exc.add_note(f"Could not persist refinement telemetry: {telemetry_error}")
            self.end_task()
            raise
        finally:
            # A successful refinement may expose a pending candidate batch
            # that the caller must validate and select next.
            self._agent = None
            self._task_id = None
        report = getattr(self.adaptor, "last_refinement_report", None)
        skipped = isinstance(report, Mapping) and report.get("status") == "skipped"
        tokens = refinement_token_metrics(report)
        if skipped and tokens["total_tokens"] is None and report.get("reason") == "refinement_interval":
            tokens = {
                "prompt_tokens": 0, "reasoning_tokens": 0, "answer_tokens": 0,
                "total_tokens": 0, "method": "skipped",
            }
        self._last_refinement_telemetry = {
            "status": "skipped" if skipped else "ok",
            "method": self.method,
            "task_index": task_index,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "artifact_dir": str(output_dir),
            "tokens": tokens,
        }
        if skipped:
            self._last_refinement_telemetry["reason"] = report.get("reason")
            self._last_refinement_telemetry["refinement_every"] = report.get("refinement_every")
        (output_dir / "refinement_telemetry.json").write_text(
            json.dumps(self._last_refinement_telemetry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return batch

    def select(
        self,
        parent_evaluation: Any,
        candidate_evaluations: Mapping[str, Any],
        *,
        selection_mode: str = "paired",
    ) -> Any:
        batch = self._pending_batch
        if batch is None:
            raise RuntimeError("RSIBatchSession.select() has no pending candidate batch")
        if not isinstance(candidate_evaluations, Mapping):
            raise TypeError("candidate_evaluations must be a mapping")
        if selection_mode != "paired":
            raise ValueError("selection_mode must be 'paired'; online selection is no longer supported")
        expected_ids = _candidate_ids(batch)
        normalized_results = {str(key): value for key, value in candidate_evaluations.items()}
        if set(normalized_results) != expected_ids:
            raise ValueError(
                "candidate_evaluations must contain exactly the pending candidate ids"
            )
        output_dir = self._pending_output_dir or Path(batch.work_dir)
        stage_dir = output_dir / f"selection_stage_{self._selection_stage:04d}"
        self._selection_stage += 1
        stage_dir.mkdir(parents=True, exist_ok=False)
        selection_input = {
            "selection_mode": selection_mode,
            "parent": parent_evaluation,
            "candidates": normalized_results,
        }
        (stage_dir / "selection_input.json").write_text(
            json.dumps(to_jsonable(selection_input), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        selection = self.adaptor.select(
            batch=batch,
            parent_evaluation=parent_evaluation,
            candidate_evaluations=normalized_results,
            output_dir=stage_dir,
            selection_mode=selection_mode,
        )
        status = str(getattr(selection, "status", "")).strip().lower()
        replacement = getattr(selection, "batch", None)
        if status == "needs_validation":
            if replacement is None:
                raise ValueError("needs_validation selection must provide a replacement batch")
            _validate_candidate_batch(replacement)
            self._pending_batch = replacement
            self._pending_output_dir = output_dir
        elif status in {"committed", "completed", "unchanged"}:
            self._pending_batch = None
            self._pending_output_dir = None
        else:
            raise ValueError(f"Unsupported RSI selection status: {status or '<missing>'}")
        return selection

    def close(self) -> None:
        if self._closed:
            return
        try:
            close = getattr(self.adaptor, "close", None)
            if callable(close):
                close()
        finally:
            self.end_task()
            self._closed = True

    def __enter__(self) -> "RSIBatchSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def create_rsi_batch(
    method: str,
    artifact_root: str | Path,
    *,
    task_count: int,
    batch_id: str | None = None,
    agent: Any | None = None,
    resume_from: str | Path | None = None,
    target_dir: str | Path | None = None,
    resume: bool = False,
    seed: int = 0,
) -> RSIBatchSession:
    return RSIBatchSession(
        method,
        artifact_root,
        task_count=task_count,
        batch_id=batch_id,
        agent=agent,
        resume_from=resume_from,
        target_dir=target_dir,
        resume=resume,
        seed=seed,
    )


def execution_status_for_error(error: BaseException) -> ExecutionStatus:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "timeout" in name or "timed out" in message:
        return ExecutionStatus.TIMEOUT
    return ExecutionStatus.ERROR


def is_rsi_infrastructure_error(error: BaseException) -> bool:
    """Identify failures that make the active RSI runtime unusable for the batch."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "modulenotfounderror",
            "docker daemon is unavailable",
            "docker image build failed",
            "docker container creation failed",
            "metaagent command failed",
        )
    )


def agent_call_from_metrics(metrics: Mapping[str, Any]) -> AgentCallRecord | None:
    for attempt in reversed(metrics.get("attempts", [])):
        raw = attempt.get("rsi_agent_call")
        if isinstance(raw, Mapping):
            usage = raw.get("usage")
            return AgentCallRecord(
                response_text=str(raw.get("response_text", "")),
                reasoning=raw.get("reasoning"),
                usage=TokenUsage(**usage) if isinstance(usage, Mapping) else None,
                provider=raw.get("provider"),
                model=raw.get("model"),
                status=raw.get("status"),
                finish_reason=raw.get("finish_reason"),
                metadata=raw.get("metadata"),
            )
    return None


def prompt_from_metrics(metrics: Mapping[str, Any], fallback: str = "") -> str:
    for attempt in reversed(metrics.get("attempts", [])):
        raw = attempt.get("rsi_agent_call")
        if isinstance(raw, Mapping) and isinstance(raw.get("prompt"), str):
            return raw["prompt"]
    return fallback


def make_step_feedback(
    *,
    step_index: int,
    metrics: Mapping[str, Any],
    action: Any,
    environment_response: Any,
    success: bool | None,
    score: float | None,
    feedback: str | None,
    fallback_prompt: str = "",
) -> StepFeedback:
    action_text = action if isinstance(action, str) else json.dumps(action, ensure_ascii=False)
    return StepFeedback(
        step_index=step_index,
        prompt=prompt_from_metrics(metrics, fallback_prompt),
        image_records=(),
        action=action_text,
        environment_response=environment_response,
        success=success,
        score=score,
        feedback=feedback,
        agent_call=agent_call_from_metrics(metrics),
    )


def make_feedback_result(
    *,
    task_id: str,
    task: Any,
    trajectory: Sequence[StepFeedback],
    execution_status: ExecutionStatus,
    success: bool,
    score: float | int | None,
    feedback: str | None,
    metrics: Mapping[str, Any],
) -> FeedbackResult:
    numeric_score = normalized_score(score)
    return FeedbackResult(
        task_id=str(task_id),
        task=task,
        trajectory=tuple(trajectory),
        evaluation=TaskEvaluation(
            execution_status=execution_status,
            success=success,
            score=numeric_score,
            feedback=feedback,
            metrics=dict(metrics),
        ),
    )


def _candidate_ids(batch: Any) -> set[str]:
    candidates = getattr(batch, "candidates", None)
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence) or not candidates:
        raise ValueError("CandidateBatch.candidates must be a non-empty sequence")
    ids = []
    for candidate in candidates:
        candidate_id = str(getattr(candidate, "candidate_id", ""))
        agent = getattr(candidate, "agent", None)
        if not callable(getattr(agent, "act", None)):
            raise TypeError("CandidateSpec.agent must expose callable act")
        ids.append(candidate_id)
    if any(not candidate_id for candidate_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("CandidateBatch candidate ids must be non-empty and unique")
    return set(ids)


def _validate_candidate_batch(batch: Any) -> None:
    _candidate_ids(batch)
    if not hasattr(batch, "work_dir"):
        raise TypeError("CandidateBatch must expose work_dir")


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized[:80] or "task"


__all__ = [
    "BackendResponse",
    "ExecutionStatus",
    "TokenUsage",
    "RSI_PROVIDERS",
    "RSIBatchSession",
    "SelectionEvaluation",
    "agent_call_from_metrics",
    "backend_response_record",
    "create_rsi_batch",
    "execution_status_for_error",
    "is_rsi_infrastructure_error",
    "is_fatal_ahe_error",
    "extract_rsi_answer",
    "make_feedback_result",
    "make_step_feedback",
    "prompt_from_metrics",
    "refinement_token_metrics",
    "rsi_token_metrics",
]
