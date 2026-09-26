"""Manage one benchmark-facing lifecycle session for an RSI method."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .contracts import (
    AgentCallRecord,
    BackendResponse,
    ExecutionStatus,
    FeedbackResult,
    StepFeedback,
    TaskEvaluation,
    TokenUsage,
    to_jsonable,
)


RSI_PROVIDERS = frozenset({"ace", "baseline", "hyperagents", "self_harness", "ahe"})


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
    online = config.get("online_evolution", False)
    indices = [int(record["generation_index"]) for record in records
               if online and record.get("generation_index") is not None]
    if not online:
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
    if online:
        for record in records:
            index = record.get("generation_index")
            selection = record.get("selection")
            if index is not None and index < before and isinstance(selection, dict):
                if selection.get("status") in {"committed", "unchanged", "completed"} and index not in task_numbers:
                    raise ValueError(f"Missing RSI artifact for committed generation {index}: {source_batch}")
    shutil.copytree(source, target)
    return source_batch, target / "rsi" / method / source_batch.name, before, phase


@dataclass(frozen=True, slots=True)
class SelectionEvaluation:
    """Compact benchmark result used only to compare parent and candidates."""

    status: str
    relative_score: float | None
    success: bool
    result_path: str
    error_type: str | None = None
    error: str | None = None


def _extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("RSI agent response does not contain a JSON object")


def extract_rsi_answer(method: str, response: BackendResponse) -> str:
    """Return the benchmark answer carried by an RSI backend response."""
    if method.strip().lower() != "ace":
        return response.text
    ace_metadata = response.metadata.get("ace") if isinstance(response.metadata, Mapping) else None
    if isinstance(ace_metadata, Mapping) and "raw_response_text" in ace_metadata:
        if not isinstance(response.text, str) or not response.text.strip():
            raise ValueError("ACE unwrapped response text is empty or not text")
        return response.text
    envelope = _extract_json_object(response.text)
    if "final_answer" not in envelope:
        raise ValueError("ACE response is missing final_answer")
    answer = envelope["final_answer"]
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False)
    raise ValueError("ACE final_answer must be a string, object, or array")


def backend_response_record(
    method: str,
    prompt: str,
    response: BackendResponse,
) -> dict[str, Any]:
    """Create a JSON-safe record used by telemetry and later refinement."""
    reasoning = response.reasoning
    if method.strip().lower() == "ace":
        try:
            envelope = _extract_json_object(response.text)
            outer_reasoning = envelope.get("reasoning")
            if not reasoning and isinstance(outer_reasoning, str):
                reasoning = outer_reasoning
        except ValueError:
            pass
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
    ) -> None:
        normalized = method.strip().lower()
        if normalized not in RSI_PROVIDERS:
            raise ValueError(f"Unsupported RSI method: {method}")
        if task_count < 1:
            raise ValueError("task_count must be positive")
        if (resume_from is None) != (target_dir is None):
            raise ValueError("resume_from and target_dir must be provided together")
        if resume_from is not None and normalized not in {"ace", "ahe", "hyperagents"}:
            raise ValueError(f"RSI method {normalized!r} does not support resume")
        self.method = normalized
        source_batch = None
        previous_batch = None
        before_task = None
        resume_phase = None
        if resume_from is not None:
            source_batch, previous_batch, before_task, resume_phase = _copy_resume_workspace(
                Path(resume_from).resolve(), Path(target_dir).resolve(), normalized
            )
            artifact_root = Path(target_dir).resolve() / "rsi"
        self.resume_before_task_index = before_task
        self.resume_phase = resume_phase
        self.resumed_batch_dir = previous_batch
        self.batch_id = batch_id or uuid4().hex
        self.artifact_dir = Path(artifact_root).resolve() / normalized / self.batch_id
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
        if resume_from is None:
            self.adaptor.init()
        else:
            resume = getattr(self.adaptor, "resume", None)
            if not callable(resume):
                raise RuntimeError(
                    f"RSI method {self.method!r} does not support resume"
                )
            resume(previous_batch, source_batch=source_batch,
                   before_task_index=before_task, target_dir=self.artifact_dir / "state")
        config_path = getattr(self.adaptor, "config_path", None)
        if previous_batch is not None:
            snapshot = previous_batch / "method_config.toml"
            if snapshot.is_file():
                (self.artifact_dir / "method_config.toml").write_bytes(snapshot.read_bytes())
        elif config_path is not None:
            resolved_config = Path(config_path).resolve()
            if resolved_config.is_file():
                (self.artifact_dir / "method_config.toml").write_bytes(
                    resolved_config.read_bytes()
                )
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
        if source_batch is not None:
            manifest["resumed_from_batch_id"] = source_batch.name
            manifest["resume_before_task_index"] = before_task
        (self.artifact_dir / "batch_manifest.json").write_text(
            json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _create_adaptor(self, task_count: int) -> Any:
        if self.method == "ace":
            from rsi_methods.ACE.ACEAdaptor import ACEAdaptor

            return ACEAdaptor()
        if self.method == "baseline":
            from rsi_methods.Baseline import BaselineAdaptor

            return BaselineAdaptor(backend=self._baseline_backend())
        if self.method == "self_harness":
            from rsi_methods.SelfHarness import SelfHarnessAdaptor

            return SelfHarnessAdaptor()
        if self.method == "ahe":
            from rsi_methods.AHE.AHEAdaptor import AHEAdaptor

            return AHEAdaptor(
                run_dir=self.artifact_dir / "state",
                timeout_seconds=float(getattr(self.agent_config, "timeout", 300.0)),
            )
        try:
            from rsi_methods.HyperAgents.HyperAgentsAdaptor import HyperAgentsAdaptor
        except ModuleNotFoundError as exc:
            requirements = Path(__file__).resolve().parents[1] / "rsi_methods" / "HyperAgents" / "requirements.txt"
            raise RuntimeError(
                "HyperAgents host dependencies are missing. Install them with: "
                f"python -m pip install -r {requirements}"
            ) from exc

        return HyperAgentsAdaptor(
            run_dir=self.artifact_dir / "state",
            reuse_existing=False,
            max_refinement_rounds=task_count,
        )

    def _baseline_backend(self) -> str | None:
        """benchmark 的 agent.backend；未提供时返回 None，交给 BaselineAdaptor 用默认后端。"""
        backend = getattr(self.agent_config, "backend", None)
        return str(backend).strip().lower() if backend else None

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
        self.adaptor.close()
        self._agent = None
        self._task_id = None

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
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "feedback.json").write_text(
            json.dumps(to_jsonable(feedback), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        started = time.perf_counter()
        try:
            batch = self.adaptor.refine([feedback], output_dir=output_dir)
            if batch is not None:
                _validate_candidate_batch(batch)
                self._pending_batch = batch
                self._pending_output_dir = output_dir
        except Exception as exc:
            self._last_refinement_telemetry = {
                "status": "error",
                "method": self.method,
                "task_index": task_index,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "artifact_dir": str(output_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "tokens": refinement_token_metrics(
                    getattr(self.adaptor, "last_refinement_report", None)
                ),
            }
            (output_dir / "refinement_telemetry.json").write_text(
                json.dumps(self._last_refinement_telemetry, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        finally:
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
        if selection_mode not in {"paired", "online"}:
            raise ValueError("selection_mode must be 'paired' or 'online'")
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
        close = getattr(self.adaptor, "close", None)
        if callable(close):
            close()
        self._agent = None
        self._task_id = None
        self._pending_batch = None
        self._pending_output_dir = None
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
) -> RSIBatchSession:
    return RSIBatchSession(
        method,
        artifact_root,
        task_count=task_count,
        batch_id=batch_id,
        agent=agent,
        resume_from=resume_from,
        target_dir=target_dir,
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
    numeric_score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
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
    "extract_rsi_answer",
    "make_feedback_result",
    "make_step_feedback",
    "prompt_from_metrics",
    "refinement_token_metrics",
    "rsi_token_metrics",
]
