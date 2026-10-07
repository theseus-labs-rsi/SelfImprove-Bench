"""RSI lifecycle adaptor for versioned DeepSeek Harness candidates."""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmark.contracts import ExecutionStatus, FeedbackResults, to_jsonable

from .runtime_agent import DSHRuntimeConfig, HyperDSHAgent
from .versioning import (
    CandidateSnapshot,
    initialize_hyperdsh,
    materialize_snapshot,
    persist_version,
    snapshot_dict,
    validate_harness,
    validate_harness_patch,
    write_active_snapshot,
    with_lineage,
)


CONFIG_PATH = Path(__file__).with_name("config.toml")
_SECRET_ASSIGNMENT = re.compile(
    r"^(?P<prefix>\s*(?:api_key|admin_api_key)\s*=\s*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'[^']*')"
    r"(?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class HyperDSHCandidateSpec:
    candidate_id: str
    parent_id: str
    artifact_dir: Path
    agent: HyperDSHAgent
    snapshot: CandidateSnapshot
    changed: bool
    patch_path: Path | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "artifact_dir": str(self.artifact_dir),
            "changed": self.changed,
            "patch_path": str(self.patch_path) if self.patch_path else None,
            "snapshot": snapshot_dict(self.snapshot),
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HyperDSHCandidateBatch:
    refinement_id: str
    stage: str
    work_dir: Path
    candidates: tuple[HyperDSHCandidateSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "refinement_id": self.refinement_id,
            "stage": self.stage,
            "work_dir": str(self.work_dir),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class HyperDSHSelectionReport:
    refinement_id: str
    selected_candidate_id: str | None
    parent_id: str
    changed: bool
    parent_score: float | None
    candidate_score: float | None
    candidate_status: str
    candidate_success: bool
    accepted: bool
    reason: str
    committed_version_dir: str | None = None


@dataclass(frozen=True, slots=True)
class HyperDSHSelectionOutput:
    status: str
    committed_candidate_id: str | None
    report: HyperDSHSelectionReport


class HyperDSHAdaptor:
    """Evolve DSH profile surfaces while preserving a clean initial lineage."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        run_dir: str | Path | None = None,
        reuse_existing: bool = False,
        max_refinement_rounds: int | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if reuse_existing:
            raise ValueError("HyperDSH only supports clean benchmark lineages")
        if max_refinement_rounds is not None and max_refinement_rounds <= 0:
            raise ValueError("max_refinement_rounds must be positive")
        self.config_path = Path(config_path or CONFIG_PATH).resolve()
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else None
        self.reuse_existing = reuse_existing
        self._round_override = max_refinement_rounds
        self._model_override = model
        self._timeout_override = timeout_seconds
        self.max_refinement_rounds = max_refinement_rounds or 1
        self.meta_model: dict[str, Any] = {}
        self._fixed_meta_environment: dict[str, str] = {}
        self.dsh_config: DSHRuntimeConfig | None = None
        self._snapshot: CandidateSnapshot | None = None
        self._pending_batch: HyperDSHCandidateBatch | None = None
        self._refinement_round = 0
        self._accepted_generation = 0
        self._docker_client: Any | None = None
        self._agents: list[HyperDSHAgent] = []
        self.last_refinement_report: dict[str, Any] | None = None
        self.last_selection_report: HyperDSHSelectionReport | None = None

    @property
    def snapshot(self) -> CandidateSnapshot:
        if self._snapshot is None:
            raise RuntimeError("HyperDSHAdaptor.init() must be called first")
        return self._snapshot

    def init(self) -> None:
        self.close()
        config = self._load_config()
        self._configure(config)
        if self.run_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            self.run_dir = project_root / "outputs" / "hyperdsh" / f"run_{uuid4().hex}"
        self._snapshot = initialize_hyperdsh(self.run_dir)
        self._pending_batch = None
        self._refinement_round = 0
        self._accepted_generation = 0
        self.last_refinement_report = None
        self.last_selection_report = None
        (self.run_dir / "method_config.toml").write_text(
            self._redacted_config_text(), encoding="utf-8"
        )

    def build(self) -> HyperDSHAgent:
        return self._agent_for_snapshot(
            self.snapshot,
            self.snapshot.run_dir / "task_runs" / str(self.snapshot.candidate_id) / uuid4().hex,
        )

    def load(self) -> HyperDSHAgent:
        # A fresh home per benchmark task prevents session and settings leakage.
        return self.build()

    def refine(
        self,
        feedback_results: FeedbackResults,
        *,
        output_dir: str | Path,
    ) -> HyperDSHCandidateBatch:
        if self._pending_batch is not None:
            raise RuntimeError("The previous HyperDSH refinement still requires selection")
        if not feedback_results:
            raise ValueError("feedback_results must not be empty")
        next_round = self._refinement_round + 1
        if next_round > self.max_refinement_rounds:
            raise RuntimeError("HyperDSH refinement budget is exhausted")

        resolved_output = Path(output_dir).resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)
        refinement_id = f"refine_{next_round:04d}_{uuid4().hex[:8]}"
        candidate_id = f"{refinement_id}_candidate_000"
        candidate_dir = resolved_output / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        evidence = _feedback_to_result(feedback_results)
        metadata = _new_generation_metadata(
            candidate_dir,
            candidate_id=candidate_id,
            parent_id=self.snapshot.candidate_id,
            run_baseline=None,
            refine_requested=True,
        )

        try:
            from rsi_methods.HyperAgents.lifecycle import (
                instantiate_agent,
                refine as refine_candidate,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HyperDSH refinement dependencies are missing; install "
                "rsi_methods/HyperDSH/requirements.txt"
            ) from exc

        optimizer = instantiate_agent(
            self.snapshot,
            runtime=self._meta_runtime(),
            run_id=uuid4().hex,
            artifact_dir=candidate_dir / "optimizer_artifacts",
        )
        try:
            transition = refine_candidate(
                optimizer,
                evidence,
                output_dir=candidate_dir,
                iterations_left=max(self.max_refinement_rounds - next_round, 0),
                metadata=metadata,
            )
            if transition.patch_path is not None:
                validate_harness_patch(transition.patch_path)
            candidate_snapshot = transition.agent.snapshot
            if not transition.changed:
                candidate_snapshot = with_lineage(
                    self.snapshot,
                    candidate_id=candidate_id,
                    parent_id=self.snapshot.candidate_id,
                    lineage_patches=self.snapshot.lineage_patches,
                )
            else:
                self._validate_snapshot(candidate_snapshot, candidate_dir)
        finally:
            optimizer.close()

        agent = self._agent_for_snapshot(
            candidate_snapshot, candidate_dir / "dsh_validation_runtime"
        )
        report = {
            "round": next_round,
            "changed": transition.changed,
            "candidate_id": candidate_id,
            "parent_id": str(self.snapshot.candidate_id),
            "patch_path": str(transition.patch_path) if transition.patch_path else None,
            "metadata": dict(transition.metadata),
        }
        self.last_refinement_report = report
        (candidate_dir / "refinement_report.json").write_text(
            json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        candidate = HyperDSHCandidateSpec(
            candidate_id=candidate_id,
            parent_id=str(self.snapshot.candidate_id),
            artifact_dir=candidate_dir,
            agent=agent,
            snapshot=candidate_snapshot,
            changed=transition.changed,
            patch_path=transition.patch_path,
            metadata=dict(transition.metadata),
        )
        batch = HyperDSHCandidateBatch(
            refinement_id=refinement_id,
            stage="candidates_ready",
            work_dir=resolved_output,
            candidates=(candidate,),
        )
        (resolved_output / "candidate_batch.json").write_text(
            json.dumps(batch.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._pending_batch = batch
        self._refinement_round = next_round
        return batch

    def select(
        self,
        batch: HyperDSHCandidateBatch,
        *,
        parent_evaluation: Any,
        candidate_evaluations: Mapping[str, Any],
        output_dir: str | Path,
    ) -> HyperDSHSelectionOutput:
        if self._pending_batch is None:
            raise RuntimeError("HyperDSHAdaptor.select() has no pending candidate batch")
        if batch is not self._pending_batch:
            raise RuntimeError("HyperDSHAdaptor.select() received a stale candidate batch")
        candidate = batch.candidates[0]
        if {str(key) for key in candidate_evaluations} != {candidate.candidate_id}:
            raise ValueError("candidate_evaluations must match the pending HyperDSH candidate")
        if candidate.parent_id != str(self.snapshot.candidate_id):
            raise RuntimeError("Committed HyperDSH state changed after refinement")

        _, parent_score, _ = _selection_evaluation(parent_evaluation, "parent")
        candidate_status, candidate_score, candidate_success = _selection_evaluation(
            candidate_evaluations[candidate.candidate_id], "candidate"
        )
        if not candidate.changed:
            accepted, reason = False, "candidate_has_no_change"
        elif candidate_status != "ok":
            accepted, reason = False, "candidate_status_not_ok"
        elif parent_score is None or candidate_score is None:
            accepted, reason = False, "selection_score_missing"
        elif candidate_score < parent_score:
            accepted, reason = False, "selection_score_regressed"
        else:
            accepted, reason = True, "changed_candidate_score_not_below_parent"

        version_dir: Path | None = None
        if accepted:
            self._accepted_generation += 1
            version_dir = self.snapshot.run_dir / "versions" / (
                f"generation_{self._accepted_generation:04d}_{candidate.candidate_id}"
            )
            persist_version(candidate.snapshot, version_dir)
            self._snapshot = candidate.snapshot
            write_active_snapshot(self.snapshot.run_dir, candidate.snapshot)

        report = HyperDSHSelectionReport(
            refinement_id=batch.refinement_id,
            selected_candidate_id=candidate.candidate_id if accepted else None,
            parent_id=candidate.parent_id,
            changed=candidate.changed,
            parent_score=parent_score,
            candidate_score=candidate_score,
            candidate_status=candidate_status,
            candidate_success=candidate_success,
            accepted=accepted,
            reason=reason,
            committed_version_dir=str(version_dir) if version_dir else None,
        )
        self._persist_selection(Path(output_dir), report)
        self._persist_selection(self.snapshot.run_dir, report)
        self.last_selection_report = report
        self._pending_batch = None
        return HyperDSHSelectionOutput(
            status="committed" if accepted else "unchanged",
            committed_candidate_id=candidate.candidate_id if accepted else None,
            report=report,
        )

    def close(self) -> None:
        for agent in self._agents:
            agent.close()
        self._agents.clear()
        self._pending_batch = None

    def _agent_for_snapshot(
        self, snapshot: CandidateSnapshot, runtime_root: Path
    ) -> HyperDSHAgent:
        if self.dsh_config is None:
            raise RuntimeError("HyperDSH runtime is not configured")
        agent = HyperDSHAgent(
            snapshot,
            runtime_root=runtime_root,
            artifact_dir=runtime_root / "artifacts",
            config=self.dsh_config,
        )
        self._agents.append(agent)
        return agent

    def _validate_snapshot(self, snapshot: CandidateSnapshot, output_dir: Path) -> None:
        target = output_dir / "candidate_validation"
        materialized = materialize_snapshot(snapshot, target)
        validate_harness(materialized / "harness")

    def _meta_runtime(self) -> Any:
        if self._docker_client is None:
            import docker

            self._docker_client = docker.DockerClient()
        environment_names = {
            self.meta_model["api_key_env"],
            self.meta_model["base_url_env"],
        }
        environment = {
            name: os.environ[name]
            for name in environment_names
            if name and os.environ.get(name)
        }
        environment.update(self._fixed_meta_environment)
        try:
            from rsi_methods.HyperAgents.lifecycle import HyperAgentRuntime
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HyperDSH refinement dependencies are missing; install "
                "rsi_methods/HyperDSH/requirements.txt"
            ) from exc
        return HyperAgentRuntime(
            docker_client=self._docker_client,
            meta_model=self.meta_model,
            task_model=self.meta_model,
            llm_environment=environment,
        )

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open("rb") as handle:
            return tomllib.load(handle)

    def _configure(self, config: Mapping[str, Any]) -> None:
        meta = dict(config["meta_agent"])
        backend_name = str(meta.get("backend", "openai"))
        backend = config["meta_backend"][backend_name]
        self.meta_model = {
            "name": str(meta["model"]),
            "backend": backend_name,
            "reasoning_effort": meta.get("reasoning_effort"),
            "max_output_tokens": int(meta.get("max_output_tokens", 32768)),
            "api_mode": str(backend.get("api_mode", "responses")),
            "api_key_env": str(backend.get("api_key_env", "OPENAI_API_KEY")),
            "base_url_env": str(backend.get("base_url_env", "OPENAI_BASE_URL")),
        }
        self._fixed_meta_environment = self._fixed_environment(
            backend,
            api_key_env=self.meta_model["api_key_env"],
            base_url_env=self.meta_model["base_url_env"],
        )
        dsh = config["dsh"]
        task = config["task_agent"]
        provider = config["task_provider"]
        model = self._model_override or str(task["model"])
        timeout = self._timeout_override or float(dsh.get("timeout_seconds", 300))
        effort = str(task.get("reasoning_effort", "")).strip() or None
        self.dsh_config = DSHRuntimeConfig(
            command=str(dsh.get("command", "dsh")),
            profile=str(dsh.get("profile", "hyperdsh")),
            timeout_seconds=float(timeout),
            telemetry_disabled=bool(dsh.get("telemetry_disabled", True)),
            provider=str(task["provider"]),
            model=model,
            reasoning_effort=effort,
            max_output_tokens=int(task.get("max_output_tokens", 65536)),
            api=str(provider.get("api", "openai-completions")),
            base_url=str(provider.get("base_url", "")),
            api_key_env=str(provider.get("api_key_env", "OPENAI_API_KEY")),
            api_key=str(provider.get("api_key") or "") or None,
            base_url_env=str(provider.get("base_url_env", "OPENAI_BASE_URL")),
            max_signal_retries=int(dsh.get("signal_retries", 1)),
        )
        configured_rounds = int(config.get("refine", {}).get("max_refinement_rounds", 1))
        self.max_refinement_rounds = self._round_override or configured_rounds

    def _redacted_config_text(self) -> str:
        """Return a reproducible config snapshot without plaintext API keys."""
        source = self.config_path.read_text(encoding="utf-8")
        return _SECRET_ASSIGNMENT.sub(
            lambda match: (
                f'{match.group("prefix")}\"<redacted>\"'
                f'{match.group("suffix")}'
            ),
            source,
        )

    @staticmethod
    def _fixed_environment(
        backend: Mapping[str, Any],
        *,
        api_key_env: str,
        base_url_env: str,
    ) -> dict[str, str]:
        environment: dict[str, str] = {}
        api_key = backend.get("api_key")
        base_url = backend.get("base_url")
        if api_key not in (None, ""):
            environment[api_key_env] = str(api_key)
        if base_url not in (None, ""):
            environment[base_url_env] = str(base_url)
        return environment

    @staticmethod
    def _persist_selection(path: Path, report: HyperDSHSelectionReport) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "selection.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _selection_evaluation(value: Any, label: str) -> tuple[str, float | None, bool]:
    def field(name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    status = str(field("status") or "").strip().lower()
    if not status:
        raise ValueError(f"{label} evaluation status must not be empty")
    raw_score = field("relative_score")
    if raw_score is None:
        score = None
    elif isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ValueError(f"{label} evaluation score must be finite or null")
    else:
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"{label} evaluation score must be finite or null")
    success = field("success")
    if not isinstance(success, bool):
        raise TypeError(f"{label} evaluation success must be bool")
    return status, score, success


def _new_generation_metadata(
    output_dir: Path,
    *,
    candidate_id: str,
    parent_id: str | int | None,
    run_baseline: str | None,
    refine_requested: bool,
) -> dict[str, Any]:
    return {
        "gen_output_dir": str(output_dir),
        "current_genid": candidate_id,
        "parent_genid": parent_id,
        "run_baseline": run_baseline,
        "prev_patch_files": [],
        "curr_patch_files": [],
        "refine_attempted": refine_requested,
        "meta_agent_command_successful": None,
        "patch_produced": False,
        "compilation_successful": False if refine_requested else None,
        "refine_successful": False if refine_requested else None,
        "run_eval": not refine_requested,
    }


def _feedback_to_result(feedback_results: Sequence[Any]) -> dict[str, Any]:
    completed = all(
        result.evaluation.execution_status == ExecutionStatus.COMPLETED
        for result in feedback_results
    )
    success_count = sum(result.evaluation.success for result in feedback_results)
    count = len(feedback_results)
    cases = []
    for result in feedback_results:
        evaluation = result.evaluation
        cases.append(
            {
                "case_id": result.task_id,
                "metrics": {
                    "execution_status": evaluation.execution_status.value,
                    "success": evaluation.success,
                    "score": evaluation.score,
                    **to_jsonable(evaluation.metrics),
                },
                "feedback": {
                    "summary": evaluation.feedback,
                    "task": to_jsonable(result.task),
                    "trajectory": to_jsonable(result.trajectory),
                },
                "artifacts": {},
            }
        )
    return {
        "schema_version": 1,
        "valid": completed,
        "selection_score": success_count / count if completed else None,
        "metrics": {
            "task_count": count,
            "success_count": success_count,
            "success_rate": success_count / count,
            "score_mean": sum(item.evaluation.score for item in feedback_results) / count,
        },
        "feedback": "Inspect the complete per-case task evidence.",
        "cases": cases,
        "artifacts": {},
        "cost": {},
        "error": None if completed else "One or more benchmark tasks did not complete.",
    }


__all__ = [
    "HyperDSHAdaptor",
    "HyperDSHCandidateBatch",
    "HyperDSHCandidateSpec",
    "HyperDSHSelectionOutput",
    "HyperDSHSelectionReport",
]
