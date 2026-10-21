"""Overall Control Loop adaptor for the HyperAgents implementation."""

from __future__ import annotations

import json
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmark.token_ledger import hyperagents_usage_report
from benchmark.interfaces import MethodContext
from rsi_methods.checkpoint import MethodCheckpointMixin
from benchmark.contracts import (
    ExecutionStatus,
    FeedbackResult,
    FeedbackResults,
    to_jsonable,
)

from .candidate import CandidateSnapshot, initialize_agent, with_lineage
from .lifecycle import (
    HyperAgentRuntime,
    InstantiatedAgent,
    instantiate_agent,
    new_generation_metadata,
    refine as refine_candidate,
)
from .runtime_agent import RuntimeAgent
from .search_archive import SearchArchive


CONFIG_PATH = Path(os.environ.get("RSI_HYPERAGENTS_CONFIG_PATH", Path(__file__).with_name("config.toml")))
_LLM_ENVIRONMENT_NAMES = {
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "AZURE_API_VERSION",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "QWEN_API_KEY",
    "QWEN_BASE_URL",
    # Runtime agents execute inside a Docker container.  Forward the exact
    # transport route selected by the supervised launcher instead of silently
    # falling back to proxy variables baked into the runtime image.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
}


@dataclass(frozen=True, slots=True)
class HyperAgentsCandidateSpec:
    """One benchmarkable workspace candidate derived from committed state."""

    candidate_id: str
    parent_id: str
    artifact_dir: Path
    agent: RuntimeAgent
    snapshot: CandidateSnapshot
    instantiated: InstantiatedAgent
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
            "snapshot": _snapshot_to_dict(self.snapshot),
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HyperAgentsCandidateBatch:
    """Single-candidate batch returned through the common refinement shape."""

    refinement_id: str
    stage: str
    work_dir: Path
    candidates: tuple[HyperAgentsCandidateSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "refinement_id": self.refinement_id,
            "stage": self.stage,
            "work_dir": str(self.work_dir),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class HyperAgentsSelectionReport:
    """Record the benchmark comparison and resulting state decision."""

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
    next_parent_id: str | None = None
    parent_selection: str = "score_child_prop"
    archived_candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class HyperAgentsSelectionOutput:
    """Public result returned after accepting or rejecting the candidate."""

    status: str
    committed_candidate_id: str | None
    report: HyperAgentsSelectionReport


class HyperAgentsAdaptor(MethodCheckpointMixin):
    """Expose HyperAgents through the common RSIMethod lifecycle."""

    checkpoint_method = "hyperagents"

    def __init__(
        self,
        context: MethodContext | None = None,
        *,
        run_dir: str | Path | None = None,
        reuse_existing: bool = False,
        max_refinement_rounds: int | None = None,
    ) -> None:
        self.bind_context(context)
        if context is not None:
            run_dir = context.artifact_dir / "state"
            max_refinement_rounds = context.task_count
        if reuse_existing:
            raise ValueError("HyperAgentsAdaptor only supports clean workspaces")
        if max_refinement_rounds is not None and max_refinement_rounds <= 0:
            raise ValueError("max_refinement_rounds must be positive")
        self.config_path = CONFIG_PATH
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else None
        self.reuse_existing = reuse_existing
        self._max_refinement_rounds_override = max_refinement_rounds
        self.workspace_root: Path | None = None
        self.workspace_dir: Path | None = None
        self.initial_source: Path | None = None
        self.meta_model: dict[str, Any] = {}
        self.task_model: dict[str, Any] = {}
        self.max_refinement_rounds = max_refinement_rounds or 1
        self._config_snapshot = b""
        self._docker_client: Any | None = None
        self._snapshot: CandidateSnapshot | None = None
        self._instantiated: InstantiatedAgent | None = None
        self._refinement_round = 0
        self.last_refinement_report: dict[str, Any] | None = None
        self.last_selection_report: HyperAgentsSelectionReport | None = None
        self._pending_batch: HyperAgentsCandidateBatch | None = None
        self._archive_seed = context.seed if context is not None else 0
        self._archive = SearchArchive(seed=self._archive_seed)
        self.parent_selection = "score_child_prop"

    @property
    def snapshot(self) -> CandidateSnapshot:
        if self._snapshot is None:
            raise RuntimeError("HyperAgentsAdaptor.init() must be called first")
        return self._snapshot

    def init(self) -> None:
        """Load method configuration and construct the initial candidate."""
        self.close()
        self._snapshot = None
        self._configure(self._load_config())
        self.workspace_dir = self._create_workspace()
        (self.workspace_dir / "method_config.toml").write_bytes(
            self._config_snapshot
        )
        self._snapshot = initialize_agent(
            self.workspace_dir,
            initial_source=self.initial_source,
            run_config={"schema_version": 1, "purpose": "overall_control_loop"},
            reuse_existing=False,
        )
        self._refinement_round = 0
        self._archive = SearchArchive(seed=self._archive_seed)
        self._archive.add(self.snapshot, score=None, valid_parent=True)
        self.last_refinement_report = None
        self.last_selection_report = None
        self._pending_batch = None
        self._persist_search(self.workspace_dir)

    def _persist_search(self, output_dir: Path) -> None:
        from benchmark.resume_state import atomic_json

        atomic_json(output_dir / "search_archive.json", {
            **self._archive.to_dict(), "active_candidate_id": str(self.snapshot.candidate_id),
            "refinement_round": self._refinement_round,
        })

    def resume(
        self, batch_dir: str | Path, *, source_batch: Path,
        before_task_index: int, target_dir: str | Path,
    ) -> None:
        """Restore a complete archive from copied, completed training rounds."""
        self.close()
        batch = Path(batch_dir).resolve()
        state = batch / "state"
        workspaces = [path for path in state.glob("run_*") if path.is_dir()]
        if len(workspaces) != 1:
            raise ValueError(f"Expected one HyperAgents workspace under {state}")
        self.run_dir = state
        self._config_snapshot = (batch / "method_config.toml").read_bytes()
        self._configure(tomllib.loads(self._config_snapshot.decode("utf-8")))
        self.workspace_dir = workspaces[0]
        archive_marker = self.workspace_dir / "search_archive.json"
        if not archive_marker.is_file():
            raise RuntimeError("HyperAgents batch predates archive restoration; use the historical code for that run")
        initial = initialize_agent(self.workspace_dir, reuse_existing=True)
        self._snapshot = initial
        self._refinement_round = 0
        self._archive = SearchArchive()
        self._archive.add(initial, score=None, valid_parent=True)
        for task in sorted(batch.glob("task_[0-9][0-9][0-9][0-9]_*")):
            if int(task.name[5:9]) >= before_task_index:
                continue
            archive_path = task / "search_archive.json"
            if not archive_path.is_file():
                raise RuntimeError(f"Missing archive checkpoint; do not mix historical search policies: {task}")
            payload = json.loads(archive_path.read_text(encoding="utf-8"))

            def relocate(raw: str) -> str:
                original = Path(raw).resolve()
                if not original.is_relative_to(source_batch.resolve()):
                    raise ValueError(f"HyperAgents archive path is outside source batch: {raw}")
                return str(batch / original.relative_to(source_batch.resolve()))

            for node in payload["nodes"]:
                saved = node["snapshot"]
                if saved["root_commit"] != initial.root_commit:
                    raise ValueError(f"HyperAgents root commit changed: {archive_path}")
                for key in ("run_dir", "workspace"):
                    saved[key] = relocate(saved[key])
                saved["lineage_patches"] = [relocate(p) for p in saved["lineage_patches"]]
                if node["artifact_dir"]:
                    node["artifact_dir"] = relocate(node["artifact_dir"])
            self._archive = SearchArchive.from_dict(payload, root=batch)
            self._snapshot = self._archive.nodes[payload["active_candidate_id"]]["snapshot"]
            self._refinement_round = payload["refinement_round"]
        self.last_refinement_report = None
        self.last_selection_report = None
        self._pending_batch = None
        self._persist_search(self.workspace_dir)

    def build(self) -> RuntimeAgent:
        """Instantiate the committed candidate in an isolated container."""
        if self._snapshot is None:
            raise RuntimeError("HyperAgentsAdaptor.init() must be called first")
        if self._instantiated is None:
            self._instantiated = instantiate_agent(
                self._snapshot,
                runtime=self._runtime(),
                run_id=uuid4().hex,
                artifact_dir=self.snapshot.run_dir / "runtime_artifacts",
            )
        return self._instantiated.task_agent()

    def load(self) -> RuntimeAgent:
        """Return a stateless runtime proxy for the current candidate."""
        return self.build()

    def refine(
        self,
        feedback_results: FeedbackResults,
        *,
        output_dir: str | Path,
    ) -> HyperAgentsCandidateBatch:
        """Generate one isolated candidate without committing it."""
        if self._snapshot is None:
            raise RuntimeError("HyperAgentsAdaptor.init() must be called first")
        if self._pending_batch is not None:
            raise RuntimeError(
                "The previous HyperAgents refinement still requires selection"
            )
        if not feedback_results:
            raise ValueError("feedback_results must not be empty")

        next_round = self._refinement_round + 1
        resolved_output = Path(output_dir).resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)
        refinement_id = f"refine_{next_round:04d}_{uuid4().hex[:8]}"
        candidate_id = f"{refinement_id}_candidate_000"
        candidate_dir = resolved_output / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        evidence = self._feedback_to_result(feedback_results)
        # The current parent has now been observed on a training task.
        parent_node = self._archive.nodes[str(self.snapshot.candidate_id)]
        if evidence["valid"]:
            parent_node["score"] = evidence["selection_score"]
        metadata = new_generation_metadata(
            candidate_dir,
            candidate_id=candidate_id,
            parent_id=self.snapshot.candidate_id,
            run_baseline=None,
            refine_requested=True,
        )
        candidate_instance = instantiate_agent(
            self.snapshot,
            runtime=self._runtime(),
            run_id=uuid4().hex,
            artifact_dir=candidate_dir / "runtime_artifacts",
        )
        try:
            transition = refine_candidate(
                candidate_instance,
                evidence,
                output_dir=candidate_dir,
                iterations_left=max(self.max_refinement_rounds - next_round, 0),
                metadata=metadata,
                search_archive=self._archive,
            )
        except Exception:
            candidate_instance.close()
            raise
        if not transition.changed:
            transition.agent.snapshot = with_lineage(
                self.snapshot,
                candidate_id=candidate_id,
                parent_id=self.snapshot.candidate_id,
                lineage_patches=self.snapshot.lineage_patches,
            )

        self.last_refinement_report = {
            "round": next_round,
            "changed": transition.changed,
            "candidate_id": candidate_id,
            "parent_id": str(self.snapshot.candidate_id),
            "patch_path": str(transition.patch_path) if transition.patch_path else None,
            "metadata": dict(transition.metadata),
            "token_usage_report": hyperagents_usage_report(candidate_dir / "agent_output" / "meta_agent_history.json"),
        }
        (candidate_dir / "refinement_report.json").write_text(
            json.dumps(
                to_jsonable(self.last_refinement_report),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        candidate = HyperAgentsCandidateSpec(
            candidate_id=candidate_id,
            parent_id=str(self.snapshot.candidate_id),
            artifact_dir=candidate_dir,
            agent=transition.agent.task_agent(),
            snapshot=transition.agent.snapshot,
            instantiated=transition.agent,
            changed=transition.changed,
            patch_path=transition.patch_path,
            metadata=dict(transition.metadata),
        )
        batch = HyperAgentsCandidateBatch(
            refinement_id=refinement_id,
            stage="candidates_ready",
            work_dir=resolved_output,
            candidates=(candidate,),
        )
        (resolved_output / "candidate_batch.json").write_text(
            json.dumps(batch.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._refinement_round = next_round
        self._pending_batch = batch
        return batch

    def select(
        self,
        batch: HyperAgentsCandidateBatch,
        *,
        parent_evaluation: Any,
        candidate_evaluations: Mapping[str, Any],
        output_dir: str | Path,
        selection_mode: str = "paired",
    ) -> HyperAgentsSelectionOutput:
        """Archive the candidate, then sample the next parent without greedy rejection."""
        pending_batch = self._pending_batch
        if pending_batch is None:
            raise RuntimeError("HyperAgentsAdaptor.select() has no pending candidate batch")
        if batch is not pending_batch:
            raise RuntimeError("HyperAgentsAdaptor.select() received a stale candidate batch")
        if not isinstance(candidate_evaluations, Mapping):
            raise TypeError("candidate_evaluations must be a mapping")
        if selection_mode not in {"paired", "online"}:
            raise ValueError("selection_mode must be 'paired' or 'online'")

        candidate = batch.candidates[0]
        expected_ids = {candidate.candidate_id}
        provided_ids = {str(key) for key in candidate_evaluations}
        if provided_ids != expected_ids:
            raise ValueError(
                "candidate_evaluations must contain exactly the pending "
                f"candidate id: {candidate.candidate_id}"
            )
        if candidate.parent_id != str(self.snapshot.candidate_id):
            raise RuntimeError("Committed HyperAgents state changed after refinement")

        candidate_evaluation = candidate_evaluations[candidate.candidate_id]
        parent_status, parent_score, _ = _selection_evaluation(parent_evaluation, "parent")
        candidate_status, candidate_score, candidate_success = _selection_evaluation(
            candidate_evaluation, "candidate"
        )
        if not candidate.changed:
            accepted, reason = False, "candidate_has_no_change"
        elif candidate_status != "ok":
            accepted, reason = False, "candidate_status_not_ok"
        elif candidate_score is None:
            accepted, reason = False, "selection_score_missing"
        else:
            accepted, reason = True, "valid_candidate_added_to_archive"
        parent_node = self._archive.nodes[candidate.parent_id]
        if parent_status == "ok" and parent_score is not None:
            parent_node["score"] = parent_score
        self._archive.add(
            candidate.snapshot, score=candidate_score, valid_parent=accepted,
            artifact_dir=candidate.artifact_dir,
        )
        # Final evaluation uses the last round's valid output, not archive ranking.
        final_round = self._refinement_round >= self.max_refinement_rounds
        policy = "final_round" if final_round else self.parent_selection
        next_snapshot = (
            candidate.snapshot if accepted else self.snapshot
        ) if final_round else self._archive.choose(policy)
        selected_candidate_id = str(next_snapshot.candidate_id)
        active_changed = next_snapshot.candidate_id != self.snapshot.candidate_id
        report = HyperAgentsSelectionReport(
            refinement_id=batch.refinement_id,
            selected_candidate_id=selected_candidate_id,
            parent_id=candidate.parent_id,
            changed=candidate.changed,
            parent_score=parent_score,
            candidate_score=candidate_score,
            candidate_status=candidate_status,
            candidate_success=candidate_success,
            accepted=accepted,
            reason=reason,
            next_parent_id=str(next_snapshot.candidate_id),
            parent_selection=policy,
            archived_candidate_id=candidate.candidate_id,
        )
        resolved_output = Path(output_dir).resolve()
        if self.workspace_dir is None:
            raise RuntimeError("HyperAgents workspace is not initialized")
        self._persist_selection_report(resolved_output, report)
        previous_instance = self._instantiated
        keep_candidate = next_snapshot.candidate_id == candidate.snapshot.candidate_id
        keep_previous = next_snapshot.candidate_id == self.snapshot.candidate_id
        if previous_instance is not None and not keep_previous:
            previous_instance.close()
        if not keep_candidate and candidate.instantiated is not previous_instance:
            candidate.instantiated.close()
        self._persist_active_snapshot(self.workspace_dir, next_snapshot, report)
        self._snapshot = next_snapshot
        self._instantiated = candidate.instantiated if keep_candidate else (
            previous_instance if keep_previous else None
        )
        self._persist_search(self.workspace_dir)
        self._persist_search(batch.work_dir)
        self.last_selection_report = report
        self._pending_batch = None
        return HyperAgentsSelectionOutput(
            status="committed" if active_changed else "unchanged",
            committed_candidate_id=selected_candidate_id if active_changed else None,
            report=report,
        )

    def close(self) -> None:
        """Release the active container while retaining persistent lineage."""
        if self._pending_batch is not None:
            for candidate in self._pending_batch.candidates:
                if candidate.instantiated is not self._instantiated:
                    candidate.instantiated.close()
            self._pending_batch = None
        if self._instantiated is not None:
            self._instantiated.close()
            self._instantiated = None

    @staticmethod
    def _persist_selection_report(
        output_dir: Path,
        report: HyperAgentsSelectionReport,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "selection.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _persist_active_snapshot(
        workspace_dir: Path,
        snapshot: CandidateSnapshot,
        report: HyperAgentsSelectionReport,
    ) -> None:
        (workspace_dir / "active_snapshot.json").write_text(
            json.dumps(_snapshot_to_dict(snapshot), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (workspace_dir / "selection.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _runtime(self) -> HyperAgentRuntime:
        if self._docker_client is None:
            import docker

            # A framework-scoped DOCKER_HOST keeps this lineage off the shared
            # daemon while preserving the default socket for existing workers.
            self._docker_client = docker.from_env() if os.environ.get("DOCKER_HOST") else docker.DockerClient()
        environment_names = {
            self.meta_model["api_key_env"],
            self.meta_model["base_url_env"],
            self.task_model["api_key_env"],
            self.task_model["base_url_env"],
        }
        return HyperAgentRuntime(
            docker_client=self._docker_client,
            meta_model=self.meta_model,
            task_model=self.task_model,
            llm_environment=self._collect_llm_environment(environment_names),
        )

    def _load_config(self) -> dict[str, Any]:
        self._config_snapshot = self.config_path.read_bytes()
        return tomllib.loads(self._config_snapshot.decode("utf-8"))

    def _configure(self, config: Mapping[str, Any]) -> None:
        state = config["state"]
        self.workspace_root = self._resolve_path(state["workspace_root"])
        if self.run_dir is not None:
            self.workspace_root = self.run_dir
        source = state.get("initial_source")
        self.initial_source = (
            self._resolve_path(source)
            if source
            else Path(__file__).with_name("hyperagents").resolve()
        )
        self.meta_model = self._configured_model(config, "meta_agent")
        self.task_model = self._configured_model(config, "task_agent")
        self.parent_selection = config.get("refine", {}).get("parent_selection", "score_child_prop")
        if self.parent_selection not in {"random", "latest", "best", "score_prop", "score_child_prop"}:
            raise ValueError(f"Unknown HyperAgents parent_selection: {self.parent_selection}")
        self.max_refinement_rounds = config.get("refine", {}).get(
            "max_refinement_rounds",
            1,
        )
        if self._max_refinement_rounds_override is not None:
            self.max_refinement_rounds = self._max_refinement_rounds_override
        if self.max_refinement_rounds <= 0:
            raise ValueError("max_refinement_rounds must be positive")

    def _create_workspace(self) -> Path:
        if self.workspace_root is None:
            raise RuntimeError("HyperAgents workspace_root is not configured")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = self.workspace_root / f"run_{uuid4().hex}"
        workspace.mkdir(parents=False, exist_ok=False)
        return workspace

    def _configured_model(
        self,
        config: Mapping[str, Any],
        role: str,
    ) -> dict[str, Any]:
        model = config[role]
        backend_name = model.get("backend", "openai")
        backend = config["backend"][backend_name]
        return self._normalize_model(
            {
                "name": model["model"],
                "backend": backend_name,
                "api_mode": model.get("api_mode", backend.get("api_mode", "responses")),
                "api_key_env": backend.get("api_key_env", "OPENAI_API_KEY"),
                "base_url_env": backend.get("base_url_env", "OPENAI_BASE_URL"),
                "reasoning_effort": model.get("reasoning_effort"),
                "max_output_tokens": model.get("max_output_tokens", 16384),
            }
        )

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        resolved = path if path.is_absolute() else self.config_path.parent / path
        return resolved.resolve()

    @staticmethod
    def _normalize_model(config: Mapping[str, Any]) -> dict[str, Any]:
        if "name" not in config:
            raise ValueError("HyperAgents model config requires a name")
        normalized = dict(config)
        effort = normalized.get("reasoning_effort")
        if effort in (None, "", "default"):
            normalized.pop("reasoning_effort", None)
        normalized.setdefault("backend", "openai")
        normalized.setdefault("api_mode", "responses")
        normalized.setdefault("api_key_env", "OPENAI_API_KEY")
        normalized.setdefault("base_url_env", "OPENAI_BASE_URL")
        route = (normalized["backend"], normalized["api_mode"])
        supported_routes = {
            ("openai", "responses"),
            ("openai", "chat_completions"),
        }
        if route not in supported_routes:
            raise ValueError(f"Unsupported model route: {route}")
        return normalized

    @staticmethod
    def _collect_llm_environment(extra_names: set[str]) -> dict[str, str]:
        requested_names = _LLM_ENVIRONMENT_NAMES | extra_names
        unsupported = requested_names - _LLM_ENVIRONMENT_NAMES
        if unsupported:
            raise ValueError(
                f"Unsupported model environment names: {sorted(unsupported)}"
            )
        return {
            name: os.environ[name]
            for name in requested_names
            if os.environ.get(name)
        }

    @classmethod
    def _feedback_to_result(cls, feedback_results: Sequence[FeedbackResult]) -> dict:
        cases = [cls._feedback_case(result) for result in feedback_results]
        completed = all(
            result.evaluation.execution_status == ExecutionStatus.COMPLETED
            for result in feedback_results
        )
        success_count = sum(result.evaluation.success for result in feedback_results)
        task_count = len(feedback_results)
        score_mean = sum(
            result.evaluation.score for result in feedback_results
        ) / task_count
        return {
            "schema_version": 1,
            "valid": completed,
            "selection_score": score_mean if completed else None,
            "metrics": {
                "task_count": task_count,
                "success_count": success_count,
                "success_rate": success_count / task_count,
                "score_mean": score_mean,
            },
            "feedback": "Inspect the complete per-case task evidence.",
            "cases": cases,
            "artifacts": {},
            "cost": cls._aggregate_usage(feedback_results),
            "error": None if completed else "One or more benchmark tasks did not complete.",
        }

    @staticmethod
    def _feedback_case(result: FeedbackResult) -> dict[str, Any]:
        evaluation = result.evaluation
        return {
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

    @staticmethod
    def _aggregate_usage(feedback_results: Sequence[FeedbackResult]) -> dict[str, int]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }
        for result in feedback_results:
            for step in result.trajectory:
                usage = step.agent_call.usage if step.agent_call else None
                if usage is None:
                    continue
                for name in totals:
                    totals[name] += getattr(usage, name) or 0
        return totals


def _snapshot_to_dict(snapshot: CandidateSnapshot) -> dict[str, Any]:
    return {
        "candidate_id": str(snapshot.candidate_id),
        "parent_id": str(snapshot.parent_id) if snapshot.parent_id is not None else None,
        "run_dir": str(snapshot.run_dir),
        "workspace": str(snapshot.workspace),
        "root_commit": snapshot.root_commit,
        "lineage_patches": [str(path) for path in snapshot.lineage_patches],
    }


def _selection_evaluation(
    value: Any, label: str
) -> tuple[str, float | None, bool]:
    status = str(_field(value, "status") or "").strip().lower()
    if not status:
        raise ValueError(f"{label} evaluation status must not be empty")
    raw_score = _field(value, "relative_score")
    if raw_score is None:
        score = None
    elif (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(float(raw_score))
    ):
        raise ValueError(f"{label} evaluation score must be finite or null")
    else:
        score = float(raw_score)
    success = _field(value, "success")
    if not isinstance(success, bool):
        raise TypeError(f"{label} evaluation success must be bool")
    return status, score, success


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "HyperAgentsAdaptor",
    "HyperAgentsCandidateBatch",
    "HyperAgentsCandidateSpec",
    "HyperAgentsSelectionOutput",
    "HyperAgentsSelectionReport",
]
