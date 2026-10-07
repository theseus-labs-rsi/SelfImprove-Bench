"""RSIMethod adaptor that connects benchmark feedback to ACE refinement."""

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

from benchmark.backend import BackendDispatcher, OpenAIBackend
from benchmark.visual_evidence import frames, refinement_image_limit, sample, visual_note

if __package__:
    from .ACEAgent import ACEAgent
    from .ace.ace.core.curator import Curator
    from .ace.ace.core.reflector import Reflector
    from .ace.playbook_utils import (
        extract_playbook_bullets,
        get_next_global_id,
        get_playbook_stats,
        update_bullet_counts,
    )
else:  # Support direct imports from this adaptor directory.
    from ACEAgent import ACEAgent
    from ace.ace.core.curator import Curator
    from ace.ace.core.reflector import Reflector
    from ace.playbook_utils import (
        extract_playbook_bullets,
        get_next_global_id,
        get_playbook_stats,
        update_bullet_counts,
    )


EMPTY_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

_GROUND_TRUTH_FIELDS = ("ground_truth", "target", "reference_answer")
CONFIG_PATH = Path(os.environ.get("RSI_ACE_CONFIG_PATH", Path(__file__).with_name("config.toml")))


@dataclass(frozen=True, slots=True)
class ACERefinementItem:
    """Trace produced while adapting one benchmark feedback item."""

    task_id: str
    reflection: str | None
    bullet_tags: tuple[dict[str, str], ...] = ()
    call_info: Mapping[str, Any] | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ACERefinementReport:
    """Observable metadata for the latest generated candidate."""

    version_before: int
    version_after: int
    feedback_count: int
    items: tuple[ACERefinementItem, ...]
    curator_operations: tuple[Mapping[str, Any], ...]
    curator_calls: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ACECandidateSpec:
    """One benchmarkable playbook candidate derived from committed state."""

    candidate_id: str
    parent_version: int
    version: int
    artifact_dir: Path
    agent: ACEAgent
    playbook: str
    next_global_id: int
    report: ACERefinementReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_version": self.parent_version,
            "version": self.version,
            "artifact_dir": str(self.artifact_dir),
        }


@dataclass(frozen=True, slots=True)
class ACECandidateBatch:
    """Single-candidate batch returned through the common refinement shape."""

    refinement_id: str
    stage: str
    work_dir: Path
    candidates: tuple[ACECandidateSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "refinement_id": self.refinement_id,
            "stage": self.stage,
            "work_dir": str(self.work_dir),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class ACESelectionReport:
    """Record the benchmark comparison and resulting ACE state decision."""

    refinement_id: str
    selected_candidate_id: str | None
    parent_version: int
    committed_version: int
    parent_score: float | None
    candidate_score: float | None
    candidate_status: str
    candidate_success: bool
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ACESelectionOutput:
    """Public result returned after accepting or rejecting the candidate."""

    status: str
    committed_candidate_id: str | None
    report: ACESelectionReport


@dataclass(frozen=True, slots=True)
class _ACEFeedbackRecord:
    task_id: str
    question: str
    context: str
    reasoning_trace: str
    predicted_answer: str
    environment_feedback: str
    bullet_ids: tuple[str, ...]
    ground_truth: str | None
    visual_frames: tuple[dict, ...] = ()


class ACEAdaptor:
    """Expose ACE through the common RSI method lifecycle."""

    def __init__(self) -> None:
        self.config_path = CONFIG_PATH
        self.max_refinement_images: int | None = 16
        self.dispatcher: BackendDispatcher | None = None
        self.workspace_root: Path | None = None
        self.workspace_dir: Path | None = None
        self.initial_playbook: str | None = None
        self.initial_playbook_path: Path | None = None
        self.log_dir: Path | None = None
        self._playbook = ""
        self._next_global_id = 1
        self._version = 0
        self._refinement_round = 0
        self._processed_feedback_count = 0
        self._initialized = False
        self._reflector: Reflector | None = None
        self._curator: Curator | None = None
        self.last_refinement_report: ACERefinementReport | None = None
        self.last_selection_report: ACESelectionReport | None = None
        self._pending_batch: ACECandidateBatch | None = None
        self._config_snapshot = b""

    def _validate_config(self) -> None:
        """Reject configurations whose meaning would otherwise be ambiguous."""
        if self.initial_playbook is not None and self.initial_playbook_path is not None:
            raise ValueError("Provide initial_playbook or initial_playbook_path, not both")
        if self.refine_max_output_tokens <= 0:
            raise ValueError("refine_max_output_tokens must be positive")
        if self.playbook_token_budget <= 0:
            raise ValueError("playbook_token_budget must be positive")
        if self.curator_batch_size <= 0:
            raise ValueError("curator_batch_size must be positive")

    def _load_config(self) -> dict[str, Any]:
        self._config_snapshot = self.config_path.read_bytes()
        return tomllib.loads(self._config_snapshot.decode("utf-8"))

    def _configure(self, config: Mapping[str, Any]) -> None:
        self.workspace_root = self._required_path(
            config["state"].get("workspace_root"),
            "state.workspace_root",
        )
        backend = config["backend"]["openai"]
        self.dispatcher = BackendDispatcher(
            {
                "openai": OpenAIBackend(
                    api_key_env=backend.get("api_key_env", "OPENAI_API_KEY"),
                    base_url_env=backend.get("base_url_env", "OPENAI_BASE_URL"),
                    api_mode=backend.get("api_mode", "responses"),
                    max_output_tokens=backend.get("max_output_tokens", 512),
                )
            }
        )
        self._configure_roles(config, backend)
        self._configure_refinement(config["refine"])

    def _configure_roles(
        self,
        config: Mapping[str, Any],
        backend: Mapping[str, Any],
    ) -> None:
        agent = config["agent"]
        reflector = config["reflector"]
        curator = config["curator"]
        self.agent_backend_name = agent.get("backend", "openai")
        self.agent_model = agent["model"]
        self.agent_api_mode = agent.get("api_mode", backend.get("api_mode"))
        self.agent_max_output_tokens = agent.get("max_output_tokens", 4096)
        self.agent_reasoning_effort = agent.get("reasoning_effort")
        self.agent_output_format = agent.get("output_format")
        self.reflector_backend_name = reflector.get("backend", "openai")
        self.reflector_model = reflector["model"]
        self.reflector_reasoning_effort = reflector.get("reasoning_effort")
        self.curator_backend_name = curator.get("backend", "openai")
        self.curator_model = curator["model"]
        self.curator_reasoning_effort = curator.get("reasoning_effort")
        self.initial_playbook_path = self._optional_path(
            config["state"].get("initial_playbook_path")
        )

    def _configure_refinement(self, config: Mapping[str, Any]) -> None:
        self.refine_max_output_tokens = config.get("max_output_tokens", 4096)
        self.playbook_token_budget = config.get("playbook_token_budget", 80000)
        self.curator_batch_size = config.get("curator_batch_size", 1)
        self.use_json_mode = config.get("use_json_mode", True)
        self.use_ground_truth = config.get("use_ground_truth", False)
        # Shared with AHE/HyperAgents; the Curator stays text-only as upstream.
        self.max_refinement_images = refinement_image_limit()

    def _optional_path(self, value: Any) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(str(value)).expanduser()
        resolved = path if path.is_absolute() else self.config_path.parent / path
        return resolved.resolve()

    def _required_path(self, value: Any, name: str) -> Path:
        path = self._optional_path(value)
        if path is None:
            raise ValueError(f"{name} must be a non-empty path")
        return path

    @property
    def playbook(self) -> str:
        """Return the currently committed ACE playbook."""
        self._require_initialized()
        return self._playbook

    @property
    def version(self) -> int:
        """Return the committed playbook version."""
        return self._version

    def init(self) -> None:
        """Load method configuration and construct the initial playbook."""
        self._initialized = False
        self._configure(self._load_config())
        self._validate_config()
        self._playbook = self._load_initial_playbook()
        self.workspace_dir = self._create_workspace()
        (self.workspace_dir / "method_config.toml").write_bytes(
            self._config_snapshot
        )
        (self.workspace_dir / "playbook.txt").write_text(
            self._playbook,
            encoding="utf-8",
        )
        self.log_dir = None
        self._next_global_id = get_next_global_id(self._playbook)
        self._reflector = Reflector(
            self.dispatcher,
            self.reflector_backend_name,
            self.reflector_model,
            self.refine_max_output_tokens,
        )
        self._curator = Curator(
            self.dispatcher,
            self.curator_backend_name,
            self.curator_model,
            self.refine_max_output_tokens,
        )
        self._version = 0
        self._refinement_round = 0
        self._processed_feedback_count = 0
        self.last_refinement_report = None
        self.last_selection_report = None
        self._pending_batch = None
        self._initialized = True

    def build(self) -> ACEAgent:
        """Build an immutable agent snapshot from the committed playbook."""
        self._require_initialized()
        return self._build_agent(self._playbook)

    def _build_agent(self, playbook: str) -> ACEAgent:
        """Build an agent snapshot for committed or candidate state."""
        return ACEAgent(
            dispatcher=self.dispatcher,
            backend_name=self.agent_backend_name,
            model=self.agent_model,
            playbook=playbook,
            api_mode=self.agent_api_mode,
            max_output_tokens=self.agent_max_output_tokens,
            reasoning_effort=self.agent_reasoning_effort,
            output_format=self.agent_output_format,
        )

    def load(self) -> ACEAgent:
        """Implement the common RSIMethod name for building an agent snapshot."""
        return self.build()

    def refine(
        self,
        feedback_results: Sequence[Any],
        *,
        output_dir: str | Path,
    ) -> ACECandidateBatch:
        """Generate and persist one candidate without committing it."""
        self._require_initialized()
        if self._pending_batch is not None:
            raise RuntimeError("The previous ACE refinement still requires selection")
        if isinstance(feedback_results, (str, bytes)) or not isinstance(
            feedback_results, Sequence
        ):
            raise TypeError("feedback_results must be a sequence")
        if not feedback_results:
            raise ValueError("feedback_results must not be empty")

        resolved_output = self._prepare_output(output_dir)
        refinement_id = f"refine_{self._refinement_round + 1}_{uuid4().hex[:8]}"
        candidate_id = f"{refinement_id}_candidate_000"
        candidate_dir = resolved_output / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        self.log_dir = candidate_dir
        records = tuple(self._normalize_feedback(item) for item in feedback_results)
        candidate_playbook = self._playbook
        refinement_items: list[ACERefinementItem] = []
        usable_reflections: list[tuple[_ACEFeedbackRecord, str]] = []

        for record in records:
            item, candidate_playbook = self._reflect_record(
                record,
                candidate_playbook,
            )
            refinement_items.append(item)
            if item.reflection is not None:
                usable_reflections.append((record, item.reflection))

        candidate_playbook, operations, curator_calls, next_global_id = (
            self._curate_reflections(
                usable_reflections,
                candidate_playbook,
                self._next_global_id,
                len(records),
            )
        )

        version_before = self._version
        version_after = version_before + int(candidate_playbook != self._playbook)
        report = ACERefinementReport(
            version_before=version_before,
            version_after=version_after,
            feedback_count=len(records),
            items=tuple(refinement_items),
            curator_operations=tuple(operations),
            curator_calls=tuple(curator_calls),
        )
        self._persist_refinement(candidate_dir, candidate_playbook, report)
        candidate = ACECandidateSpec(
            candidate_id=candidate_id,
            parent_version=version_before,
            version=version_after,
            artifact_dir=candidate_dir,
            agent=self._build_agent(candidate_playbook),
            playbook=candidate_playbook,
            next_global_id=next_global_id,
            report=report,
        )
        batch = ACECandidateBatch(
            refinement_id=refinement_id,
            stage="candidates_ready",
            work_dir=resolved_output,
            candidates=(candidate,),
        )
        (resolved_output / "candidate_batch.json").write_text(
            json.dumps(batch.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._refinement_round += 1
        self._processed_feedback_count += len(records)
        self.last_refinement_report = report
        self._pending_batch = batch
        return batch

    def select(
        self,
        batch: ACECandidateBatch,
        *,
        parent_evaluation: Any,
        candidate_evaluations: Mapping[str, Any],
        output_dir: str | Path,
    ) -> ACESelectionOutput:
        """Commit the selected candidate, or retain the committed parent."""
        self._require_initialized()
        pending_batch = self._pending_batch
        if pending_batch is None:
            raise RuntimeError("ACEAdaptor.select() has no pending candidate batch")
        if batch is not pending_batch:
            raise RuntimeError("ACEAdaptor.select() received a stale candidate batch")
        if not isinstance(candidate_evaluations, Mapping):
            raise TypeError("candidate_evaluations must be a mapping")

        candidate = batch.candidates[0]
        expected_ids = {candidate.candidate_id}
        provided_ids = {str(key) for key in candidate_evaluations}
        if provided_ids != expected_ids:
            raise ValueError(
                "candidate_evaluations must contain exactly the pending "
                f"candidate id: {candidate.candidate_id}"
            )
        if candidate.parent_version != self._version:
            raise RuntimeError("Committed ACE state changed after candidate generation")

        candidate_evaluation = candidate_evaluations[candidate.candidate_id]
        _, parent_score, _ = _selection_evaluation(parent_evaluation, "parent")
        candidate_status, candidate_score, candidate_success = _selection_evaluation(
            candidate_evaluation, "candidate"
        )
        if candidate_status != "ok":
            accepted, reason = False, "candidate_status_not_ok"
        elif parent_score is None or candidate_score is None:
            accepted, reason = False, "selection_score_missing"
        elif candidate_score < parent_score:
            accepted, reason = False, "selection_score_regressed"
        else:
            accepted, reason = True, "candidate_score_not_below_parent"
        selected_candidate_id = candidate.candidate_id if accepted else None
        report = ACESelectionReport(
            refinement_id=batch.refinement_id,
            selected_candidate_id=selected_candidate_id,
            parent_version=candidate.parent_version,
            committed_version=candidate.version if accepted else self._version,
            parent_score=parent_score,
            candidate_score=candidate_score,
            candidate_status=candidate_status,
            candidate_success=candidate_success,
            accepted=accepted,
            reason=reason,
        )
        resolved_output = Path(output_dir).resolve()
        self._persist_selection_report(resolved_output, report)
        if self.workspace_dir is None:
            raise RuntimeError("ACE workspace is not initialized")
        self._persist_selection_report(self.workspace_dir, report)
        if not accepted:
            self.last_selection_report = report
            self._pending_batch = None
            return ACESelectionOutput(
                status="unchanged",
                committed_candidate_id=None,
                report=report,
            )

        self._persist_playbook(self.workspace_dir, candidate.playbook)
        self._playbook = candidate.playbook
        self._version = candidate.version
        self._next_global_id = candidate.next_global_id
        self.last_selection_report = report
        self._pending_batch = None
        return ACESelectionOutput(
            status="committed",
            committed_candidate_id=candidate.candidate_id,
            report=report,
        )

    def _prepare_output(self, output_dir: str | Path) -> Path:
        """Prepare the method-owned output directory for one refinement."""
        resolved_output = Path(output_dir).resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)
        self.log_dir = resolved_output
        return resolved_output

    def _create_workspace(self) -> Path:
        if self.workspace_root is None:
            raise RuntimeError("ACE workspace_root is not configured")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = self.workspace_root / f"run_{uuid4().hex}"
        workspace.mkdir(parents=False, exist_ok=False)
        return workspace

    @staticmethod
    def _persist_refinement(
        output_dir: Path,
        playbook: str,
        report: ACERefinementReport,
    ) -> None:
        """Persist one completed refinement before committing in-memory state."""
        (output_dir / "playbook.txt").write_text(
            playbook,
            encoding="utf-8",
        )
        (output_dir / "refinement_report.json").write_text(
            json.dumps(
                asdict(report),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _persist_playbook(
        output_dir: Path,
        playbook: str,
    ) -> None:
        """Persist selected playbook state before changing its pointer."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "playbook.txt").write_text(playbook, encoding="utf-8")

    @staticmethod
    def _persist_selection_report(
        output_dir: Path,
        report: ACESelectionReport,
    ) -> None:
        """Persist the decision separately from the selected state."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "selection.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _reflect_record(
        self,
        record: _ACEFeedbackRecord,
        playbook: str,
    ) -> tuple[ACERefinementItem, str]:
        """Reflect on one task without mutating committed adaptor state."""
        if not record.predicted_answer:
            item = ACERefinementItem(
                task_id=record.task_id,
                reflection=None,
                skipped_reason="trajectory contains no model action",
            )
            return item, playbook

        assert self._reflector is not None
        ground_truth = record.ground_truth if self.use_ground_truth else None
        bullets = extract_playbook_bullets(playbook, list(record.bullet_ids))
        reflection, bullet_tags, call_info = self._reflector.reflect(
            question=record.question,
            reasoning_trace=record.reasoning_trace,
            predicted_answer=record.predicted_answer,
            ground_truth=ground_truth,
            environment_feedback=record.environment_feedback,
            bullets_used=bullets,
            use_ground_truth=ground_truth is not None,
            use_json_mode=self.use_json_mode,
            call_id=f"refine_{self._refinement_round + 1}_{record.task_id}",
            log_dir=self._log_dir_text(),
            reasoning_effort=self.reflector_reasoning_effort,
            **({"image_paths": [f["path"] for f in record.visual_frames]}
               if record.visual_frames else {}),
        )
        updated_playbook = update_bullet_counts(playbook, bullet_tags)
        item = ACERefinementItem(
            task_id=record.task_id,
            reflection=reflection,
            bullet_tags=tuple(dict(tag) for tag in bullet_tags),
            call_info=dict(call_info),
        )
        return item, updated_playbook

    def _curate_reflections(
        self,
        reflections: Sequence[tuple[_ACEFeedbackRecord, str]],
        playbook: str,
        next_global_id: int,
        total_feedback: int,
    ) -> tuple[str, list[Mapping[str, Any]], list[Mapping[str, Any]], int]:
        """Apply curator updates in bounded chunks and return uncommitted state."""
        all_operations: list[Mapping[str, Any]] = []
        curator_calls: list[Mapping[str, Any]] = []
        candidate = playbook
        assert self._curator is not None

        for chunk_index, start in enumerate(
            range(0, len(reflections), self.curator_batch_size),
            start=1,
        ):
            chunk = reflections[start : start + self.curator_batch_size]
            reflection_text, context_text = self._format_curator_chunk(chunk)
            candidate, next_global_id, operations, call_info = self._curator.curate(
                current_playbook=candidate,
                recent_reflection=reflection_text,
                question_context=context_text,
                current_step=start + len(chunk),
                total_samples=total_feedback,
                token_budget=self.playbook_token_budget,
                playbook_stats=get_playbook_stats(candidate),
                use_ground_truth=(
                    self.use_ground_truth
                    and all(record.ground_truth is not None for record, _ in chunk)
                ),
                use_json_mode=self.use_json_mode,
                call_id=f"refine_{self._refinement_round + 1}_curate_{chunk_index}",
                log_dir=self._log_dir_text(),
                next_global_id=next_global_id,
                reasoning_effort=self.curator_reasoning_effort,
            )
            all_operations.extend(dict(operation) for operation in operations)
            curator_calls.append(dict(call_info))

        return candidate, all_operations, curator_calls, next_global_id

    def _normalize_feedback(self, feedback: Any) -> _ACEFeedbackRecord:
        """Convert the common FeedbackResult shape into ACE prompt fields."""
        task_id = _text(_field(feedback, "task_id")).strip()
        if not task_id:
            raise ValueError("Each feedback result must contain task_id")

        task = _field(feedback, "task", {})
        trajectory = _field(feedback, "trajectory", ()) or ()
        evaluation = _field(feedback, "evaluation")
        if evaluation is None:
            raise ValueError(f"Feedback {task_id!r} must contain evaluation")
        if isinstance(trajectory, (str, bytes)) or not isinstance(trajectory, Sequence):
            raise TypeError(f"Feedback {task_id!r} trajectory must be a sequence")

        question = self._extract_question(task, trajectory)
        trace, answer, bullet_ids = self._format_trajectory(trajectory)
        selected_images = sample(frames(feedback), self.max_refinement_images)
        return _ACEFeedbackRecord(
            task_id=task_id,
            question=question,
            context=self._extract_context(task),
            reasoning_trace=trace + visual_note(selected_images),
            predicted_answer=answer,
            environment_feedback=self._format_environment_feedback(
                trajectory,
                evaluation,
            ),
            bullet_ids=bullet_ids,
            ground_truth=self._extract_ground_truth(task),
            visual_frames=tuple(selected_images),
        )

    def _format_trajectory(
        self,
        trajectory: Sequence[Any],
    ) -> tuple[str, str, tuple[str, ...]]:
        """Preserve ordered task history and recover ACE response metadata."""
        trace_blocks: list[str] = []
        final_answer = ""
        bullet_ids: list[str] = []

        for index, step in enumerate(trajectory):
            action_text = _text(_field(step, "action", ""))
            agent_call = _field(step, "agent_call", None)
            response_text = _text(
                _field(agent_call, "response_text", action_text)
            )
            _, parsed, parsed_reasoning = _decode_action(response_text)
            metadata = _field(agent_call, "metadata", {}) or {}
            ace_metadata = _field(metadata, "ace", {}) or {}
            if action_text.strip():
                final_answer = action_text.strip()
            bullet_ids.extend(
                _string_list(
                    _field(
                        ace_metadata,
                        "bullet_ids",
                        parsed.get("bullet_ids", ()),
                    )
                )
            )
            backend_reasoning = _text(_field(agent_call, "reasoning", ""))
            ace_reasoning = _text(_field(ace_metadata, "reasoning", ""))
            reasoning = _text(
                _field(
                    step,
                    "reasoning",
                    ace_reasoning
                    or backend_reasoning
                    or parsed_reasoning
                    or parsed.get("reasoning", ""),
                )
            )
            raw_response_text = _text(
                _field(ace_metadata, "raw_response_text", response_text)
            )
            trace_blocks.append(
                self._format_step(
                    index,
                    step,
                    action_text,
                    reasoning,
                    raw_response_text,
                )
            )

        return "\n\n".join(trace_blocks) or "(empty trajectory)", final_answer, tuple(dict.fromkeys(bullet_ids))

    @staticmethod
    def _format_step(
        index: int,
        step: Any,
        action: str,
        reasoning: str,
        response_text: str,
    ) -> str:
        step_index = _field(step, "step_index", index)
        parts = [
            f"Step {step_index}",
            f"Prompt: {_text(_field(step, 'prompt', ''))}",
            f"Action: {action}",
            f"Environment response: {_text(_field(step, 'environment_response', ''))}",
            f"Success: {_text(_field(step, 'success', None))}",
            f"Score: {_text(_field(step, 'score', None))}",
        ]
        if reasoning:
            parts.append(f"Model reasoning: {reasoning}")
        if response_text and response_text != action:
            parts.append(f"Raw model response: {response_text}")
        feedback = _text(_field(step, "feedback", ""))
        if feedback:
            parts.append(f"Step feedback: {feedback}")
        return "\n".join(parts)

    @staticmethod
    def _format_environment_feedback(trajectory: Sequence[Any], evaluation: Any) -> str:
        parts = [
            "Execution status: "
            f"{_enum_text(_field(evaluation, 'execution_status', 'completed'))}",
            f"Task success: {_text(_field(evaluation, 'success', False))}",
            f"Task score: {_text(_field(evaluation, 'score', 0.0))}",
        ]
        overall_feedback = _text(_field(evaluation, "feedback", ""))
        if overall_feedback:
            parts.append(f"Task feedback: {overall_feedback}")
        metrics = _field(evaluation, "metrics", None)
        if metrics:
            parts.append(f"Task metrics: {_text(metrics)}")
        for index, step in enumerate(trajectory):
            step_feedback = _text(_field(step, "feedback", ""))
            if step_feedback:
                parts.append(f"Step {index} feedback: {step_feedback}")
        return "\n".join(parts)

    @staticmethod
    def _extract_question(task: Any, trajectory: Sequence[Any]) -> str:
        for name in ("question", "prompt", "instruction", "description"):
            value = _text(_field(task, name, "")).strip()
            if value:
                return value
        if isinstance(task, str) and task.strip():
            return task.strip()
        if trajectory:
            prompt = _text(_field(trajectory[0], "prompt", "")).strip()
            if prompt:
                return prompt
        return "(task question unavailable)"

    @staticmethod
    def _extract_context(task: Any) -> str:
        for name in ("context", "public_context"):
            value = _text(_field(task, name, "")).strip()
            if value:
                return value
        return "(empty context)"

    def _extract_ground_truth(self, task: Any) -> str | None:
        if not self.use_ground_truth:
            return None
        for name in _GROUND_TRUTH_FIELDS:
            value = _text(_field(task, name, "")).strip()
            if value:
                return value
        return None

    @staticmethod
    def _format_curator_chunk(
        chunk: Sequence[tuple[_ACEFeedbackRecord, str]],
    ) -> tuple[str, str]:
        reflections: list[str] = []
        contexts: list[str] = []
        for record, reflection in chunk:
            reflections.append(f"[Task {record.task_id}]\n{reflection}")
            contexts.append(
                f"[Task {record.task_id}]\nQuestion: {record.question}\n"
                f"Context: {record.context}\nFeedback: {record.environment_feedback}"
            )
        return "\n\n".join(reflections), "\n\n".join(contexts)

    def _load_initial_playbook(self) -> str:
        if self.initial_playbook_path is not None:
            path = Path(self.initial_playbook_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Initial playbook not found: {path}")
            content = path.read_text(encoding="utf-8")
        else:
            content = self.initial_playbook or ""
        return content.strip() or EMPTY_PLAYBOOK

    def _log_dir_text(self) -> str | None:
        return str(self.log_dir) if self.log_dir is not None else None

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("ACEAdaptor.init() must be called first")


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


def _enum_text(value: Any) -> str:
    enum_value = getattr(value, "value", value)
    return _text(enum_value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _decode_action(action: Any) -> tuple[str, Mapping[str, Any], str]:
    text = _text(_field(action, "text", action))
    backend_reasoning = _text(_field(action, "reasoning", ""))
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    return text, parsed if isinstance(parsed, Mapping) else {}, backend_reasoning


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if str(item)]


__all__ = [
    "ACEAdaptor",
    "ACECandidateBatch",
    "ACECandidateSpec",
    "ACERefinementItem",
    "ACERefinementReport",
    "ACESelectionOutput",
    "ACESelectionReport",
]
