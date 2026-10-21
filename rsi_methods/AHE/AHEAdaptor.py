"""Expose AHE initialization, agent construction, generation and selection.

``init`` owns a new lineage directory, while ``build`` binds one NexAU Agent
to its currently selected harness. ``refine`` proposes variants; the benchmark
evaluates them, and ``select`` commits the next harness state.
"""

from __future__ import annotations

import os
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from benchmark.interfaces import MethodContext
from rsi_methods.checkpoint import MethodCheckpointMixin

from .agent import AHEAgent
from .build import build as build_agent
from .init import AHEState, apply_agent_limits, initialize
from .refine import CandidateBatch, refine as generate_candidates
from .select import SelectionOutcome, select as select_candidate
from .runtime_policy import AHERuntimePolicy

CONFIG_PATH = Path(__file__).with_name("config.toml")
_SECRET_ASSIGNMENT = re.compile(
    r"^(?P<prefix>\s*(?:api_key|admin_api_key)\s*=\s*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'[^']*')"
    r"(?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)


def _resume_workspace_relative(selected: str, source: Path, copied: Path) -> Path:
    """Resolve a selected workspace after its benchmark directory was moved."""
    selected_path = Path(selected).resolve()
    try:
        relative = selected_path.relative_to(source)
    except ValueError:
        parts = selected_path.parts
        candidates = (Path(*parts[index + 1:]) for index in range(len(parts) - 1, -1, -1)
                      if parts[index].lower() == "state")
        relative = next((path for path in candidates if (copied / path).is_dir()), None)
        if relative is None:
            raise ValueError(f"Selected AHE workspace cannot be relocated: {selected}") from None
    workspace = (copied / relative).resolve()
    workspace.relative_to(copied)
    if not workspace.is_dir():
        raise FileNotFoundError(f"Selected AHE workspace is missing: {workspace}")
    return relative


class AHEAdaptor(MethodCheckpointMixin):
    """Hold one AHE lineage across benchmark tasks and generations."""

    checkpoint_method = "ahe"
    runtime_policy = AHERuntimePolicy()

    def __init__(
        self,
        run_dir: str | Path | MethodContext | None = None,
        *,
        config_path: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        refinement_every: int | None = None,
        timeout_seconds: float = 300,
        candidate_count: int = 1,
    ) -> None:
        context = run_dir if isinstance(run_dir, MethodContext) else None
        self.bind_context(context)
        if context is not None:
            run_dir = context.artifact_dir / "state"
            timeout_seconds = context.timeout
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        project_root = Path(__file__).resolve().parents[3]
        default_dir = project_root / "outputs" / "ahe" / f"run_{uuid4().hex}"
        self.run_dir = Path(run_dir or default_dir).expanduser().resolve()
        self.config_path = Path(config_path).resolve() if config_path else CONFIG_PATH
        self._model_override = model
        self._reasoning_effort_override = reasoning_effort
        self._refinement_every_override = refinement_every
        self.model: str | None = model
        self.timeout_seconds = timeout_seconds
        self.candidate_count = candidate_count
        self._fixed_llm_environment: dict[str, str] = {}
        self.state: AHEState | None = None
        self._agent: AHEAgent | None = None
        self._refinement_round = 0
        self._observed_training_tasks = 0
        self._pending_batch: CandidateBatch | None = None
        self.last_refinement_report: dict[str, Any] | None = None
        self.refinement_every = 5
        self.reasoning_effort = "high"
        self.code_agent_limits = {
            "max_context_tokens": 100_000,
            "max_iterations": 16,
            "max_tokens": 4_096,
        }
        self.evolve_agent_limits = {
            "max_context_tokens": 120_000,
            "max_iterations": 64,
            "max_tokens": 8_192,
        }

    def checkpoint_policy(self) -> dict[str, Any]:
        """Persist method changes that would make old checkpoints incomparable."""
        return {
            "version": "upstream-active-lineage-v3",
            "filesystem_isolation": "bubblewrap-allowlist-v1",
            "refinement_usage": "debugger-and-evolver-tracers-v1",
            "game_history_turns": 8,
            "refinement_every": self.refinement_every,
            "candidate_count": self.candidate_count,
            "code_agent_limits": dict(self.code_agent_limits),
            "evolve_agent_limits": dict(self.evolve_agent_limits),
            "task_isolation": True,
            "screenshots": True,
        }

    def init(self) -> None:
        """Create a fresh state by copying the upstream AHE agent surfaces.

        This delegates to ``init.initialize``; that function rejects an
        existing run directory, so calling ``init`` twice cannot silently
        replace the parent harness or an earlier experiment.
        """

        if self.state is not None:
            raise RuntimeError("AHEAdaptor.init() has already been called")
        self._configure(self._load_config())
        self.state = initialize(self.run_dir)
        apply_agent_limits(
            self.state.current_workspace / "code_agent.yaml", self.code_agent_limits,
            reasoning_effort=self.reasoning_effort,
        )
        apply_agent_limits(
            self.state.evolve_agent_dir / "evolve_agent.yaml", self.evolve_agent_limits,
            reasoning_effort=self.reasoning_effort,
        )
        (self.run_dir / "method_config.toml").write_text(
            self._redacted_config_text(), encoding="utf-8"
        )
        (self.run_dir / "ahe_policy.json").write_text(
            json.dumps(self.checkpoint_policy(), indent=2), encoding="utf-8",
        )

    def resume(
        self, previous_batch: str | Path, *, source_batch: str | Path,
        before_task_index: int, target_dir: str | Path,
    ) -> None:
        """Continue from the copied AHE state and its last selected harness."""
        previous = Path(previous_batch).resolve()
        source_state = Path(source_batch).resolve() / "state"
        copied_state = previous / "state"
        target = Path(target_dir).resolve()
        if self.state is not None or not copied_state.is_dir() or target.exists():
            raise ValueError("AHE resume requires a copied state and a fresh target")
        self._configure(self._load_config())
        policy_path = copied_state / "ahe_policy.json"
        if (
            not policy_path.is_file()
            or json.loads(policy_path.read_text(encoding="utf-8")) != self.checkpoint_policy()
        ):
            raise RuntimeError("AHE resume policy differs; use historical code for old runs")
        reports = sorted(previous.glob("task_*/selection_stage_*/ahe_selection.json"))
        selected = (
            json.loads(reports[-1].read_text(encoding="utf-8"))["selected_workspace"]
            if reports else str(source_state / "workspace")
        )
        relative = _resume_workspace_relative(selected, source_state, copied_state)
        copied_state.rename(target)
        self.run_dir = target
        self.state = AHEState(target, target / relative, target / "evolve_agent")
        iterations = [int(path.name[10:]) for path in (target / "runs").glob("iteration_*")]
        self._refinement_round = max(iterations, default=0)
        self._observed_training_tasks = before_task_index

    def build(self) -> AHEAgent:
        """Return the Agent bound to the currently selected harness.

        Repeated calls within one benchmark task reuse this object and its
        NexAU process. ``load`` starts a fresh object at every task boundary.
        """

        if self.state is None:
            raise RuntimeError("AHEAdaptor.init() must be called before build()")
        if self._agent is None:
            self._agent = build_agent(
                self.state, model=self.model, timeout_seconds=self.timeout_seconds,
                llm_environment=self._llm_environment(),
                reasoning_effort=self.reasoning_effort, agent_limits=self.code_agent_limits,
            )
        return self._agent

    def load(self) -> AHEAgent:
        """Start a fresh runtime so history cannot leak across benchmark tasks."""

        self._release_agent()
        return self.build()

    def refine(
        self,
        feedback_results: Sequence[Any],
        *,
        output_dir: str | Path,
    ) -> CandidateBatch | None:
        """Generate candidates from this generation's score and traces.

        An unresolved batch blocks another refinement: selecting a winner or
        retaining the parent must happen before the lineage can advance.
        Candidate scores are intentionally absent until benchmark evaluation.
        """

        if self.state is None:
            raise RuntimeError("AHEAdaptor.init() must be called before refine()")
        if self._pending_batch is not None:
            raise RuntimeError("The previous AHE candidate batch still requires selection")
        self._observed_training_tasks += 1
        self.last_refinement_report = None
        if self._observed_training_tasks % self.refinement_every:
            self._release_agent()
            self.last_refinement_report = {
                "status": "skipped",
                "reason": "refinement_interval",
                "observed_training_tasks": self._observed_training_tasks,
                "refinement_every": self.refinement_every,
                "token_usage_report": {"method": "no_model_calls", "complete": True, "total_tokens": 0,
                                       "prompt_num_tokens": 0, "response_num_tokens": 0, "reasoning_num_tokens": 0},
            }
            return None
        next_round = self._refinement_round + 1
        batch = generate_candidates(
            self.state, self.build(), feedback_results,
            output_dir=output_dir, iteration=next_round, model=self.model,
            candidate_count=self.candidate_count,
            llm_environment=self._llm_environment(),
            agent_limits=self.code_agent_limits,
            reasoning_effort=self.reasoning_effort,
        )
        self._refinement_round = next_round
        self._pending_batch = batch
        self.last_refinement_report = json.loads((Path(output_dir) / "ahe_refine.json").read_text(encoding="utf-8"))
        return batch

    def select(
        self,
        batch: CandidateBatch,
        *,
        parent_evaluation: Any,
        candidate_evaluations: Mapping[str, Any],
        output_dir: str | Path,
        selection_mode: str = "paired",
    ) -> SelectionOutcome:
        """Resolve exactly the batch returned by the last refine call.

        ``select.py`` owns the score comparison and state transition. Once
        it succeeds, every Agent from the evaluated generation is closed;
        the next ``build()`` starts a clean process on the chosen workspace.
        An invalid evaluation discards the batch so it cannot poison later tasks.
        """

        if self.state is None or self._pending_batch is None:
            raise RuntimeError("AHEAdaptor.select() has no pending candidate batch")
        if batch is not self._pending_batch:
            raise RuntimeError("AHEAdaptor.select() received a stale candidate batch")
        try:
            outcome = select_candidate(
                self.state, batch, parent_evaluation=parent_evaluation,
                candidate_evaluations=candidate_evaluations, output_dir=output_dir,
                selection_mode=selection_mode,
            )
        except BaseException:
            self._discard_pending()
            self._release_agent()
            raise
        self.close()
        return outcome

    def _release_agent(self) -> None:
        if self._agent is not None:
            self._agent.close()
            self._agent = None

    def _discard_pending(self) -> None:
        if self._pending_batch is not None:
            for candidate in self._pending_batch.candidates:
                candidate.agent.close()
            self._pending_batch = None

    def close(self) -> None:
        """Release parent and candidate processes, including between generations."""

        self._release_agent()
        self._discard_pending()

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open("rb") as config_file:
            return tomllib.load(config_file)

    def _configure(self, config: Mapping[str, Any]) -> None:
        configured_model = config.get("agent", {}).get("model")
        if self._model_override is None and configured_model:
            self.model = str(configured_model)
        backend = config.get("backend", {}).get("openai", {})
        fixed_environment: dict[str, str] = {}
        for value_key, env_key, default_env in (
            ("api_key", "api_key_env", "OPENAI_API_KEY"),
            ("base_url", "base_url_env", "OPENAI_BASE_URL"),
        ):
            value = backend.get(value_key)
            if value in (None, ""):
                continue
            environment_name = backend.get(env_key, default_env)
            if not isinstance(environment_name, str) or not environment_name:
                raise ValueError(f"backend.openai.{env_key} must be non-empty")
            fixed_environment[environment_name] = str(value)
        self._fixed_llm_environment = fixed_environment
        effort = config.get("agent", {}).get("reasoning_effort", "high")
        if not isinstance(effort, str) or not effort:
            raise ValueError("agent.reasoning_effort must be a non-empty string")
        self.reasoning_effort = self._reasoning_effort_override or effort
        limits = config.get("limits", {})
        if not isinstance(limits, Mapping):
            raise ValueError("limits must be a table")

        def positive(name: str, default: int) -> int:
            value = limits.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"limits.{name} must be a positive integer")
            return value

        self.refinement_every = positive("refinement_every", self.refinement_every)
        if self._refinement_every_override is not None:
            if isinstance(self._refinement_every_override, bool) or not isinstance(self._refinement_every_override, int) or self._refinement_every_override < 1:
                raise ValueError("refinement_every must be a positive integer")
            self.refinement_every = self._refinement_every_override
        self.code_agent_limits = {
            "max_context_tokens": positive("agent_max_context_tokens", 100_000),
            "max_iterations": positive("agent_max_iterations", 16),
            "max_tokens": positive("agent_max_output_tokens", 4_096),
        }
        self.evolve_agent_limits = {
            "max_context_tokens": positive("evolve_max_context_tokens", 120_000),
            "max_iterations": positive("evolve_max_iterations", 64),
            "max_tokens": positive("evolve_max_output_tokens", 8_192),
        }

    def _llm_environment(self) -> dict[str, str]:
        config = self._load_config()
        backend = config.get("backend", {}).get("openai", {})
        aliases = {
            "LLM_API_KEY": backend.get("api_key_env", "OPENAI_API_KEY"),
            "LLM_BASE_URL": backend.get("base_url_env", "OPENAI_BASE_URL"),
        }
        names = set(aliases) | set(aliases.values())
        environment = {
            name: os.environ[name]
            for name in names
            if isinstance(name, str) and name and os.environ.get(name)
        }
        environment.update(self._fixed_llm_environment)
        for target, source in aliases.items():
            if target not in environment and source in environment:
                environment[target] = environment[source]
        return environment

    def _redacted_config_text(self) -> str:
        """Return a reproducible config snapshot without plaintext API keys."""
        source = self.config_path.read_text(encoding="utf-8")
        return _SECRET_ASSIGNMENT.sub(
            lambda match: (
                f'{match.group("prefix")}"<redacted>"'
                f'{match.group("suffix")}'
            ),
            source,
        )


# Temporary end-to-end refine check: remove this block after a successful run.
if __name__ == "__main__":
    from benchmark.config import config_from_mapping
    from benchmark.runner import AgentController, _run_one
    from benchmark.tasks import build_manifest

    output_dir = Path(__file__).resolve().parents[3] / "outputs" / "ahe" / f"refine_smoke_{uuid4().hex}"
    config = config_from_mapping({
        "seed": 0,
        "schedule": [{"game": "farm", "difficulty": 1, "count": 1}],
        "holdout": [{"game": "farm", "difficulty": 1, "count": 1}],
        "holdout_tasks_per_difficulty": 1,
        "agent": {"provider": "ahe", "max_attempts": 1, "timeout": 300},
        "output_dir": str(output_dir),
    })
    task = build_manifest(config).training[0]
    controller = AgentController(config, output_dir, training_count=1)
    try:
        controller.session.adaptor.refinement_every = 1
        state = controller.session.adaptor.state
        assert state is not None
        # Keep this explicit smoke run smaller than the production safety limits.
        apply_agent_limits(
            state.current_workspace / "code_agent.yaml",
            {**controller.session.adaptor.code_agent_limits, "max_iterations": 8},
        )
        apply_agent_limits(
            state.evolve_agent_dir / "evolve_agent.yaml",
            {**controller.session.adaptor.evolve_agent_limits, "max_iterations": 16},
        )
        print(f"AHE refine smoke directory: {output_dir}", flush=True)
        record = _run_one(task, controller=controller, ordinal=1, output_dir=output_dir)
        selection = record.get("selection") or {}
        print(f"AHE selection: {selection.get('status')}; score: {record.get('relative_score')}", flush=True)
        if selection.get("status") not in {"committed", "unchanged"}:
            raise RuntimeError(f"AHE refine smoke failed; inspect {output_dir}")
    finally:
        controller.close()
