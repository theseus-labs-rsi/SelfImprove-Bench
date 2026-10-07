"""Expose AHE initialization, agent construction, generation and selection.

``init`` owns a new lineage directory, while ``build`` binds one NexAU Agent
to its currently selected harness. ``refine`` proposes variants; the benchmark
evaluates them, and ``select`` commits the next harness state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .agent import AHEAgent
from .build import build as build_agent
from .init import AHEState, apply_agent_limits, initialize
from .refine import CandidateBatch, refine as generate_candidates
from .select import SelectionOutcome, select as select_candidate


class AHEAdaptor:
    """Hold one AHE lineage across benchmark tasks and generations."""

    def __init__(
        self,
        run_dir: str | Path | None = None,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 300,
        candidate_count: int = 1,
        refinement_every: int = 5,
    ) -> None:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if isinstance(refinement_every, bool) or not isinstance(refinement_every, int) or refinement_every < 1:
            raise ValueError("refinement_every must be a positive integer")
        project_root = Path(__file__).resolve().parents[3]
        default_dir = project_root / "outputs" / "ahe" / f"run_{uuid4().hex}"
        self.run_dir = Path(run_dir or default_dir).expanduser().resolve()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.candidate_count = candidate_count
        self.state: AHEState | None = None
        self._agent: AHEAgent | None = None
        self._refinement_round = 0
        self._observed_training_tasks = 0
        self.refinement_every = refinement_every
        # XBR's bounds, retained without importing its embedded credentials or
        # changing the benchmark's azure/gpt-5.6-sol model/reasoning settings.
        self.code_agent_limits = {"max_context_tokens": 100_000, "max_iterations": 16, "max_tokens": 4_096}
        self.evolve_agent_limits = {"max_context_tokens": 120_000, "max_iterations": 64, "max_tokens": 8_192}
        self._pending_batch: CandidateBatch | None = None
        self.last_refinement_report: dict[str, Any] | None = None

    def checkpoint_policy(self) -> dict[str, Any]:
        """Persist method changes that would make old checkpoints incomparable."""
        return {
            "version": "xbr-visual-v1",
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
        self.state = initialize(self.run_dir)
        apply_agent_limits(self.state.current_workspace / "code_agent.yaml", self.code_agent_limits)
        apply_agent_limits(self.state.evolve_agent_dir / "evolve_agent.yaml", self.evolve_agent_limits)

    def build(self) -> AHEAgent:
        """Return the Agent bound to the currently selected harness.

        Repeated calls within one task reuse the NexAU process. ``load``
        closes the prior task's process; ``select`` closes the active process
        when it commits a different workspace.
        """

        if self.state is None:
            raise RuntimeError("AHEAdaptor.init() must be called before build()")
        if self._agent is None:
            self._agent = build_agent(
                self.state,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                timeout_seconds=self.timeout_seconds,
                agent_limits=self.code_agent_limits,
            )
        return self._agent

    def load(self) -> AHEAgent:
        """Isolate NexAU history between tasks, retaining the selected harness."""

        self._release_agent()
        return self.build()

    def _release_agent(self) -> None:
        if self._agent is not None:
            self._agent.close()
            self._agent = None

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
        if self._observed_training_tasks % self.refinement_every:
            self._release_agent()
            self.last_refinement_report = {
                "status": "skipped", "reason": "refinement_interval",
                "observed_training_tasks": self._observed_training_tasks,
                "refinement_every": self.refinement_every,
                "token_usage_report": {
                    "method": "no_model_calls", "complete": True, "total_tokens": 0,
                    "prompt_num_tokens": 0, "response_num_tokens": 0,
                    "reasoning_num_tokens": 0,
                },
            }
            return None
        next_round = self._refinement_round + 1
        batch = generate_candidates(
            self.state, self.build(), feedback_results,
            output_dir=output_dir, iteration=next_round, model=self.model,
            reasoning_effort=self.reasoning_effort,
            candidate_count=self.candidate_count,
            agent_limits=self.code_agent_limits,
        )
        self._refinement_round = next_round
        self.last_refinement_report = json.loads(
            (Path(output_dir) / "ahe_refine.json").read_text(encoding="utf-8"))
        self._pending_batch = batch
        return batch

    def select(
        self,
        batch: CandidateBatch,
        *,
        parent_evaluation: Any,
        candidate_evaluations: Mapping[str, Any],
        output_dir: str | Path,
    ) -> SelectionOutcome:
        """Resolve exactly the batch returned by the last refine call.

        ``select.py`` owns the score comparison and state transition. Once
        it succeeds, every Agent from the evaluated generation is closed;
        the next ``build()`` starts a clean process on the chosen workspace.
        An invalid evaluation leaves this batch pending for correction.
        """

        if self.state is None or self._pending_batch is None:
            raise RuntimeError("AHEAdaptor.select() has no pending candidate batch")
        if batch is not self._pending_batch:
            raise RuntimeError("AHEAdaptor.select() received a stale candidate batch")
        outcome = select_candidate(
            self.state, batch, parent_evaluation=parent_evaluation,
            candidate_evaluations=candidate_evaluations, output_dir=output_dir,
        )
        self.close()
        return outcome

    def close(self) -> None:
        """Release parent and candidate processes, including between generations."""

        self._release_agent()
        if self._pending_batch is not None:
            for candidate in self._pending_batch.candidates:
                candidate.agent.close()
            self._pending_batch = None


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
        state = controller.session.adaptor.state
        assert state is not None
        # Limit only this fresh run's copied configs. AHE's cloned source and
        # normal benchmark runs keep their original iteration budgets.
        for config_path, original, limited in (
            (state.current_workspace / "code_agent.yaml", "max_iterations: 300", "max_iterations: 8"),
            (state.evolve_agent_dir / "evolve_agent.yaml", "max_iterations: 500", "max_iterations: 16"),
        ):
            source = config_path.read_text(encoding="utf-8")
            if original not in source:
                raise RuntimeError(f"Unexpected AHE iteration limit in {config_path}")
            config_path.write_text(source.replace(original, limited, 1), encoding="utf-8")
        print(f"AHE refine smoke directory: {output_dir}", flush=True)
        record = _run_one(task, controller=controller, ordinal=1, output_dir=output_dir)
        selection = record.get("selection") or {}
        print(f"AHE selection: {selection.get('status')}; score: {record.get('relative_score')}", flush=True)
        if selection.get("status") not in {"committed", "unchanged"}:
            raise RuntimeError(f"AHE refine smoke failed; inspect {output_dir}")
    finally:
        controller.close()
