"""SelfImprove-Bench adapter for SkillOpt's validation-gated skill optimizer."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.method_support import _json_text, _mapping_value, _list_value
from benchmark.contracts import to_jsonable
from benchmark.native_support import NativeAdapter, native_package, valid_score
from benchmark.resume_state import atomic_json
from copy import deepcopy


class SkillOptAdapter(NativeAdapter):
    """Use SkillOpt's native bounded edit application and paired validation gate."""

    method_name = "skillopt"
    state_filename = "best_skill.json"
    allowed_options = frozenset({"max_edits", "max_skill_chars", "minibatch_size", "workers",
        "failure_only", "validation_trials", "scheduler", "min_edits", "gate_metric"})

    def __init__(self, context):
        super().__init__(context)
        self.max_edits = int(self.options.get("max_edits", 4))
        if self.max_edits < 1:
            raise ValueError("skillopt max_edits must be positive")
        self.validation_trials = int(self.options.get("validation_trials", 5))
        if min(self.validation_trials, int(self.options.get("minibatch_size", 8)), int(self.options.get("workers", 1))) < 1:
            raise ValueError("SkillOpt trials/minibatch_size/workers must be positive")

    def _agent(self, state):
        # Rejection and optimization history are not target-agent instructions.
        return super()._agent({"best_skill": state["best_skill"]})

    def refine(self, feedback, *, output_dir):
        self.start_refinement(feedback, output_dir)
        native_package("skillopt", Path(__file__).parent / "skillopt")
        from skillopt.gradient.reflect import run_minibatch_reflect
        from skillopt.gradient.aggregate import merge_patches
        from skillopt.optimizer.clip import rank_and_select
        from skillopt.optimizer.scheduler import build_scheduler
        from skillopt.evaluation.gate import evaluate_gate, select_gate_score
        from skillopt.model.transport import optimizer_transport
        state = deepcopy(self._state)
        step = int(state.get("step", 0)) + 1
        scheduler = build_scheduler(str(self.options.get("scheduler", "constant")),
            max_lr=self.max_edits, min_lr=int(self.options.get("min_edits", 1)),
            total_steps=max(1, self._method_context.task_count))
        budget = scheduler.get_lr(step)
        results = []
        for i, item in enumerate(feedback):
            identity = f"trace-{i}"
            conversation = [{"step": s.step_index, "action": s.action,
                "env_feedback": s.environment_response, "reasoning": to_jsonable(s.agent_call),
                "prompt": s.prompt} for s in item.trajectory]
            # Include prompts as real conversation entries, not a discarded key.
            conversation = [entry for s, action in zip(item.trajectory, conversation)
                            for entry in ({"role": "user", "content": s.prompt}, action)]
            atomic_json(output_dir / "predictions" / identity / "conversation.json", conversation)
            results.append({"id": identity, "hard": int(item.evaluation.success),
                            "soft": item.evaluation.score or 0.0,
                            "task_description": item.trajectory[0].prompt if item.trajectory else "Game task",
                            "fail_reason": item.evaluation.feedback})
        def transport(*, system, user, stage="optimizer", **kwargs):
            return self.ask_text(system + "\n\n" + user, stage=stage), {}
        token = optimizer_transport.set(transport)
        try:
            raw = run_minibatch_reflect(results, state["best_skill"],
                str(output_dir / "predictions"), str(output_dir / "patches"),
                workers=int(self.options.get("workers", 1)),
                failure_only=bool(self.options.get("failure_only", False)),
                minibatch_size=int(self.options.get("minibatch_size", 8)), edit_budget=budget,
                random_seed=self._method_context.seed + step,
                step_buffer_context=_json_text(state.get("step_buffer", [])))
            failures = [p["patch"] for p in raw if p and p.get("source_type") == "failure"]
            successes = [p["patch"] for p in raw if p and p.get("source_type") == "success"]
            merged = merge_patches(state["best_skill"], failures, successes,
                workers=int(self.options.get("workers", 1)), verbose=False)
            patch = rank_and_select(state["best_skill"], merged, budget)
        finally:
            optimizer_transport.reset(token)
        previous_budget, self.max_edits = self.max_edits, budget
        try:
            candidate = self.normalize_candidate({"patch": patch}, current=state)
        finally:
            self.max_edits = previous_budget
        atomic_json(output_dir / "native_patch.json", {"raw": raw, "merged": merged, "selected": patch})
        accepted, gate = False, None
        if candidate["best_skill"] != state["best_skill"]:
            base = [self.rollout(state, label=f"base-validation-{i}", split="validation", index=i)
                    for i in range(self.validation_trials)]
            cand = [self.rollout(candidate, label=f"candidate-validation-{i}", split="validation", index=i)
                    for i in range(self.validation_trials)]
            def scores(runs):
                hard, soft, failures = [], [], []
                for index, result in enumerate(runs):
                    record = result.get("record", {})
                    hard.append(bool(record.get("success")) if record.get("status") == "ok" else False)
                    try:
                        soft.append(valid_score(result))
                    except (RuntimeError, KeyError, TypeError) as exc:
                        soft.append(0.0)
                        failures.append({"trial": index, "error": str(exc),
                            "status": record.get("status")})
                return sum(hard) / len(runs), sum(soft) / len(runs), failures
            bh, bs, base_failures = scores(base)
            ch, cs, candidate_failures = scores(cand)
            metric = str(self.options.get("gate_metric", "hard"))
            baseline = select_gate_score(bh, bs, metric=metric)
            gate = evaluate_gate(candidate["best_skill"], ch, state["best_skill"], baseline,
                state["best_skill"], baseline, int(state.get("best_step", 0)), step,
                cand_soft=cs, metric=metric)
            accepted = gate.action != "reject"
        result = candidate if accepted else state
        result["step"] = step
        result["best_step"] = gate.best_step if gate else state.get("best_step", 0)
        result["step_buffer"] = [*state.get("step_buffer", []),
                                  {"step": step, "patch": patch, "accepted": accepted}]
        if not accepted and patch.get("edits"):
            result.setdefault("rejected_edits", []).append(patch)
        self._state = result
        self.finish_refinement(result, accepted=accepted,
            detail={"native_gate": gate, "edit_budget": budget, "step": step,
                    "base_validation_failures": base_failures if gate else [],
                    "candidate_validation_failures": candidate_failures if gate else []})

    def initial_state(self) -> Mapping[str, Any]:
        return {
            "best_skill": (
                "# Executive strategy\n\n"
                "Verify state and prerequisites, choose the smallest valid action, and "
                "follow the requested response schema exactly.\n"
            ),
            "rejected_edits": [],
            "step": 0,
        }

    def acting_instructions(self) -> str:
        return """You are a frozen target agent with one optimized executive skill.
Apply best_skill as the strategy layer with zero extra inference calls. Rejected edits are
optimizer history and must not be applied. Return only the task-requested answer/action."""

    def refinement_instructions(self, evidence: Sequence[Mapping[str, Any]]) -> str:
        return f"""You are SkillOpt's reflect–aggregate–select optimizer. Convert the full
scored rollout and trajectory into at most {self.max_edits} bounded edits on best_skill.
Use only append, insert_after, replace, or delete operations. Make edits conditional and
transferable, keep the target model frozen, and do not encode task answers. Previously
rejected edits are negative optimizer evidence. Candidate acceptance is decided later by
paired benchmark validation. Return an empty edits array when the current skill should be
preserved; an empty patch is a valid no-op refinement.

Return: {{"patch": {{"reasoning": "...", "edits": [{{"op": "append|insert_after|replace|delete",
"target": "...", "content": "..."}}]}}}}."""

    @staticmethod
    def _skillopt_update():
        module_name = "_selfimprove_skillopt_update"
        module = sys.modules.get(module_name)
        if module is None:
            path = Path(__file__).resolve().parent / "skillopt" / "optimizer" / "skill.py"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load SkillOpt update operator from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        return module.apply_patch_with_report

    def normalize_candidate(
        self, payload: Mapping[str, Any], *, current: Mapping[str, Any],
    ) -> dict[str, Any]:
        patch = payload.get("patch")
        if isinstance(patch, str):
            patch = _mapping_value(patch, "SkillOpt patch")
        if not isinstance(patch, Mapping):
            raise ValueError("SkillOpt optimizer response requires patch.edits")
        patch = dict(patch)
        patch["edits"] = _list_value(patch.get("edits"), "SkillOpt patch.edits")
        edits = patch["edits"]
        if len(edits) > self.max_edits:
            raise ValueError(f"SkillOpt patch must contain at most {self.max_edits} edits")
        if edits:
            updated, reports = self._skillopt_update()(
                str(current["best_skill"]), dict(patch)
            )
        else:
            updated = str(current["best_skill"])
            reports = [{
                "status": "skipped_empty_patch",
                "reasoning": str(patch.get("reasoning") or ""),
            }]
        if not updated.strip():
            raise ValueError("SkillOpt patch produced an empty skill")
        if len(updated) > int(self.options.get("max_skill_chars", 40000)):
            raise ValueError("SkillOpt candidate exceeds max_skill_chars")
        return {
            "best_skill": updated,
            "rejected_edits": list(current.get("rejected_edits", [])),
            "step": int(current.get("step", 0)) + 1,
            "last_edit_report": reports,
        }



__all__ = ["SkillOptAdapter"]
