"""SelfImprove-Bench adapter for Regularized Recursive Self-Improvement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.method_support import StructuredStateAdapter

from .rrsi.components import K, K_STR


class RRSIAdapter(StructuredStateAdapter):
    """Apply RRSI's component tagging, edit budget, noise floor, and novelty gate."""

    method_name = "rrsi"
    state_filename = "harness_frontier.json"
    allowed_options = frozenset({
        "edit_budget", "delta", "beta0", "beta1", "w_s", "w_c", "w_n",
    })

    def __init__(self, context):
        super().__init__(context)
        self.edit_budget = int(self.options.get("edit_budget", 2))
        self.delta = float(self.options.get("delta", 0.0))
        self.beta0 = float(self.options.get("beta0", 0.10))
        self.beta1 = float(self.options.get("beta1", 40.0))
        self.w_s = float(self.options.get("w_s", 100.0))
        self.w_c = float(self.options.get("w_c", 15.0))
        self.w_n = float(self.options.get("w_n", 0.5))
        if self.edit_budget < 1:
            raise ValueError("rrsi edit_budget must be positive")

    def initial_state(self) -> Mapping[str, Any]:
        return {
            "harness": {
                "prompt": "Solve carefully and emit only the requested answer/action.",
                "control_flow": [], "config": {}, "output_plumbing": [],
                "context_mgmt": [], "client_tool": [], "skill": [],
                "memory": [], "subagent": [],
            },
            "accepted_components": [],
            "frontier": {"best_score": None, "round": 0},
        }

    def acting_instructions(self) -> str:
        return """You are the incumbent RRSI harness. Apply the active prompt, control
flow, context management, skills, memory, tools, and output plumbing. Structural entries
are operational policies, not permission to invent unavailable tools. Follow the task's
response schema exactly."""

    def refinement_instructions(self, evidence: Sequence[Mapping[str, Any]]) -> str:
        return f"""You are RRSI's evidence-aware regularized proposer. Analyze the current
incumbent's trajectory, preserve successful habits, and propose at most {self.edit_budget}
atomic edits. Every edit must name one component from {K}, state a falsifiable hypothesis,
and supply the replacement value for that component. Prefer low-complexity edits; when the
incumbent is stalled, one edit may explore a previously unused structural component from
{K_STR}. Do not smuggle task answers into the harness.

Return: {{"state": {{"harness": {{...all component keys...}},
"accepted_components": [...], "frontier": {{"best_score": number|null,
"round": integer}}}}, "edits": [{{"component": "...", "hypothesis": "..."}}]}}."""

    def normalize_candidate(
        self, payload: Mapping[str, Any], *, current: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = super().normalize_candidate(payload, current=current)
        harness = state.get("harness")
        if not isinstance(harness, Mapping) or set(harness) != set(K):
            raise ValueError("RRSI harness must preserve the complete component vocabulary")
        edits = payload.get("edits")
        if not isinstance(edits, list) or not 1 <= len(edits) <= self.edit_budget:
            raise ValueError(f"RRSI proposal must contain 1..{self.edit_budget} edits")
        for edit in edits:
            if not isinstance(edit, Mapping) or edit.get("component") not in K:
                raise ValueError("every RRSI edit must declare a valid component")
            if not str(edit.get("hypothesis") or "").strip():
                raise ValueError("every RRSI edit must include a hypothesis")
        accepted = list(current.get("accepted_components", []))
        for edit in edits:
            component = edit["component"]
            if component not in accepted:
                accepted.append(component)
        state["accepted_components"] = accepted
        current_frontier = current.get("frontier", {})
        state["frontier"] = {
            "best_score": current_frontier.get("best_score"),
            "round": int(current_frontier.get("round", 0)) + 1,
        }
        return state

    def accept_candidate(self, parent: Any, candidate: Any) -> tuple[bool, str]:
        parent_score = getattr(parent, "relative_score", None)
        candidate_score = getattr(candidate, "relative_score", None)
        if getattr(candidate, "status", "") != "ok" or not isinstance(candidate_score, (int, float)):
            return False, "candidate evaluation is not admissible"
        if not isinstance(parent_score, (int, float)):
            return True, "candidate completed and parent score is unavailable"
        delta_score = float(candidate_score) - float(parent_score)
        edits = (self._pending or {}).get("report", {}).get("proposal", {}).get("edits", [])
        used = set(self._state.get("accepted_components", []))
        novelty = len({
            edit.get("component") for edit in edits if isinstance(edit, Mapping)
            and edit.get("component") in set(K_STR) - used
        })
        parent_tokens = getattr(parent, "total_tokens", None)
        candidate_tokens = getattr(candidate, "total_tokens", None)
        delta_cost = 0.0
        if isinstance(parent_tokens, int) and parent_tokens > 0 and isinstance(candidate_tokens, int):
            delta_cost = (candidate_tokens - parent_tokens) / parent_tokens
        if delta_score > self.delta:
            cost_limit = self.beta0 + self.beta1 * delta_score
            cost_ok = delta_cost <= cost_limit
            gate_detail = f"dC={delta_cost:+.6f} <= budget={cost_limit:+.6f}"
        else:
            shaped = self.w_s * delta_score - self.w_c * delta_cost + self.w_n * novelty
            cost_ok = shaped > 0
            gate_detail = f"shaped={shaped:+.6f}, novelty={novelty}"
        best_score = self._state.get("frontier", {}).get("best_score")
        score_floor = float(best_score) - self.delta if isinstance(best_score, (int, float)) else float(parent_score) - self.delta
        accepted = float(candidate_score) >= score_floor and cost_ok
        return accepted, (
            f"RRSI candidate admissible ({gate_detail})"
            if accepted else
            f"failed RRSI floor/cost gate (dS={delta_score:+.6f}, {gate_detail})"
        )

    def select(self, **kwargs):
        candidate_evaluations = kwargs["candidate_evaluations"]
        result = super().select(**kwargs)
        if result.status == "committed" and result.committed_candidate_id is not None:
            score = candidate_evaluations[result.committed_candidate_id].relative_score
            previous = self._state["frontier"].get("best_score")
            if isinstance(score, (int, float)):
                self._state["frontier"]["best_score"] = (
                    max(float(previous), float(score))
                    if isinstance(previous, (int, float)) else float(score)
                )
            self._persist_state()
        return result


__all__ = ["RRSIAdapter"]
