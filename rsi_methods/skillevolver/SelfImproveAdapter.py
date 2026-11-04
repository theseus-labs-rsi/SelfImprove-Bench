"""SelfImprove-Bench adapter for SkillEvolver's persistent skill tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.method_support import StructuredStateAdapter


class SkillEvolverAdapter(StructuredStateAdapter):
    """Retain SkillEvolver's compact SKILL router plus reusable assets/evidence."""

    method_name = "skillevolver"
    state_filename = "skill_tree.json"
    allowed_options = frozenset({"max_skill_chars"})

    def initial_state(self) -> Mapping[str, Any]:
        return {
            "SKILL.md": (
                "# General task strategy\n\n"
                "Verify the current state, choose the smallest valid action, and obey the "
                "requested output schema.\n"
            ),
            "references": {},
            "scripts": {},
            "evidence": [],
        }

    def acting_instructions(self) -> str:
        return """You are an agent using an evolved SkillEvolver skill tree. SKILL.md is
the routing and decision layer; references contain durable lessons; scripts describe only
available reusable procedures. Apply conditional guidance, verify transfer assumptions,
and return exactly the task-requested answer/action."""

    def refinement_instructions(self, evidence: Sequence[Mapping[str, Any]]) -> str:
        return """You are SkillEvolver. Explore contrasting strategy overlays in thought,
analyze the supplied execution trace and evaluator feedback, and update one persistent
transferable skill tree. Keep SKILL.md compact as the routing/decision layer. Put durable
failure/success contrasts in references and reusable deterministic procedures in scripts.
Preserve assets that remain valid; add conditional branches instead of overfitting to the
latest task. Never include the current task's answer.

Return: {"state": {"SKILL.md": "...", "references": {"name.md": "..."},
"scripts": {"name": "..."}, "evidence": [...]}, "exploration_contrasts": [...]}."""

    def normalize_candidate(
        self, payload: Mapping[str, Any], *, current: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = super().normalize_candidate(payload, current=current)
        if set(state) != {"SKILL.md", "references", "scripts", "evidence"}:
            raise ValueError("SkillEvolver candidate must preserve the complete skill tree")
        skill = state["SKILL.md"]
        if not isinstance(skill, str) or not skill.strip():
            raise ValueError("SkillEvolver SKILL.md must be non-empty")
        if len(skill) > int(self.options.get("max_skill_chars", 40000)):
            raise ValueError("SkillEvolver SKILL.md exceeds max_skill_chars")
        if not isinstance(state["references"], Mapping) or not isinstance(state["scripts"], Mapping):
            raise ValueError("SkillEvolver references and scripts must be objects")
        if not isinstance(state["evidence"], list):
            raise ValueError("SkillEvolver evidence must be an array")
        return state


__all__ = ["SkillEvolverAdapter"]
