"""SelfImprove-Bench adapter for Recuris Skill Memory evolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.method_support import StructuredStateAdapter


_MEMORY_KEYS = ("E", "W", "rho", "C")


class RecurisAdapter(StructuredStateAdapter):
    """Evolve Recuris's E/W/rho/C memory while keeping the task model frozen."""

    method_name = "recuris"
    state_filename = "skill_memory.json"
    allowed_options = frozenset({"max_entries_per_component"})

    def initial_state(self) -> Mapping[str, Any]:
        return {
            "skill_memory": {
                "E": [],
                "W": ["Track verified task state separately from assumptions."],
                "rho": ["Retrieve only memories whose trigger matches the current verified state."],
                "C": ["Before acting, verify prerequisites and the required response schema."],
            },
            "revision_log": [],
        }

    def acting_instructions(self) -> str:
        return """You are a frozen downstream agent augmented by Recuris Skill Memory.
Use E as experiential memory, W as working-state procedure, rho as retrieval policy,
and C as control/checking rules. Ground every retrieval in verified task state and never
treat memory as an answer key. Return exactly the action or answer the task requests."""

    def refinement_instructions(self, evidence: Sequence[Mapping[str, Any]]) -> str:
        return """You are the Recuris meta-agent. Diagnose the complete trajectory at the
smallest responsible Skill Memory component: E (experience), W (working memory), rho
(state-grounded retrieval), or C (control/checking). Patch only memory components for
which the trajectory shows both an action opportunity and a memory-owned cause. Keep the
downstream task model frozen. Preserve useful entries, make triggers explicit, and add a
revision_log record linking each change to concrete feedback evidence.

Return: {"state": {"skill_memory": {"E": [...], "W": [...], "rho": [...],
"C": [...]}, "revision_log": [...]}, "diagnosis": [...]}."""

    def normalize_candidate(
        self, payload: Mapping[str, Any], *, current: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = super().normalize_candidate(payload, current=current)
        memory = state.get("skill_memory")
        if not isinstance(memory, Mapping) or set(memory) != set(_MEMORY_KEYS):
            raise ValueError("Recuris candidate must contain exactly E, W, rho, and C")
        maximum = int(self.options.get("max_entries_per_component", 100))
        for key in _MEMORY_KEYS:
            entries = memory[key]
            if not isinstance(entries, list) or len(entries) > maximum:
                raise ValueError(f"Recuris {key} must be an array of at most {maximum} entries")
        if not isinstance(state.get("revision_log"), list):
            raise ValueError("Recuris revision_log must be an array")
        return state


__all__ = ["RecurisAdapter"]
