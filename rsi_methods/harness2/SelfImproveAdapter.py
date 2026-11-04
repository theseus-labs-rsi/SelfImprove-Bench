"""SelfImprove-Bench adapter for Harness²'s recursive harness state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.method_support import StructuredStateAdapter


_COMPONENTS = (
    "system_prompt", "guardrails", "memory", "tool_guidance",
    "subagents", "skills", "scripts", "plugins",
)


class Harness2Adapter(StructuredStateAdapter):
    """Keep Harness²'s eight-component harness and improvement memory intact."""

    method_name = "harness2"
    state_filename = "harness.json"
    allowed_options = frozenset({"recursion_mode", "proposal_count"})

    def initial_state(self) -> Mapping[str, Any]:
        return {
            "components": {
                "system_prompt": "Solve the task carefully and return exactly the requested action.",
                "guardrails": [],
                "memory": [],
                "tool_guidance": [],
                "subagents": [],
                "skills": [],
                "scripts": [],
                "plugins": [],
            },
            "improvement_harness": {
                "adopted_lessons": [],
                "contrastive_evidence": [],
            },
            "recursion_mode": str(self.options.get("recursion_mode", "sequential")),
        }

    def acting_instructions(self) -> str:
        return """You are an agent executed under a Harness² harness. Apply the active
eight-component harness as operational guidance. Treat memory and adopted lessons as
transferable evidence, not task answers. Obey the task's response format exactly."""

    def refinement_instructions(self, evidence: Sequence[Mapping[str, Any]]) -> str:
        return """You are the Harness² Propose–Probe–Compose improver. Use the complete
task trajectory and external evaluation feedback below as the probe evidence. Contrast
successful and failed behavior, then compose one revised persistent harness. Preserve the
eight editable components (system_prompt, guardrails, memory, tool_guidance, subagents,
skills, scripts, plugins). Update improvement_harness with evidence-supported lessons.
Do not encode the current task's answer or benchmark-specific hidden targets.

Return: {"state": {"components": {...all eight keys...},
"improvement_harness": {"adopted_lessons": [...], "contrastive_evidence": [...]},
"recursion_mode": "parallel|sequential"}, "composition_notes": "..."}"""

    def normalize_candidate(
        self, payload: Mapping[str, Any], *, current: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = super().normalize_candidate(payload, current=current)
        components = state.get("components")
        if not isinstance(components, Mapping) or set(components) != set(_COMPONENTS):
            raise ValueError("Harness² candidate must preserve exactly the eight harness components")
        improvement = state.get("improvement_harness")
        if not isinstance(improvement, Mapping):
            raise ValueError("Harness² candidate requires improvement_harness")
        for key in ("adopted_lessons", "contrastive_evidence"):
            if not isinstance(improvement.get(key), list):
                raise ValueError(f"Harness² improvement_harness.{key} must be an array")
        if state.get("recursion_mode") not in {"parallel", "sequential"}:
            raise ValueError("Harness² recursion_mode must be parallel or sequential")
        return state


__all__ = ["Harness2Adapter"]
