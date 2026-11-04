"""SelfImprove-Bench adapter for SkillOpt's validation-gated skill optimizer."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.method_support import StructuredStateAdapter


class SkillOptAdapter(StructuredStateAdapter):
    """Use SkillOpt's native bounded edit application and paired validation gate."""

    method_name = "skillopt"
    state_filename = "best_skill.json"
    allowed_options = frozenset({"max_edits", "max_skill_chars"})

    def __init__(self, context):
        super().__init__(context)
        self.max_edits = int(self.options.get("max_edits", 4))
        if self.max_edits < 1:
            raise ValueError("skillopt max_edits must be positive")

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
        if not isinstance(patch, Mapping) or not isinstance(patch.get("edits"), list):
            raise ValueError("SkillOpt optimizer response requires patch.edits")
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

    def select(self, **kwargs):
        pending = self._pending
        result = super().select(**kwargs)
        if pending is not None and result.status == "unchanged":
            proposal = pending.get("report", {}).get("proposal", {}).get("patch", {})
            if proposal.get("edits"):
                self._state.setdefault("rejected_edits", []).append(proposal)
                self._persist_state()
        return result


__all__ = ["SkillOptAdapter"]
