"""SelfImprove-Bench adapter for SkillEvolver's persistent skill tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.method_support import _json_text, _mapping_value, feedback_evidence
from benchmark.native_support import NativeAdapter, valid_score, write_files
from benchmark.artifact_agent import ArtifactAgent
from benchmark.resume_state import atomic_json
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib


class SkillEvolverAdapter(NativeAdapter):
    """Retain SkillEvolver's compact SKILL router plus reusable assets/evidence."""

    method_name = "skillevolver"
    state_filename = "skill_tree.json"
    allowed_options = frozenset({"max_skill_chars", "iterations", "exploration_trials",
                                "validation_trials", "workers", "reward_signal_mode"})

    def __init__(self, context):
        super().__init__(context)
        self.iterations = int(self.options.get("iterations", 2))
        self.exploration_trials = int(self.options.get("exploration_trials", 4))
        self.validation_trials = int(self.options.get("validation_trials", 5))
        self.workers = int(self.options.get("workers", 4))
        if min(self.iterations, self.exploration_trials, self.validation_trials, self.workers) < 1:
            raise ValueError("SkillEvolver iteration/trial/worker settings must be positive")
        if self.options.get("reward_signal_mode", "continuous") not in {"discrete", "continuous"}:
            raise ValueError("reward_signal_mode must be discrete or continuous")

    def _agent(self, state):
        files = {"SKILL.md": state["SKILL.md"]}
        for directory in ("references", "scripts"):
            files.update({f"{directory}/{name}": text for name, text in state[directory].items()})
        digest = hashlib.sha256(_json_text(files).encode()).hexdigest()[:16]
        return ArtifactAgent(self, files, self.acting_instructions(), always_on=("SKILL.md",),
                             root=self.state_dir / "agents" / digest)

    def refine(self, feedback, *, output_dir):
        self.start_refinement(feedback, output_dir)
        protocol = (Path(__file__).parent / "skill-evolver" / "SKILL.md").read_text()
        mode = self.options.get("reward_signal_mode", "continuous")
        protocol += ("\nUse binary task success as the reward signal." if mode == "discrete"
                     else "\nUse the continuous relative task score as the reward signal.")
        current = deepcopy(self._state)
        evidence = feedback_evidence(feedback)
        versions = [deepcopy(current)]
        contrasts = []
        for iteration in range(self.iterations):
            variants = []
            for trial in range(self.exploration_trials):
                variants.append(self.propose(current, evidence,
                    stage=f"iteration-{iteration}-overlay-{trial}",
                    instruction=protocol + "\nExploration phase: create a deliberately contrasting strategy overlay."
                    + "\nEarlier overlays in this iteration:\n" + _json_text(variants)))
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self.rollout, variant,
                    label=f"iteration-{iteration}-trial-{trial}") for trial, variant in enumerate(variants)]
                runs = [future.result() for future in futures]
            run_scores = []
            for run in runs:
                try:
                    run_scores.append(valid_score(run))
                except (RuntimeError, KeyError, TypeError):
                    run_scores.append(0.0)
            evidence = feedback_evidence([r["feedback"] for r in runs])
            analysis = self.ask(protocol + "\nAnalyze all real trial traces below. Identify missed guidance, "
                "unnecessary detours, and durable contrasts. Return JSON.\n"
                + _json_text({"variants": variants, "traces": evidence}), stage=f"iteration-{iteration}-analysis")
            current = self.propose(current, evidence, stage=f"iteration-{iteration}-update",
                instruction=protocol + "\nUpdate phase: distill the next skill from these real traces.\n"
                + _json_text(analysis))
            versions.append(deepcopy(current))
            contrasts.append({"iteration": iteration, "analysis": analysis,
                              "scores": [int(r.get("record", {}).get("status") == "ok"
                                               and bool(r.get("record", {}).get("success")))
                                         if mode == "discrete"
                                         else score for r, score in zip(runs, run_scores)]})
            atomic_json(output_dir / "versions" / f"v{iteration + 1}.json", current)
        # Native finalize selects a snapshot BEFORE the one final validation;
        # validation data never feeds back into exploration or version choice.
        choice = self.ask(protocol + "\nFinalize phase: select the best transferable skill version "
            "using training evidence only. Return JSON {\"version\": integer, \"reason\": str}.\n"
            + _json_text({"versions": versions, "contrasts": contrasts}), stage="finalize")
        version = choice.get("version")
        if isinstance(version, str) and version.strip().isdigit():
            version = int(version.strip())
        elif isinstance(version, str):
            version = _mapping_value(version, "SkillEvolver finalize response").get("version")
            if isinstance(version, str) and version.strip().isdigit():
                version = int(version.strip())
        if isinstance(version, bool) or not isinstance(version, int) or not 0 <= version < len(versions):
            raise ValueError("SkillEvolver finalize returned an invalid version")
        selected = versions[version]
        validation = [self.rollout(selected, label=f"final-validation-{i}", split="validation", index=i)
                      for i in range(self.validation_trials)]
        scores, validation_failures = [], []
        for index, result in enumerate(validation):
            try:
                scores.append(valid_score(result))
            except (RuntimeError, KeyError, TypeError) as exc:
                scores.append(0.0)
                validation_failures.append({"trial": index, "error": str(exc),
                    "status": result.get("record", {}).get("status")})
        # The native workflow records final validation; it does not replace its
        # finalization policy with a new strict paired-score admission rule.
        self.finish_refinement(selected, accepted=True,
            detail={"finalize": choice, "validation_scores": scores,
                    "validation_failures": validation_failures, "contrasts": contrasts})

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
        return """You are SkillEvolver. Create the strategy artifact for the requested phase,
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
        for key in ("references", "scripts"):
            if isinstance(state.get(key), str):
                state[key] = _mapping_value(state[key], f"SkillEvolver {key}")
        if isinstance(state.get("evidence"), str):
            from benchmark.method_support import _list_value
            state["evidence"] = _list_value(state["evidence"], "SkillEvolver evidence")
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
        for directory in ("references", "scripts"):
            for name, content in state[directory].items():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or not isinstance(content, str):
                    raise ValueError("SkillEvolver assets must be relative text files")
                if directory == "scripts" and name.endswith(".py"):
                    compile(content, name, "exec")
        return state


__all__ = ["SkillEvolverAdapter"]
