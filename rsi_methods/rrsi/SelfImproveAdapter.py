"""SelfImprove-Bench adapter for Regularized Recursive Self-Improvement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.method_support import _json_text, _list_value, feedback_evidence
from benchmark.native_support import NativeAdapter, valid_score, write_files
from benchmark.artifact_agent import ArtifactAgent
from .rrsi.calibrate import calibrate
from .rrsi.history import History, exploration, stall_flag
from .rrsi.config import RRSIConfig
from .rrsi.evaluate import TaskResult, aggregate
from .rrsi.schedule import edit_budget
from .rrsi.selection import Candidate, select_round
from copy import deepcopy
import hashlib
import json

from .rrsi.components import K, K_STR


class RRSIAdapter(NativeAdapter):
    """Apply RRSI's component tagging, edit budget, noise floor, and novelty gate."""

    method_name = "rrsi"
    state_filename = "harness_frontier.json"
    allowed_options = frozenset({
        "edit_budget", "delta", "beta0", "beta1", "w_s", "w_c", "w_n",
        "trials", "candidate_count", "b_min", "b_max", "stall_window", "repair_rounds",
        "invalid_missing_frac",
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
        self.cfg = RRSIConfig(T=max(1, context.task_count),
            k=int(self.options.get("trials", 2)), m=int(self.options.get("candidate_count", 2)),
            b_min=int(self.options.get("b_min", 1)), b_max=int(self.options.get("b_max", 4)),
            w=int(self.options.get("stall_window", 3)),
            repair_rounds=int(self.options.get("repair_rounds", 5)),
            invalid_missing_frac=float(self.options.get("invalid_missing_frac", .15)),
            delta=self.options.get("delta"), beta0=self.beta0, beta1=self.beta1,
            w_s=self.w_s, w_c=self.w_c, w_n=self.w_n)
        if min(self.cfg.k, self.cfg.m, self.cfg.b_min, self.cfg.w) < 1 or self.cfg.b_max < self.cfg.b_min:
            raise ValueError("invalid RRSI trial/candidate/budget settings")

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

    def _agent(self, state):
        harness = state["harness"]
        files = {"harness.md": _json_text({k: v for k, v in harness.items()
                  if k not in {"client_tool", "skill", "subagent"}})}
        for component, directory in (("client_tool", "scripts"), ("skill", "skills"),
                                     ("subagent", "sub_agents")):
            entries = harness[component]
            if isinstance(entries, dict):
                for name, content in entries.items():
                    files[f"{directory}/{name}"] = content
            elif entries:
                raise RuntimeError(f"legacy RRSI {component} list is not executable; restart with file-based artifacts")
        digest = hashlib.sha256(_json_text(files).encode()).hexdigest()[:16]
        return ArtifactAgent(self, files, self.acting_instructions(),
            always_on=tuple(k for k in files if "/" not in k), root=self.state_dir / "agents" / digest,
            actor_config=harness["config"])

    def refinement_instructions(self, evidence: Sequence[Mapping[str, Any]]) -> str:
        return f"""You are RRSI's evidence-aware regularized proposer. Analyze the current
incumbent's trajectory, preserve successful habits, and propose at most {self.edit_budget}
atomic edits. Every edit must name one component from {K}, state a falsifiable hypothesis,
and supply the replacement value for that component. Prefer low-complexity edits; when the
incumbent is stalled, one edit may explore a previously unused structural component from
{K_STR}. Do not smuggle task answers into the harness.

Executable structural components: client_tool is {{"helper.py": "Python source"}},
skill is {{"name/SKILL.md": "skill text"}}, subagent is {{"name/AGENT.md": "instructions"}}.
These files are read/run/delegated by the task actor. Other components are always-on
operating instructions; config may only specify temperature, max_output_tokens,
reasoning_effort. Never claim unavailable framework hooks are executable.

Return: {{"state": {{"harness": {{...all component keys...}},
"accepted_components": [...], "frontier": {{"best_score": number|null,
"round": integer}}}}, "edits": [{{"component": "...", "hypothesis": "..."}}]}}."""

    def normalize_candidate(
        self, payload: Mapping[str, Any], *, current: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = super().normalize_candidate(payload, current=current)
        harness = state.get("harness")
        if isinstance(harness, str):
            from benchmark.method_support import _mapping_value
            harness = _mapping_value(harness, "RRSI harness")
            state["harness"] = harness
        if not isinstance(harness, Mapping) or set(harness) != set(K):
            raise ValueError("RRSI harness must preserve the complete component vocabulary")
        for component in ("client_tool", "skill", "subagent"):
            entries = harness[component]
            if entries and not isinstance(entries, dict):
                raise ValueError(f"RRSI {component} must map relative filenames to text")
            for name, content in (entries or {}).items():
                from pathlib import Path
                if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(content, str):
                    raise ValueError("invalid RRSI artifact file")
                if component == "client_tool":
                    if not name.endswith(".py"):
                        raise ValueError("RRSI executable helpers must be Python files")
                    compile(content, name, "exec")
        cfg = harness["config"]
        if isinstance(cfg, str):
            from benchmark.method_support import _mapping_value
            cfg = _mapping_value(cfg, "RRSI harness.config")
            harness["config"] = cfg
        if not isinstance(cfg, dict) or set(cfg) - {"temperature", "max_output_tokens", "reasoning_effort"}:
            raise ValueError("unsupported RRSI actor config")
        if "max_output_tokens" in cfg and (not isinstance(cfg["max_output_tokens"], int) or cfg["max_output_tokens"] < 1):
            raise ValueError("RRSI max_output_tokens must be positive")
        if "temperature" in cfg and (not isinstance(cfg["temperature"], (int, float)) or not 0 <= cfg["temperature"] <= 2):
            raise ValueError("invalid RRSI temperature")
        edits = _list_value(payload.get("edits"), "RRSI edits")
        if not 1 <= len(edits) <= self.edit_budget:
            raise ValueError(f"RRSI proposal must contain 1..{self.edit_budget} edits")
        for edit in edits:
            if not isinstance(edit, Mapping) or edit.get("component") not in K:
                raise ValueError("every RRSI edit must declare a valid component")
            if not str(edit.get("hypothesis") or "").strip():
                raise ValueError("every RRSI edit must include a hypothesis")
        changed = {key for key in K if harness[key] != current["harness"][key]}
        declared = {edit["component"] for edit in edits}
        if changed != declared or len(declared) != len(edits):
            raise ValueError("RRSI edit declarations must match actual changed components exactly")
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

    def refine(self, feedback, *, output_dir):
        self.start_refinement(feedback, output_dir)
        round_index = len(self._history)
        self.edit_budget = (int(self.options["edit_budget"]) if "edit_budget" in self.options
            else edit_budget(round_index, self.cfg.T, self.cfg.b_min, self.cfg.b_max))
        incumbent = deepcopy(self._state)
        history = History(write_files(output_dir, {"edit_history.jsonl": "".join(
            json.dumps(row) + "\n" for row in incumbent.get("edit_history", []))}) / "edit_history.jsonl")
        def measure(state, label):
            runs = [self.rollout(state, label=f"{label}-{i}") for i in range(self.cfg.k)]
            missing = sum(r["record"].get("status") != "ok" for r in runs)
            ev = aggregate(label, self.cfg.k, {feedback[-1].task_id: TaskResult(
                rewards=[valid_score(r) if r["record"].get("status") == "ok" else 0.0 for r in runs],
                missing=missing,
                tokens=[r["record"].get("llm_metrics", {}).get("tokens", {}).get("total_tokens") for r in runs])})
            return ev, runs
        baseline, runs = measure(incumbent, "incumbent")
        if baseline.missing > self.cfg.invalid_missing_frac * baseline.n_expected:
            self._state = incumbent
            self.finish_refinement(incumbent, accepted=False,
                detail={"round": round_index, "status": "incumbent_invalid",
                        "missing": baseline.missing, "expected": baseline.n_expected})
            return
        # The original H0 is part of S*. Re-measure stored frontier harnesses
        # on this round's SAME task; scores from unrelated tasks never mix.
        frontier = incumbent.get("frontier", {}).get("harnesses", [])
        best = baseline.S
        for n, harness in enumerate(frontier):
            old = {**incumbent, "harness": harness}
            ev, _ = measure(old, f"frontier-{n}")
            best = max(best, ev.S)
        if self.cfg.delta is None:
            self.delta = calibrate([baseline], z=self.cfg.delta_z)["delta"]
        evidence = feedback_evidence([r["feedback"] for r in runs])
        analysis = self.ask("Diagnose failed and successful trajectories. Return JSON with "
            "falsifiable low-complexity improvement hypotheses and prune targets.\n" + _json_text(evidence), stage="analyst")
        trajectory = [h.get("incumbent", {}).get("S", baseline.S) for h in self._history] + [baseline.S]
        stalled = stall_flag(trajectory, round_index, self.cfg.w, self.delta)
        explore = exploration(round_index, stalled, history.tried(), self.cfg.m_draft)
        prune = history.prune_set(round_index, self.cfg.n_prune)
        candidates, states = [], {}
        for n in range(self.cfg.m):
            instruction = "\nAnalyst report:\n" + _json_text(analysis)
            instruction += "\nNative exploration/pruning/history:\n" + _json_text({
                "exploration": explore, "prune_set": prune, "history": history.render()})
            if stalled and n < self.cfg.m_draft:
                instruction += "\nExplore one previously unused structural component, or prune unproductive complexity."
            state, payload = None, None
            for attempt in range(max(1, self.cfg.repair_rounds + 1)):
                try:
                    payload = self.ask(self.refinement_instructions(evidence) + instruction
                        + "\nCurrent state:\n" + _json_text(incumbent)
                        + "\nEvidence:\n" + _json_text(evidence), stage=f"proposal-{n}-{attempt}")
                    state = self.normalize_candidate(payload, current=incumbent)
                    if stalled and n < self.cfg.m_draft and explore["untried"] and not any(
                            e["component"] in explore["untried"] for e in payload["edits"]):
                        raise ValueError("reserved exploratory candidate must edit an untried component")
                    critic = self.ask("Independently screen this edit for answer leakage, unsupported mechanisms, "
                        "and whether its hypothesis is supported by the trace. Return JSON {\"accept\": bool, \"reason\": str}.\n"
                        + _json_text({"proposal": payload, "analysis": analysis}), stage=f"critic-{n}-{attempt}")
                    if critic.get("accept") is True:
                        break
                    state = None
                    instruction += "\nRepair critic objections: " + _json_text(critic)
                except (ValueError, SyntaxError) as exc:
                    state = None
                    instruction += "\nRepair validation error: " + str(exc)
            label = f"candidate-{n}"
            if state is None:
                candidates.append(Candidate(label, [], gate_failure="critic_reject"))
                continue
            ev, _ = measure(state, label)
            if ev.missing > self.cfg.invalid_missing_frac * ev.n_expected:
                candidates.append(Candidate(label, payload["edits"], gate_failure="eval_invalid"))
                continue
            states[label] = state
            candidates.append(Candidate(label, payload["edits"], ev=ev))
        counts = history.incumbent_component_counts()
        winner, decisions = select_round(candidates, baseline, best, self.delta, self.cfg, counts)
        for candidate, decision in zip(candidates, decisions):
            accepted = winner is not None and candidate.variant == winner.variant
            history.append_candidate(round_index, candidate.variant, candidate.edits,
                "ACCEPTED" if accepted else ("LOST" if decision.admissible else "REJECTED"),
                decision.delta_S, decision.delta_C, accepted, decision.S, decision.C,
                diff=None, detail=decision.reason)
        state = states[winner.variant] if winner else incumbent
        harnesses = list(frontier)
        if incumbent["harness"] not in harnesses:
            harnesses.append(incumbent["harness"])
        if winner and state["harness"] not in harnesses:
            harnesses.append(state["harness"])
        state["frontier"] = {"best_score": max(best, winner.ev.S if winner else baseline.S),
                             "round": round_index + 1, "harnesses": harnesses}
        state["edit_history"] = history.records()
        # Round/calibration/frontier progress must survive rejected proposals.
        self._state = deepcopy(state)
        self.finish_refinement(state, accepted=winner is not None,
            detail={"delta": self.delta, "edit_budget": self.edit_budget,
                    "incumbent": baseline.to_json(), "S_star": best,
                    "decisions": [d.to_json() for d in decisions]})



__all__ = ["RRSIAdapter"]
