"""Game port of Recuris's Skill Memory runtime and native admission gates."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from benchmark.backend import OpenAIBackend
from benchmark.contracts import TokenUsage
from benchmark.method_support import _json_text, _mapping_value, feedback_evidence
from benchmark.native_support import NativeAdapter, native_package, valid_score, write_files


def _write_memory_package(root, files):
    package = write_files(root, files)
    (package / "em").mkdir(parents=True, exist_ok=True)
    return package


class RecurisAdapter(NativeAdapter):
    method_name = "recuris"
    state_filename = "skill_memory.json"
    allowed_options = frozenset({"max_entries_per_component", "validation_trials", "trials",
                                 "alpha", "regression_cap", "bootstrap_samples", "material"})

    def __init__(self, context):
        super().__init__(context)
        native_package("recuris", Path(__file__).parent / "src" / "recuris")
        self.validation_trials = int(self.options.get("validation_trials", 5))
        self.trials = int(self.options.get("trials", 2))
        if min(self.validation_trials, self.trials, int(self.options.get("bootstrap_samples", 3000))) < 1:
            raise ValueError("Recuris validation/trial/bootstrap counts must be positive")
        if not 0 < float(self.options.get("alpha", .05)) < 1:
            raise ValueError("Recuris alpha must be between zero and one")

    def initial_state(self):
        manifest = yaml.safe_load((Path(__file__).parent / "skill_memories" / "_base" / "manifest.yaml").read_text())
        manifest["name"] = "benchmark-games"
        manifest["delivery"].extend([{"use": "standing_inject"}, {"use": "exemplar_bounce"}])
        manifest["checkers"] = [{"use": "truth_protocol"}]
        return {"files": {"manifest.yaml": yaml.safe_dump(manifest, sort_keys=False)},
                "revision_log": [], "prescription_ledger": []}

    def acting_instructions(self):
        return ("Use verified working memory and retrieved experience. Submit the requested "
                "game action or answer through game_action when that tool is offered. "
                "The action is executed by the environment only after native pre-execution review.")

    def _agent(self, state):
        if "files" not in state:
            raise RuntimeError("legacy prompt-only Recuris checkpoint must be restarted")
        return _MemoryAgent(self, state["files"])

    def refinement_instructions(self, evidence):
        return """You are Recuris's meta-agent. Diagnose the smallest memory-owned cause
with a real action opportunity in the trace. Patch one E/W/RHO/C component only:
E = em/*.md cards; W = manifest wm/grounding/board; RHO = manifest delivery;
C = manifest checkers. Preserve all other components. The package is real:
manifest selects native builtins; cards need YAML frontmatter id/type/trigger/source.
Available delivery: standing_inject (turn_start), need_driven_retrieval (intent_recorded),
exemplar_bounce (pre_write on game_action). Checker: truth_protocol (draft_ready).
The runtime grounds real memory delivery receipts, not game success claims. Do not
configure service-request receipt grounding or invent environment tool receipts.
Do not include task answers. Abstain if evidence is not memory-owned.
Return JSON {"state": {"files": {"manifest.yaml": "...", "em/name.md": "..."},
"revision_log": [...]}, "component": "E|W|RHO|C", "cluster_id": "causal failure id",
"primitive": "specific fix", "carrier": "runtime counter that must fire",
"action_opportunity": "present", "patchability": "memory", "diagnosis": "trace evidence"}.
Carriers: ledger_update_ok, deliver_inject, retrieval_delivered, write_review_bounce,
truth_bounce, checker_bounce. Reply {"abstain": true, "reason": "..."} when unsupported."""

    def normalize_candidate(self, payload, *, current):
        from recuris.runtime import TurnRuntime
        state = super().normalize_candidate(payload, current=current)
        files = state.get("files")
        if isinstance(files, str):
            files = _mapping_value(files, "Recuris files")
            state["files"] = files
        if not isinstance(files, dict) or "manifest.yaml" not in files:
            raise ValueError("Recuris requires a manifest.yaml Skill Memory package")
        files = deepcopy(files)
        if len(files) > int(self.options.get("max_entries_per_component", 100)) + 1:
            raise ValueError("Recuris experiential memory exceeds the configured limit")
        for name, text in files.items():
            if name != "manifest.yaml" and not (name.startswith("em/") and name.endswith(".md")):
                raise ValueError("Recuris automated patches support manifest and EM markdown only")
            if not isinstance(text, str):
                raise ValueError("Recuris package files must contain text")
        before = yaml.safe_load(current["files"]["manifest.yaml"])
        after = yaml.safe_load(files["manifest.yaml"])
        if isinstance(after, str):
            after = _mapping_value(after, "Recuris manifest")
        if not isinstance(after, dict):
            raise ValueError("Recuris manifest must be a mapping")
        defaults = {"wm": {}, "grounding": {}, "board": {}}
        for key, default in defaults.items():
            value = after.get(key)
            if value is None:
                after[key] = dict(default)
            elif isinstance(value, str):
                try:
                    after[key] = _mapping_value(value, f"Recuris manifest.{key}")
                except ValueError:
                    aliases = ({"generic_item", "knowledge_need"} if key == "wm" else
                               {"delivery_receipt"} if key == "grounding" else set())
                    if value in aliases:
                        after[key] = {"entry_kind" if key == "wm" else "matcher": value}
                    else:
                        raise
            elif not isinstance(value, Mapping):
                raise ValueError(f"Recuris manifest.{key} must be an object")
            else:
                after[key] = dict(value)
        for key in ("delivery", "checkers"):
            value = after.get(key, [])
            if isinstance(value, str):
                value = _mapping_value(value, f"Recuris manifest.{key}").get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"Recuris manifest.{key} must be an array")
            after[key] = [{"use": item} if isinstance(item, str) else item for item in value]
        self._normalize_memory_triggers(files)
        files["manifest.yaml"] = yaml.safe_dump(after, sort_keys=False)
        changed = set()
        groups = {"wm": "W", "grounding": "W", "board": "W", "delivery": "RHO",
                  "checkers": "C", "gate": "C"}
        for key in set(before) | set(after):
            if before.get(key) != after.get(key):
                if key == "gate":
                    raise ValueError("game port has no customer-exit terminal gate; gate edits cannot execute")
                if key not in groups:
                    raise ValueError(f"unsupported Recuris manifest edit: {key}")
                changed.add(groups[key])
        if {k: v for k, v in files.items() if k != "manifest.yaml"} != {
                k: v for k, v in current["files"].items() if k != "manifest.yaml"}:
            changed.add("E")
        if changed != {payload.get("component")}:
            raise ValueError("Recuris diagnosis must match exactly one changed component")
        if payload.get("action_opportunity") != "present" or payload.get("patchability") != "memory":
            raise ValueError("Recuris patch needs a demonstrated memory-owned opportunity")
        carriers = {"E": {"deliver_inject", "retrieval_delivered", "write_review_bounce"},
                    "W": {"ledger_update_ok"},
                    "RHO": {"deliver_inject", "retrieval_delivered", "write_review_bounce"},
                    "C": {"truth_bounce", "checker_bounce"}}
        if payload.get("carrier") not in carriers[payload["component"]]:
            raise ValueError("Recuris prescribed carrier is incompatible with the component")
        if after["grounding"].get("matcher", "delivery_receipt") != "delivery_receipt":
            raise ValueError("game port supports real memory-delivery grounding only")
        if after["wm"].get("entry_kind", "generic_item") not in {"generic_item", "knowledge_need"}:
            raise ValueError("game port requires a knowledge working-memory entry kind")
        if "local:" in files["manifest.yaml"]:
            raise ValueError("local strategy code requires a separately reviewed native integration")
        digest = hashlib.sha256(_json_text(files).encode()).hexdigest()[:16]
        TurnRuntime.load(str(_write_memory_package(self.state_dir / "packages" / digest, files)), {"game_action"})
        state["files"] = files
        state["prescription_ledger"] = deepcopy(current.get("prescription_ledger", []))
        return state

    @staticmethod
    def _normalize_memory_triggers(files):
        """Complete scalar trigger shorthand into the native event/tool schema."""
        for name, body in list(files.items()):
            if not name.startswith("em/") or not name.endswith(".md"):
                continue
            if not body.startswith("---"):
                raise ValueError(f"Recuris memory card {name} needs YAML frontmatter")
            marker = body.find("\n---", 3)
            if marker < 0:
                raise ValueError(f"Recuris memory card {name} has unterminated frontmatter")
            front = yaml.safe_load(body[3:marker]) or {}
            if not isinstance(front, Mapping):
                front = _mapping_value(front, f"{name} frontmatter")
            trigger = front.get("trigger")
            if isinstance(trigger, str):
                value = trigger.strip()
                if value in {"turn_start", "intent_recorded", "draft_ready", "pre_write"}:
                    front["trigger"] = {"event": value, "tool": "*"}
                elif value == "game_action":
                    front["trigger"] = {"event": "pre_write", "tool": value}
                elif value:
                    front["trigger"] = {"event": "turn_start", "tool": "*"}
                else:
                    raise ValueError(f"Recuris memory card {name} has an empty trigger")
            elif trigger is None:
                front["trigger"] = {"event": "turn_start", "tool": "*"}
            elif not isinstance(trigger, Mapping):
                front["trigger"] = _mapping_value(trigger, f"{name} trigger")
            files[name] = "---\n" + yaml.safe_dump(dict(front), sort_keys=False).rstrip() + body[marker:]

    def refine(self, feedback, *, output_dir):
        from recuris.metaagent.gates import held_out_paired_gate, fingerprint_verify, Ledger
        self.start_refinement(feedback, output_dir)
        evidence = feedback_evidence(feedback)
        prompt = self.refinement_instructions([]) + "\nCurrent package:\n" + _json_text(self._state)
        prompt += "\nTraining evidence:\n" + _json_text(evidence)
        payload = None
        candidate = None
        error = None
        for attempt in range(2):
            try:
                if attempt:
                    prompt += "\nRepair the previous JSON/package validation error. The trigger field must be an object with event and tool keys.\n"
                    prompt += "Previous response:\n" + _json_text(payload) + "\nValidation error:\n" + str(error)
                payload = self.ask(prompt, stage="diagnose-patch" if not attempt else "repair-patch")
                if payload.get("abstain"):
                    self.finish_refinement(self._state, accepted=False, detail={"abstention": payload})
                    return
                candidate = self.normalize_candidate(payload, current=self._state)
                break
            except (ValueError, TypeError, AttributeError, KeyError, yaml.YAMLError) as exc:
                error = exc
        if candidate is None:
            self.finish_refinement(self._state, accepted=False,
                detail={"candidate_rejected": str(error), "repair_attempts": 2})
            return
        prescription = SimpleNamespace(cluster_id=str(payload.get("cluster_id", "")),
            carrier=SimpleNamespace(value=payload["carrier"]), primitive=str(payload.get("primitive", "")))
        if not prescription.cluster_id or not prescription.primitive:
            raise ValueError("Recuris prescription requires cluster_id and primitive")
        key = list(Ledger.key(prescription))
        if any(row["key"] == key for row in self._state.get("prescription_ledger", [])):
            self.finish_refinement(self._state, accepted=False, detail={"repeat_prescription": key})
            return
        base, cand, counters = {}, {}, Counter()
        for i in range(self.validation_trials):
            base[str(i)], cand[str(i)] = [], []
            for trial in range(self.trials):
                a = self.rollout(self._state, label=f"base-{i}-{trial}", split="validation", index=i)
                b = self.rollout(candidate, label=f"candidate-{i}-{trial}", split="validation", index=i)
                base[str(i)].append(valid_score(a))
                cand[str(i)].append(valid_score(b))
                counters.update(b.get("mechanism_fingerprint", {}))
        activated = fingerprint_verify(SimpleNamespace(fingerprint=counters), prescription)
        verdict = held_out_paired_gate(base, cand, alpha=float(self.options.get("alpha", .05)),
            reg_cap=int(self.options.get("regression_cap", 0)),
            n_boot=int(self.options.get("bootstrap_samples", 3000)), seed=self._method_context.seed,
            material=float(self.options.get("material", 0)))
        accepted = verdict.accept and activated
        result = candidate if accepted else deepcopy(self._state)
        result.setdefault("prescription_ledger", []).append({"key": key, "accepted": accepted,
            "reason": verdict.reason if activated else "prescribed mechanism never activated"})
        self._state = result
        self.finish_refinement(result, accepted=accepted,
            detail={"native_gate": verdict, "activated": activated, "fingerprint": dict(counters)})


class _MemoryAgent:
    """HarnessPort: native runtime owns ledger, retrieval, checks and bounces."""
    def __init__(self, owner, files):
        from recuris.runtime import TurnRuntime
        self.owner = owner
        digest = hashlib.sha256(_json_text(files).encode()).hexdigest()[:16]
        self.runtime = TurnRuntime.load(str(_write_memory_package(owner.state_dir / "packages" / digest, files)), {"game_action"})
        self.state = self.runtime.new_state(digest)
        self.messages, self.users, self.responses = [], [], []
        self.wm = ""

    @property
    def fingerprint(self): return dict(self.state.fp.counters)

    def act(self, prompt, *, image_paths=()):
        self.users.append(prompt)
        self.messages.append({"role": "user", "content": OpenAIBackend._build_chat_messages(
            prompt, tuple(Path(p) for p in image_paths))[0]["content"]})
        self.responses = []
        draft = self.runtime.run_turn(self.state, self, prompt)
        self.messages[-1] = {"role": "assistant", "content": draft.response.text}
        fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")
        usage = {}
        for field in fields:
            values = [getattr(r.usage, field, None) for r in self.responses]
            usage[field] = sum(values) if all(isinstance(v, int) for v in values) else None
        return replace(draft.response, usage=TokenUsage(**usage),
            metadata={**dict(draft.response.metadata or {}), "recuris_fingerprint": self.fingerprint,
                      "recuris_events": list(self.state.fp.events)})

    def _call(self, instruction, *, drafting=False):
        tools = [{"type": "function", "function": {
            "name": "game_action", "description": "Submit the exact game action/answer for pre-execution review.",
            "parameters": {"type": "object", "properties": {"action": {"type": "string"}},
                           "required": ["action"], "additionalProperties": False}}}] if drafting else None
        response = self.owner.dispatcher.distribute("openai", "", messages=[
            {"role": "system", "content": self.owner.acting_instructions()}, *self.messages,
            {"role": "user", "content": instruction}], tools=tools, model=self.owner.actor_model,
            api_mode=self.owner.api_mode, max_output_tokens=self.owner.actor_max_output_tokens,
            reasoning_effort=self.owner.actor_reasoning_effort, temperature=self.owner.temperature)
        self.responses.append(response)
        return response

    def _draft(self, instruction):
        response = self._call(instruction, drafting=True)
        if response.tool_calls:
            if len(response.tool_calls) != 1 or response.tool_calls[0].name != "game_action":
                raise ValueError("Recuris game port requires exactly one game_action call")
            args = response.tool_calls[0].arguments
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict) or not isinstance(args.get("action"), str):
                raise ValueError("game_action requires an action string")
            response = replace(response, text=args["action"])
        return SimpleNamespace(response=response)

    def incoming_kind(self, incoming): return "user"
    def user_text(self, incoming): return incoming
    def genuine_user_texts(self): return list(self.users)
    def extract_receipts(self): return []  # Never fabricate game execution receipts.
    def read_tool_results(self): return {}
    def ledger_llm(self, instruction): return self._call(instruction).text
    def llm_draft(self, wm_text):
        board = self.runtime.board_text(self.state)
        if board:
            wm_text += "\n" + board
        self.wm = wm_text
        return self._draft(wm_text + "\nSubmit the requested action with game_action.")
    def redraft_with_correction(self, wm_text, correction):
        return self._draft(wm_text + "\n" + correction)
    def bounce_and_redraft(self, draft, results_by_call):
        return self._draft(self.wm + "\nRejected action: " + draft.response.text
                           + "\nPre-execution review:\n" + _json_text(results_by_call))
    def commit(self, draft): self.messages.append({"role": "assistant", "content": draft.response.text})
    def draft_content(self, draft): return draft.response.text
    def draft_tool_calls(self, draft):
        return draft.response.tool_calls
    def set_draft_content(self, draft, text): draft.response = replace(draft.response, text=text)


__all__ = ["RecurisAdapter"]
