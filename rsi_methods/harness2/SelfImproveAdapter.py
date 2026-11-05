"""Game-domain port of the native Harness² recursion and blind selector."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from benchmark.artifact_agent import ArtifactAgent
from benchmark.method_support import _json_text, _parse_json_object
from benchmark.native_support import NativeAdapter, write_files
from benchmark.resume_state import atomic_json

from .harness2.core import recursion, selection, tree
from .harness2.core.library import Library
from .harness2.core.proposer import Reply, ProposerError
from .harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec, RunView
from .harness2.substrate.opencode import OPENCODE_SURFACE


class Harness2Adapter(NativeAdapter):
    method_name = "harness2"
    state_filename = "harness.json"
    allowed_options = frozenset({"recursion_mode", "proposal_count", "workers"})

    def __init__(self, context):
        super().__init__(context)
        self.mode = {"parallel": "agg", "sequential": "seq"}.get(
            self.options.get("recursion_mode", "sequential"))
        self.k = int(self.options.get("proposal_count", 1))
        self.workers = int(self.options.get("workers", 1))
        if self.mode is None or min(self.k, self.workers) < 1:
            raise ValueError("invalid Harness² recursion_mode/proposal_count/workers")

    def initial_state(self):
        return {"files": {"systemprompt.md": "Solve carefully; return exactly the requested action.",
                          "guardrails.md": "Verify prerequisites before acting.",
                          "LongTermMEMORY.md": "", "tool_descriptions/tool_guidance.md": ""},
                "library": {}}

    def acting_instructions(self):
        return "Apply the active harness. Use the artifact tools when helpful."

    def _agent(self, state):
        files = state.get("files")
        if files is None:
            raise RuntimeError("legacy prompt-only Harness² checkpoint must be restarted")
        digest = hashlib.sha256(_json_text(files).encode()).hexdigest()[:16]
        return ArtifactAgent(self, files, self.acting_instructions(),
            always_on=OPENCODE_SURFACE.always_on, root=self.state_dir / "agents" / digest)

    def refine(self, feedback, *, output_dir):
        self.start_refinement(feedback, output_dir)
        port = _GamePort(self, feedback, output_dir)
        prop = _APIProposer(self)
        base_tree = dict(self._state["files"])
        port.validate_tree(base_tree)
        library_root = write_files(output_dir / "library", self._state.get("library", {}))
        library = Library(library_root, port.domain_of(port.task_id), OPENCODE_SURFACE)
        base = []
        for i in range(self.k):
            label = f"base{i + 1}"
            run_dir = port.execute(port.task_id, port.base_harness(), label)
            base.append(RunView(label=label, run_id=label, run_dir=run_dir,
                                ok=True, delivered=port.delivered(run_dir)))
        result = recursion.run_mode(self.mode, port.task_id, self.k, base,
            adapter=port, prop=prop, work_dir=output_dir, run_id_of=lambda label: label,
            base_tree=base_tree, library_dir=library.context_dir(),
            propose_workers=self.workers, rollout_workers=self.workers)
        atomic_json(output_dir / "native_recursion.json", result.to_dict())
        pool = selection.selection_pool(self.mode, self.k, result.rollouts, base)
        if self.k == 1:
            chosen = selection.trivial_selection(port.task_id, self.k, self.mode, pool,
                                                 adapter=port, work_dir=output_dir)
        else:
            chosen = selection.run_selection(port.task_id, self.k, self.mode, pool,
                adapter=port, prop=prop, work_dir=output_dir, pool_kind="finals")
        state = dict(self._state)
        if chosen.ok:
            state["files"] = result.trees[chosen.chosen_label]
            experience = selection.reflect_adoption(port.task_id, chosen, pool,
                adapter=port, prop=prop, work_dir=output_dir,
                decisions={s.label: s.notes for s in result.stages})
            library.add(port.task_id, state["files"], why=chosen.why,
                        label=chosen.chosen_label, run_id=chosen.chosen_run_id,
                        experience=experience)
            state["library"] = {p.relative_to(library_root).as_posix(): p.read_text()
                                for p in library_root.rglob("*") if p.is_file()}
        self.finish_refinement(state, accepted=chosen.ok,
            detail={"native_mode": self.mode, "selection": chosen.to_dict(),
                    "recursion_ok": result.ok, "errors": result.errors})


class _APIProposer:
    """Keep native prompts/context; adapt only the CLI file-edit transport."""
    def __init__(self, owner):
        self.owner = owner

    def call(self, mission, context_dir, label, *, harvest=None, **kwargs):
        context_dir = Path(context_dir)
        files = {p.relative_to(context_dir).as_posix(): p.read_text(encoding="utf-8")
                 for p in sorted(context_dir.rglob("*")) if p.is_file()}
        instruction = ("\nReturn JSON {\"files\": {relative_path: full_text}, \"reasoning\": \"...\"}. "
                       "files must be the complete edited harness, with paths relative to context/harness/."
                       if harvest else "\nFollow the requested fenced response format exactly.")
        tries = max(1, int(kwargs.get("max_attempts", 1)))
        for attempt in range(tries):
            try:
                response = self.owner.ask_text(str(kwargs.get("system", "")) + "\n" + mission
                    + instruction + "\nContext files:\n" + _json_text(files), stage=label)
                break
            except Exception as exc:
                if attempt + 1 == tries:
                    raise ProposerError(f"API proposer failed: {exc}", attempts=tries) from exc
        workspace = None
        if harvest:
            try:
                payload = _parse_json_object(response)
                workspace = write_files(self.owner._work_dir / "api_proposals" / label, payload["files"])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                # A malformed edited artifact uses the native stage's repair
                # attempt; reserve ProposerError for failed model sessions.
                return Reply(label=label, text=response, ok=False, error=str(exc),
                             attempts=attempt + 1, workspace=None)
        return Reply(label=label, text=response, ok=True, attempts=attempt + 1, workspace=workspace)


class _GamePort:
    def __init__(self, owner, feedback, work):
        self.owner, self.feedback, self.work = owner, feedback, work
        self.task_id = feedback[-1].task_id
        self.results = {}
        self.substrate = SimpleNamespace(surface=lambda: OPENCODE_SURFACE)
        channel = ChannelSpec(kind="claude_cli", model=owner.optimizer_model)
        self.spec = BenchSpec(name="games", runs_root=work, install_mode="workspace_config",
            prompts=BenchPrompts(work="Complete the game objective with valid actions.",
                yardstick="Follow the public rules and produce a coherent action sequence."),
            channels={"proposer": channel, "selector": channel}, allow_subagents=True)
        self._base = write_files(work / "base_harness", owner._state["files"])

    def tasks(self, split="all"): return [self.task_id]
    def domain_of(self, task_id):
        return str(self.feedback[-1].evaluation.metrics.get("game", "game"))
    def task_prompt(self, task_id):
        return self.feedback[-1].trajectory[0].prompt if self.feedback[-1].trajectory else "Complete the public game objective."
    def task_files(self, task_id): return None
    def base_harness(self): return self._base
    def execute(self, task_id, harness_dir, run_id, **kwargs):
        files = tree.read_tree(OPENCODE_SURFACE, harness_dir or self._base)
        result = self.owner.rollout({"files": files}, label=run_id)
        run_dir = self.work / "public_runs" / run_id
        steps = [{"prompt": step.prompt, "action": step.action,
                  "observation": step.environment_response} for step in result["feedback"].trajectory]
        write_files(run_dir, {"trajectory.md": _json_text(steps),
                             "deliverable.md": _json_text([s["action"] for s in steps])})
        self.results[str(run_dir)] = result
        return run_dir
    def delivered(self, run_dir):
        result = self.results.get(str(run_dir))
        return bool(result and result["record"].get("status") == "ok" and result["feedback"].trajectory)
    def trajectory(self, run_dir, *, obs_trunc): return (run_dir / "trajectory.md").read_text()
    def deliverable_files(self, run_dir): return {"actions.md": self.deliverable_text(run_dir)}
    def deliverable_text(self, run_dir): return (run_dir / "deliverable.md").read_text()
    def score(self, run_dir, task_id): return None
    def validate_tree(self, files):
        tree.validate(OPENCODE_SURFACE, files)
        if any(p.startswith("plugin/") and p.endswith(".ts") for p in files):
            raise ValueError("The API game runtime cannot execute OpenCode TypeScript plugins; use executable Python helpers or subagents.")


__all__ = ["Harness2Adapter"]
