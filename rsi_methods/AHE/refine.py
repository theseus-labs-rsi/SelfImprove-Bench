"""Extract AHE candidate generation from its evaluate/analyze/improve loop.

This module uses benchmark feedback as evidence and runs the upstream Evolve
Agent on independent copies of the committed harness.  It deliberately does
not evaluate or rank candidates; the benchmark and ``select.py`` own those
steps after this function returns a CandidateBatch.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from benchmark.contracts import to_jsonable
from benchmark.token_ledger import nexau_usage_snapshot

from .agent import AHEAgent, configure_reasoning_effort
from .build import build
from .init import AHEState, apply_agent_limits
from .history import load_history, save_history
from .sandbox import reject_symlinks, sandbox_command, training_history_paths


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One unevaluated harness variant exposed to the benchmark runner."""

    candidate_id: str
    workspace: Path
    agent: AHEAgent = field(repr=False)

    @property
    def artifact_dir(self) -> Path:
        return self.agent.artifact_dir


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """Variants generated from one parent; no winner is chosen here."""

    refinement_id: str
    work_dir: Path
    parent_workspace: Path
    parent_score: float
    candidates: tuple[CandidateSpec, ...]
    refinement_usage: Mapping[str, Any] = field(default_factory=dict)


def select_parent(state: AHEState, agent: AHEAgent) -> tuple[Path, dict[str, Any] | None]:
    """Resolve the committed parent and optional historical-best evidence.

    Upstream AHE records ``best_ever.json`` for the Evolve Agent, while its
    normal iteration still evolves the active workspace.  Its explicit
    rollback is used on resume.  Therefore a historical snapshot becomes
    the parent only after a state transition has selected or restored it;
    reading best-ever metadata alone must not switch the benchmark's parent.
    """

    parent = state.current_workspace.resolve()
    if agent.workspace != parent or not (parent / "code_agent.yaml").is_file():
        raise RuntimeError("AHE parent Agent does not match the committed workspace")
    best_path = state.run_dir / "best_ever.json"
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else None
    return parent, best


def _read_feedback(feedback_results: Sequence[Any]) -> tuple[dict[str, Any], float]:
    """Use the benchmark's numeric score without inventing a pass threshold."""

    if len(feedback_results) != 1:
        raise ValueError("AHE currently expects one benchmark task per refinement")
    payload = to_jsonable(feedback_results[0])
    if not isinstance(payload, dict) or not isinstance(payload.get("evaluation"), dict):
        raise TypeError("AHE refine requires one FeedbackResult-like record")
    score = payload["evaluation"].get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("AHE refine requires a finite benchmark score")
    if not str(payload.get("task_id") or "").strip():
        raise ValueError("AHE refine requires a task_id")
    return payload, float(score)


def _trace_from_feedback(feedback: dict[str, Any], parent_agent: AHEAgent) -> Path:
    """Use the evaluated task's trace, including after its Agent was closed."""
    for step in reversed(feedback.get("trajectory") or []):
        call = step.get("agent_call") if isinstance(step, dict) else None
        metadata = call.get("metadata") if isinstance(call, dict) else None
        trace = metadata.get("ahe_trace_path") if isinstance(metadata, dict) else None
        if isinstance(trace, str) and trace:
            path = Path(trace)
            if not path.is_file():
                raise FileNotFoundError(f"AHE feedback trace is missing: {path}")
            return path
    if parent_agent.trace_paths:
        path = parent_agent.trace_paths[-1]
        if path.is_file():
            return path
        raise FileNotFoundError(f"AHE parent Agent trace is missing: {path}")
    raise RuntimeError("AHE refine needs a trace from the evaluated Agent")


def _safe_task_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._")[:80] or "task"


def _prepare_evidence(
    parent: Path,
    feedback: dict[str, Any],
    score: float,
    trace_path: Path,
    iteration_dir: Path,
    trace_paths: Sequence[Path] = (),
) -> tuple[Path, str]:
    """Materialize the parent snapshot and trace where AHE's prompt expects them.

    The original layout is ``runs/iteration_NNN/input/{workspace,benchmark,
    analysis}``.  ``reward.txt`` contains the actual continuous score here;
    the generated analysis explicitly avoids calling it a binary pass/fail.
    """

    input_dir = iteration_dir / "input"
    shutil.copytree(parent, input_dir / "workspace", ignore=shutil.ignore_patterns(".git"))
    name = _safe_task_name(str(feedback["task_id"]))
    trial_dir = input_dir / "benchmark" / "local" / f"{name}__{uuid4().hex[:8]}"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    shutil.copy2(trace_path, trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json")
    # Relocatable evidence, independent of the tracer's image serialization.
    from benchmark.visual_evidence import bundle_feedback
    bundle_feedback(feedback, trial_dir)
    _write_task_trace(trial_dir, trace_path)
    bundled_traces = []
    for index, original in enumerate(trace_paths or (trace_path,)):
        target = trial_dir / "agent" / "steps" / f"{index:04d}"
        target.mkdir(parents=True)
        shutil.copy2(original, target / original.name)
        request_path = original.parent / "request.json"
        if request_path.is_file():
            request = json.loads(request_path.read_text())
            copied_images = []
            from benchmark.visual_evidence import image_file
            for image_index, value in enumerate(request.get("image_paths") or ()):
                source = image_file(value)
                image_target = target / f"input_{image_index}{source.suffix.lower()}"
                shutil.copy2(source, image_target)
                copied_images.append(image_target.relative_to(trial_dir).as_posix())
            request["image_paths"] = copied_images
            request["image_path_base"] = "trial_dir (directory containing visual_manifest.json)"
            (target / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2))
        bundled_traces.append((target / original.name).relative_to(trial_dir).as_posix())
    (trial_dir / "agent" / "all_step_traces.json").write_text(
        json.dumps({"path_base": "trial_dir", "paths": bundled_traces}, indent=2), encoding="utf-8",
    )
    (trial_dir / "verifier" / "reward.txt").write_text(str(score), encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps({"task_id": feedback["task_id"], "score": score}, ensure_ascii=False),
        encoding="utf-8",
    )
    return trial_dir, name


def _write_task_trace(trial_dir: Path, trace_path: Path) -> None:
    """Rebuild the whole-task conversation, as the text version's trace shows it.

    The game calls ``agent.run`` once per step, and the NexAU converter keeps
    only the first run's input with blank assistant text.  The benchmark
    trajectory has every step's prompt, pre-action screenshot and raw reply,
    so the cleaned trace is rewritten from it in the same schema.
    """

    trajectory = json.loads((trial_dir / "feedback.json").read_text()).get("trajectory") or []
    if not trajectory:
        return
    target = trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json"
    shutil.copy2(target, target.with_name("nexau_native_trace.json"))
    try:
        trace = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        trace = {}
    if not isinstance(trace, dict):
        trace = {}
    system = [m for m in trace.get("messages") or () if isinstance(m, dict) and m.get("role") == "system"][:1]
    messages = list(system)
    for index, step in enumerate(trajectory):
        step_index = step.get("step_index", index)
        screenshots = "".join(
            f"\n[screenshot: {image['path']} (relative to trial_dir) - observation before step {step_index}'s action]"
            for image in step.get("image_records") or ()
        )
        reply = (step.get("agent_call") or {}).get("response_text") or step.get("action") or ""
        messages.append({"role": "user", "step_index": step_index,
                         "content": f"{step.get('prompt') or ''}{screenshots}"})
        messages.append({"role": "assistant", "step_index": step_index, "content": reply})
    trace.update(
        input={"message": messages[len(system)]["content"]},
        output={"response": messages[-1]["content"]},
        user_message=messages[len(system)]["content"],
        messages=messages,
        messages_count=len(messages),
        reconstruction=(
            "Whole-task trajectory rebuilt from feedback.json: one user prompt with its "
            "screenshot reference and one agent reply per game step. During play the "
            "agent's own context kept only the most recent turns."
        ),
        native_evidence={
            "last_step": "agent/nexau_native_trace.json",
            "all_steps": "agent/all_step_traces.json",
            "path_base": "trial_dir",
        },
    )
    target.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_analysis(
    iteration_dir: Path,
    trial_dir: Path,
    task_name: str,
    feedback: dict[str, Any],
    score: float,
    *,
    model: str | None = None,
    llm_environment: Mapping[str, str] | None = None,
) -> str:
    """Require a successful Debugger diagnosis before candidate generation."""

    trace_path = trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json"
    analysis_dir = iteration_dir / "input" / "analysis"
    detail_dir = analysis_dir / "detail"
    detail_dir.mkdir(parents=True)
    evaluator_note = str(feedback["evaluation"].get("feedback") or "")
    diagnosis = _ask_debugger(
        trace_path, score, evaluator_note, model=model, llm_environment=llm_environment,
        output_dir=analysis_dir,
    )
    relative_trace = trace_path.relative_to(iteration_dir.parent.parent).as_posix()
    overview = (
        f"# Benchmark evidence for {task_name}\n\n"
        f"- Relative score: {score}\n"
        f"- Execution status: {feedback['evaluation'].get('execution_status')}\n"
        f"- Trace (complete task trajectory: one prompt with screenshot reference and one agent "
        f"reply per game step; during play the agent's context kept the 8 most recent turns): `{relative_trace}`\n"
        f"- Per-step actions, game state and progress: `{trial_dir.relative_to(iteration_dir.parent.parent).as_posix()}/feedback.json`\n"
        f"- Every screenshot (pre-action observations) with step index and sha256: "
        f"`{trial_dir.relative_to(iteration_dir.parent.parent).as_posix()}/visual_manifest.json`\n"
        f"- All original per-step NexAU traces and requests: `{relative_trace.rsplit('/', 1)[0]}/all_step_traces.json`\n"
        f"- Detailed analysis: `runs/{iteration_dir.name}/input/analysis/detail/{task_name}.md`\n\n"
        "The score is continuous; no binary pass/fail label is inferred from execution status.\n"
    )
    detail = f"# {task_name}\n\nScore: {score}\n\nEvaluator feedback: {evaluator_note}\n\nTrace: `{relative_trace}`\n\n{diagnosis}\n"
    (analysis_dir / "overview.md").write_text(overview, encoding="utf-8")
    (detail_dir / f"{task_name}.md").write_text(detail, encoding="utf-8")
    return overview


def _ask_debugger(
    trace_path: Path, score: float, evaluator_note: str, *,
    model: str | None = None,
    llm_environment: Mapping[str, str] | None = None,
    output_dir: Path | None = None,
) -> str:
    """Run the bundled Debugger in this interpreter; never skip diagnosis."""

    upstream = Path(__file__).resolve().parent / "agentic-harness-engineering"
    source = upstream / "agents/evolve_agent/skills/agent-debugger-cli/_source"
    if not (source / "agent_debugger_core/cli/adb.py").is_file():
        raise RuntimeError("AHE Agent Debugger source is required; refinement stopped")
    env = os.environ.copy()
    env.update(llm_environment or {})
    if model:
        env["LLM_MODEL"] = model
    for target, fallback in (
        ("LLM_BASE_URL", "OPENAI_BASE_URL"), ("LLM_API_KEY", "OPENAI_API_KEY"),
    ):
        if not env.get(target) and env.get(fallback):
            env[target] = env[fallback]
    env["AHE_HOME"] = str(upstream)
    env["AHE_DEBUGGER_SCOPED_LLM"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(source), str(upstream)))
    question = (
        f"Analyze this agent trace against benchmark score {score}. "
        f"Evaluator feedback: {evaluator_note}. Identify evidence-backed causes; "
        "do not infer task success from process completion. Read the trial's "
        "feedback.json and agent/all_step_traces.json for native tool calls and "
        "runtime evidence; the cleaned conversation is a benchmark reconstruction. "
        "Identify the earliest supported failure, relevant harness component, "
        "concrete evidence and an actionable correction. Separate observed facts "
        "from hypotheses; do not invent missing runtime logs."
    )
    try:
        command = sandbox_command(
            [sys.executable, "-m", "agent_debugger_core.cli.adb", "ask",
             "-t", str(trace_path.resolve()), "-q", question, "--format", "json"],
            readable=[trace_path.parent.parent], cwd=trace_path.parent.parent,
            python_paths=[source, upstream],
        )
        completed = subprocess.run(
            command,
            env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("AHE Agent Debugger timed out; refinement stopped") from exc
    except OSError as exc:
        raise RuntimeError("AHE Agent Debugger could not start; refinement stopped") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "AHE Agent Debugger failed; refinement stopped. "
            f"Check Debugger dependencies and LLM configuration. "
            f"Details: {(completed.stderr or completed.stdout)[-1000:]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AHE Agent Debugger returned invalid JSON; refinement stopped") from exc
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError("AHE Agent Debugger did not report success; refinement stopped")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "debugger_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (output_dir / "debugger_usage_snapshot.json").write_text(
            json.dumps(result.get("usage_snapshot"), ensure_ascii=False, indent=2), encoding="utf-8",
        )
    diagnosis = result.get("response") or result.get("answer")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        raise RuntimeError("AHE Agent Debugger returned no diagnosis; refinement stopped")
    if diagnosis.lstrip().startswith("[budget-exceeded]"):
        raise RuntimeError("AHE Agent Debugger exhausted its budget; refinement stopped")
    return diagnosis.strip()


def _prepare_candidate(parent: Path, candidate_root: Path) -> Path:
    """Copy the parent and give this variant its own Git history for AHE edits."""

    reject_symlinks(parent)
    workspace = candidate_root / "workspace"
    shutil.copytree(parent, workspace, ignore=shutil.ignore_patterns(".git"))
    for command in (
        ["git", "init", "--quiet"],
        ["git", "add", "-A"],
        ["git", "-c", "user.name=AHE Adapter", "-c", "user.email=ahe-adapter@local.invalid",
         "commit", "--quiet", "-m", "parent snapshot"],
    ):
        subprocess.run(command, cwd=workspace, check=True, capture_output=True, text=True)
    return workspace


def _evolution_query(
    state: AHEState,
    iteration_dir: Path,
    workspace: Path,
    score: float,
    best_ever: dict[str, Any] | None,
    hint: str | None,
) -> str:
    """Tell the original Evolve Agent how the external benchmark differs."""

    workspace_rel = workspace.relative_to(state.run_dir).as_posix()
    analysis_rel = f"runs/{iteration_dir.name}/input/analysis"
    lines = [
        f"Iteration {iteration_dir.name} evaluation completed.",
        f"Your writable harness is `{workspace_rel}/`.",
        f"Read `{analysis_rel}/overview.md` and its detail file first. "
        "Inspect linked native traces when the diagnosis lacks evidence for a specific question.",
        f"The benchmark's objective is relative_score; parent score is {score}.",
        "The reward file contains that continuous score. Execution completion is not a task pass.",
        "Make evidence-backed harness changes in your writable workspace only.",
        "Do not run Harbor or select a winner. The benchmark evaluates returned candidates later.",
        "Read `evolution_history.md` and linked change_evaluation.json files when present. "
        "These record same-task parent/candidate comparisons, not a fixed-dataset trend.",
        "There is no comparable global best score because each round can use a different "
        "training task. Continue the active workspace; do not rank snapshots across tasks.",
        f"Write the change manifest to "
        f"`{workspace.parent.relative_to(state.run_dir).as_posix()}/change_manifest.json`.",
    ]
    if best_ever is not None:
        lines.append(f"Historical best metadata for context: {json.dumps(best_ever, ensure_ascii=False)}")
    if hint:
        lines.append(f"Strategy constraint for this candidate: {hint}")
    return "\n\n".join(lines)


def _run_evolver(
    state: AHEState,
    candidate_root: Path,
    iteration_dir: Path,
    workspace: Path,
    query: str,
    model: str | None,
    reasoning_effort: str | None,
    timeout_seconds: float,
    llm_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the original Evolve Agent in a separate import namespace."""

    request_path = candidate_root / "evolve_request.json"
    result_path = candidate_root / "evolve_result.json"
    from benchmark.visual_evidence import visual_note
    visual_manifests = list((iteration_dir / "input" / "benchmark").rglob("visual_manifest.json"))
    if len(visual_manifests) != 1:
        raise RuntimeError("Expected exactly one current-task visual evidence manifest")
    manifest_path = visual_manifests[0]
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["model_sample"]
    image_paths = [str(manifest_path.parent / entry["path"]) for entry in selected]
    if selected:
        query += visual_note(selected)
        query += (
            f"\nAll frames and their step mapping: {manifest_path.relative_to(state.run_dir)}. "
            "Use read_visual_file on individual evidence images when more context is needed. "
            "Shell output and image file names are not image observations."
        )
    request_path.write_text(json.dumps({
        "run_dir": str(state.run_dir), "evolve_agent_dir": str(state.evolve_agent_dir),
        "workspace_rel": workspace.relative_to(state.run_dir).as_posix(),
        "iteration": int(iteration_dir.name.split("_")[-1]), "query": query,
        "reasoning_effort": reasoning_effort,
        "image_paths": image_paths,
        "visual_manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update(llm_environment or {})
    code_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(code_root), env.get("PYTHONPATH"))))
    if model:
        env["LLM_MODEL"] = model
    # Bind individual result files, not candidate_root: evaluation state is host-owned.
    worker_outputs = [result_path, candidate_root / "evolve_trace.json",
                      candidate_root / "evolve_usage_snapshot.json", candidate_root / "change_manifest.json"]
    for path in worker_outputs:
        path.touch(exist_ok=False)
    runtime_dir = candidate_root / "evolve_agent_runtime"
    config_path = _prepare_evolve_runtime(state.evolve_agent_dir, runtime_dir)
    prompt_path = config_path.parent / "evolve_prompt.md"
    manifest_rel = (candidate_root / "change_manifest.json").relative_to(state.run_dir).as_posix()
    prompt_path.write_text(prompt_path.read_text(encoding="utf-8").replace(
        "Write to experiment root directory (NOT inside workspace/).",
        f"Write to `{manifest_rel}` (NOT inside the writable harness).",
    ), encoding="utf-8")
    command = sandbox_command(
        [sys.executable, "-m", "rsi_methods.AHE.refine", "--worker", str(request_path), str(result_path)],
        readable=[request_path, runtime_dir, iteration_dir / "input", state.evolve_agent_dir,
                  *training_history_paths(state.run_dir)],
        writable=[workspace, *worker_outputs], cwd=state.run_dir,
    )
    completed = subprocess.run(
        command,
        env=env, capture_output=True, text=True, timeout=timeout_seconds,
    )
    (candidate_root / "evolve.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"AHE candidate generation failed; inspect {candidate_root / 'evolve.log'}")
    reject_symlinks(workspace)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _strategy_hints() -> list[str]:
    """Reuse the variant strategy hints declared in upstream AHE config."""

    import yaml

    config_path = Path(__file__).resolve().parent / "agentic-harness-engineering" / "configs" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return list(config.get("best_of_n", {}).get("strategy_hints", []))


def _evolve_token_report(
    candidates: list[dict[str, Any]], *, debugger_invoked: bool, source_path: Path,
    debugger_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sum provider-reported NexAU spans; do not invent missing subfield usage."""

    components = list(candidates)
    if debugger_invoked:
        components.append({"role": "debugger", "usage": debugger_usage})
    complete = bool(candidates) and all(
        isinstance(item.get("usage"), dict)
        and isinstance(item["usage"].get("total_tokens"), int)
        and not isinstance(item["usage"]["total_tokens"], bool)
        and item["usage"]["total_tokens"] >= 0
        and item["usage"].get("model_calls", 1) != 0
        for item in components
    )

    def total(field: str) -> int | None:
        if not complete:
            return None
        values = [item["usage"].get(field) for item in components]
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
            return None
        return sum(values)

    return {
        "method": "ahe_evolve_tracer", "debugger_invoked": debugger_invoked,
        "complete": complete, "source_path": str(source_path),
        "total_tokens": total("total_tokens"),
        "prompt_num_tokens": total("input_tokens"),
        "response_num_tokens": total("output_tokens"),
        "reasoning_num_tokens": total("reasoning_tokens"),
        "candidates": candidates, "debugger_usage": debugger_usage,
    }


def refine(
    state: AHEState,
    parent_agent: AHEAgent,
    feedback_results: Sequence[Any],
    *,
    output_dir: str | Path,
    iteration: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    candidate_count: int = 1,
    timeout_seconds: float = 1800,
    agent_limits: dict[str, int] | None = None,
    llm_environment: Mapping[str, str] | None = None,
) -> CandidateBatch:
    """Generate independent candidates and return them without selecting one."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    parent, best_ever = select_parent(state, parent_agent)
    feedback, score = _read_feedback(feedback_results)
    trace_path = _trace_from_feedback(feedback, parent_agent)
    iteration_dir = state.run_dir / "runs" / f"iteration_{iteration:03d}"
    iteration_dir.mkdir(parents=True, exist_ok=False)
    history = load_history(state.run_dir)
    history.append({
        "refinement_id": iteration_dir.name,
        "task_id": feedback["task_id"],
        "parent_score": score,
    })
    save_history(state.run_dir, history)
    trial_dir, task_name = _prepare_evidence(
        parent, feedback, score, trace_path, iteration_dir,
        parent_agent.trace_paths,
    )
    _write_analysis(
        iteration_dir, trial_dir, task_name, feedback, score,
        model=model, llm_environment=llm_environment,
    )
    debugger_invoked = True
    debugger_usage_path = iteration_dir / "input/analysis/debugger_usage_snapshot.json"
    debugger_usage = json.loads(debugger_usage_path.read_text(encoding="utf-8")) if debugger_usage_path.is_file() else None
    work_dir = state.run_dir / "candidates" / f"iteration_{iteration:03d}"
    hints = _strategy_hints() if candidate_count > 1 else []
    candidates: list[CandidateSpec] = []
    evolve_usage: list[dict[str, Any]] = []
    try:
        for index in range(candidate_count):
            candidate_id = f"variant_{index}"
            root = work_dir / candidate_id
            root.mkdir(parents=True, exist_ok=False)
            workspace = _prepare_candidate(parent, root)
            query = _evolution_query(
                state, iteration_dir, workspace, score, best_ever,
                hints[index] if index < len(hints) else None,
            )
            (root / "evolution_query.md").write_text(query, encoding="utf-8")
            evolve_result = _run_evolver(
                state, root, iteration_dir, workspace, query, model,
                reasoning_effort, timeout_seconds, llm_environment,
            )
            if agent_limits:
                # Reapply bounds after the evolver edits the candidate workspace.
                apply_agent_limits(workspace / "code_agent.yaml", agent_limits)
            evolve_usage.append({"candidate_id": candidate_id, "source_path": str(root / "evolve_usage_snapshot.json"),
                                 "usage": evolve_result.get("usage_snapshot")})
            candidate_agent = build(
                state, workspace=workspace, model=model,
                reasoning_effort=reasoning_effort,
                agent_limits=agent_limits,
                llm_environment=llm_environment,
            )
            candidates.append(CandidateSpec(candidate_id, workspace, candidate_agent))
    except BaseException:
        for candidate in candidates:
            candidate.agent.close()
        raise
    usage_report = _evolve_token_report(evolve_usage, debugger_invoked=debugger_invoked,
                                       debugger_usage=debugger_usage, source_path=Path(output_dir) / "ahe_refine.json")
    batch = CandidateBatch(f"iteration_{iteration:03d}", work_dir, parent, score, tuple(candidates), {"total_num_tokens": usage_report["total_tokens"]})
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "ahe_refine.json").write_text(json.dumps({
        "refinement_id": batch.refinement_id, "parent_workspace": str(parent),
        "parent_score": score, "candidate_ids": [c.candidate_id for c in candidates],
        "work_dir": str(work_dir),
        "token_usage_report": usage_report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return batch


def _adapt_score_prompt(prompt_path: Path) -> None:
    """Retarget only the run-local Evolve prompt, never the cloned AHE source.

    AHE's original objective is binary pass@1.  Our benchmark supplies a
    continuous score, so changing only the user query would leave the system
    prompt contradicting the evidence in reward.txt.  The marker makes this
    conversion safe when multiple candidate workers use the same copy.
    """

    marker = "<!-- benchmark-score objective -->"
    prompt = prompt_path.read_text(encoding="utf-8")
    if marker in prompt:
        return
    replacements = {
        "maximize **pass@1** (single-attempt success rate)":
            "maximize **mean benchmark score per attempt**",
        "**The sole optimization target is pass@1** — the probability that a single attempt succeeds. Every change you make should raise pass@1. Timed-out tasks count as failures — analyze why the agent ran out of time. Only pure infrastructure exceptions (sandbox crash, etc.) can be ignored.":
            "**The sole optimization target is mean benchmark score per attempt.** Compare numeric scores; do not infer task success from execution status. Analyze timeouts as evidence of a possible regression. Exclude only genuine infrastructure failures.",
        "When the experiment runs k>1 rollouts (indicated in the query), use the extra signal to diagnose:\n- **Partial-pass tasks** (some rollouts pass, some fail) are the most valuable. Compare the passing and failing rollouts of the *same task*, find the divergence point, and make the successful strategy the *reliable default*.\n- **pass@k** gauges capability ceiling but is NOT the target. Your goal is to turn pass@k successes into pass@1 successes by making the winning strategy consistent.":
            "When multiple rollouts are available, compare their numeric scores and traces. Find the divergence behind higher and lower scores on the same task; improve the reliability of the higher-scoring behavior. Do not optimize a binary pass@k metric.",
        "**For iteration 2+:** Compare task results across iterations. Check which tasks flipped (fail→pass) and which regressed (pass→fail). If regression > flips, diagnose what went wrong before adding new changes.":
            "**For iteration 2+:** Review recorded same-task parent/candidate score "
            "comparisons and investigate regressions before adding changes. Do not "
            "compare different training tasks as if they were the same evaluation set.",
        "After creating or modifying middleware**, read at least one `agent/nexau.txt` from a failed task — it contains runtime logs (middleware init errors, warnings, crashes) that static validation cannot catch":
            "After creating or modifying middleware**, inspect available runtime logs or traces for init errors, warnings, and crashes",
    }
    for original, adapted in replacements.items():
        if original not in prompt:
            raise RuntimeError(f"Cannot adapt an unexpected AHE prompt: {original[:60]}")
        prompt = prompt.replace(original, adapted, 1)
    convention_start = prompt.index("> **Loop convention")
    convention_end = prompt.index("\n```", convention_start)
    prompt = (
        prompt[:convention_start]
        + "> **Benchmark loop convention:** You are producing iteration `{{ iteration }}`. "
        "Its `input/` contains the active parent's evaluation on the current training task. "
        "After your edits, the benchmark evaluates your candidate on this SAME task. "
        "The next iteration may use a DIFFERENT task; do not infer regressions from "
        "cross-task score changes. Prior same-task comparisons and archived manifests "
        "are linked in `evolution_history.md`. The directory layout below is the "
        "upstream reference: this adapter stores candidate outputs under "
        "`candidates/iteration_NNN/variant_N/`, not `runs/.../evolve/`. "
        "Use the exact evidence and manifest paths in the user query. The initial "
        "configuration is `method_config.toml`, not `config_snapshot.yaml`.\n"
        + prompt[convention_end:]
    )
    prompt_path.write_text(f"{marker}\n{prompt}", encoding="utf-8")


def _prepare_evolve_runtime(source_dir: Path, runtime_dir: Path) -> Path:
    """Keep upstream middleware and add the benchmark's visual reader."""

    import yaml

    shutil.copytree(source_dir, runtime_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _adapt_score_prompt(runtime_dir / "evolve_prompt.md")
    config_path = runtime_dir / "evolve_agent.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    middlewares = config.get("middlewares", [])
    for middleware in middlewares:
        if middleware.get("import") == "middleware.context_compaction:ContextCompactionMiddleware":
            middleware.setdefault("params", {})["max_context_tokens"] = config["max_context_tokens"]
    # Enable the already-vendored visual reader; keep binary output untruncated.
    if not any(t.get("name") == "read_visual_file" for t in config.get("tools", [])):
        config.setdefault("tools", []).append({
            "name": "read_visual_file",
            "yaml_path": "./tool_descriptions/read_visual_file.tool.yaml",
            "binding": "tools.file_tools:read_visual_file",
        })
    for middleware in middlewares:
        if middleware.get("import") == "middleware.long_tool_output:LongToolOutputMiddleware":
            bypass = middleware.setdefault("params", {}).setdefault("bypass_tool_names", [])
            if "read_visual_file" not in bypass:
                bypass.append("read_visual_file")
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    return config_path


def _worker(request_path: Path, result_path: Path) -> None:
    """Extract upstream ``run_evolve_agent_on_variant`` without Harbor calls."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    run_dir = Path(request["run_dir"])
    evolve_dir = request_path.parent / "evolve_agent_runtime"
    config_path = evolve_dir / "evolve_agent.yaml"
    upstream_root = Path(__file__).resolve().parent / "agentic-harness-engineering"
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(evolve_dir))
    os.environ["EVOLVE_WORK_DIR"] = str(run_dir)
    from dotenv import load_dotenv
    from nexau import Agent, AgentConfig

    load_dotenv(upstream_root / ".env", override=False)
    config = AgentConfig.from_yaml(config_path=config_path)
    configure_reasoning_effort(config, request.get("reasoning_effort"))
    # Keep the Evolve Agent on the same model-compatible request contract as
    # the coding Agent; changing its run-local YAML is unnecessary.
    config.llm_config.temperature = None
    agent = Agent(config=config)
    from benchmark.visual_evidence import chat_message
    raw = agent.run(
        message=chat_message(request["query"], request.get("image_paths") or []),
        context={"date": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
                 "username": "agentic-harness-engineering", "working_directory": str(run_dir),
                 "workspace_path": request["workspace_rel"], "iteration": request["iteration"],
                 "adb_llm": None},
    )
    result = raw[0] if isinstance(raw, tuple) else raw
    evolve_dir_out = request_path.parent
    trace = getattr(agent, "full_trace", None) or list(agent.history)
    messages = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in trace]
    (evolve_dir_out / "evolve_trace.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
    snapshots = [nexau_usage_snapshot(tracer.dump_traces()) for tracer in agent.config.tracers
                 if isinstance(tracer, InMemoryTracer)]
    snapshot = snapshots[0] if len(snapshots) == 1 else None
    (evolve_dir_out / "evolve_usage_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps({"result": str(result or ""), "usage_snapshot": snapshot},
                                      ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("usage: python -m rsi_methods.AHE.refine --worker REQUEST RESULT")
    _worker(Path(sys.argv[2]), Path(sys.argv[3]))
