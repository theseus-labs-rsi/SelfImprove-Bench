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
from typing import Any, Sequence
from uuid import uuid4

from benchmark.contracts import to_jsonable
from benchmark.token_ledger import nexau_usage_snapshot

from .agent import AHEAgent, configure_reasoning_effort
from .build import build
from .init import AHEState, apply_agent_limits


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
    (trial_dir / "agent" / "all_step_traces.json").write_text(
        json.dumps([str(path) for path in trace_paths], ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (trial_dir / "verifier" / "reward.txt").write_text(str(score), encoding="utf-8")
    (trial_dir / "feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps({"task_id": feedback["task_id"], "score": score}, ensure_ascii=False),
        encoding="utf-8",
    )
    return trial_dir, name


def _write_analysis(
    iteration_dir: Path,
    trial_dir: Path,
    task_name: str,
    feedback: dict[str, Any],
    score: float,
) -> str:
    """Provide sourced evidence; use Agent Debugger when already installed.

    If ``adb`` is unavailable, the report says so and points the Evolve Agent
    to the raw trace.  It does not claim a root cause without analysis.
    """

    trace_path = trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json"
    analysis_dir = iteration_dir / "input" / "analysis"
    detail_dir = analysis_dir / "detail"
    detail_dir.mkdir(parents=True)
    evaluator_note = str(feedback["evaluation"].get("feedback") or "")
    diagnosis = _ask_debugger(trace_path, score, evaluator_note)
    relative_trace = trace_path.relative_to(iteration_dir.parent.parent).as_posix()
    overview = (
        f"# Benchmark evidence for {task_name}\n\n"
        f"- Relative score: {score}\n"
        f"- Execution status: {feedback['evaluation'].get('execution_status')}\n"
        f"- Trace (latest bounded-history window): `{relative_trace}`\n"
        f"- All original per-step traces: `{relative_trace.rsplit('/', 1)[0]}/all_step_traces.json`\n"
        f"- Detailed analysis: `runs/{iteration_dir.name}/input/analysis/detail/{task_name}.md`\n\n"
        "The score is continuous; no binary pass/fail label is inferred from execution status.\n"
    )
    detail = f"# {task_name}\n\nScore: {score}\n\nEvaluator feedback: {evaluator_note}\n\nTrace: `{relative_trace}`\n\n{diagnosis}\n"
    (analysis_dir / "overview.md").write_text(overview, encoding="utf-8")
    (detail_dir / f"{task_name}.md").write_text(detail, encoding="utf-8")
    return overview


def _ask_debugger(trace_path: Path, score: float, evaluator_note: str) -> str:
    """Run an installed Agent Debugger without installing software implicitly."""

    adb = shutil.which("adb")
    if adb is None:
        return "Agent Debugger is unavailable. Inspect the linked raw trace directly."
    question = (
        f"Analyze this agent trace against benchmark score {score}. "
        f"Evaluator feedback: {evaluator_note}. Identify evidence-backed causes; "
        "do not infer task success from process completion."
    )
    try:
        completed = subprocess.run(
            [adb, "ask", "-t", str(trace_path), "-q", question, "--format", "json"],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "Agent Debugger timed out; inspect the raw trace directly."
    if completed.returncode != 0:
        return f"Agent Debugger failed; inspect the raw trace. Error: {completed.stderr[-1000:]}"
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return f"Agent Debugger returned non-JSON output: {completed.stdout[-1000:]}"
    return str(result.get("response") or result.get("answer") or "No diagnosis returned.")


def _prepare_candidate(parent: Path, candidate_root: Path) -> Path:
    """Copy the parent and give this variant its own Git history for AHE edits."""

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
        f"Read `{analysis_rel}/overview.md` and its detail file, then inspect the linked trace.",
        f"The benchmark's objective is relative_score; parent score is {score}.",
        "The reward file contains that continuous score. Execution completion is not a task pass.",
        "Make evidence-backed harness changes in your writable workspace only.",
        "Do not run Harbor or select a winner. The benchmark evaluates returned candidates later.",
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
) -> None:
    """Run the original Evolve Agent in a separate import namespace."""

    request_path = candidate_root / "evolve_request.json"
    result_path = candidate_root / "evolve_result.json"
    request_path.write_text(json.dumps({
        "run_dir": str(state.run_dir), "evolve_agent_dir": str(state.evolve_agent_dir),
        "workspace_rel": workspace.relative_to(state.run_dir).as_posix(),
        "iteration": int(iteration_dir.name.split("_")[-1]), "query": query,
        "reasoning_effort": reasoning_effort,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    code_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(code_root), env.get("PYTHONPATH"))))
    if model:
        env["LLM_MODEL"] = model
    completed = subprocess.run(
        [sys.executable, "-m", "rsi_methods.AHE.refine", "--worker", str(request_path), str(result_path)],
        env=env, capture_output=True, text=True, timeout=timeout_seconds,
    )
    (candidate_root / "evolve.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"AHE candidate generation failed; inspect {candidate_root / 'evolve.log'}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _strategy_hints() -> list[str]:
    """Reuse the variant strategy hints declared in upstream AHE config."""

    import yaml

    config_path = Path(__file__).resolve().parent / "agentic-harness-engineering" / "configs" / "base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return list(config.get("best_of_n", {}).get("strategy_hints", []))


def _evolve_token_report(
    candidates: list[dict[str, Any]], *, debugger_invoked: bool, source_path: Path,
) -> dict[str, Any]:
    """Sum provider-reported NexAU spans; do not invent missing subfield usage."""

    complete = not debugger_invoked and bool(candidates) and all(
        isinstance(item.get("usage"), dict)
        and isinstance(item["usage"].get("total_tokens"), int)
        and not isinstance(item["usage"]["total_tokens"], bool)
        for item in candidates
    )

    def total(field: str) -> int | None:
        if not complete:
            return None
        values = [item["usage"].get(field) for item in candidates]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            return None
        return sum(values)

    return {
        "method": "ahe_evolve_tracer", "debugger_invoked": debugger_invoked,
        "complete": complete, "source_path": str(source_path),
        "total_tokens": total("total_tokens"),
        "prompt_num_tokens": total("input_tokens"),
        "response_num_tokens": total("output_tokens"),
        "reasoning_num_tokens": total("reasoning_tokens"),
        "candidates": candidates,
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
) -> CandidateBatch:
    """Generate independent candidates and return them without selecting one."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    parent, best_ever = select_parent(state, parent_agent)
    feedback, score = _read_feedback(feedback_results)
    if not parent_agent.trace_paths:
        raise RuntimeError("AHE refine needs a trace from the parent Agent")
    iteration_dir = state.run_dir / "runs" / f"iteration_{iteration:03d}"
    iteration_dir.mkdir(parents=True, exist_ok=False)
    trial_dir, task_name = _prepare_evidence(
        parent, feedback, score, parent_agent.trace_paths[-1], iteration_dir,
        parent_agent.trace_paths,
    )
    debugger_invoked = shutil.which("adb") is not None
    _write_analysis(iteration_dir, trial_dir, task_name, feedback, score)
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
                reasoning_effort, timeout_seconds,
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
            )
            candidates.append(CandidateSpec(candidate_id, workspace, candidate_agent))
    except BaseException:
        for candidate in candidates:
            candidate.agent.close()
        raise
    batch = CandidateBatch(f"iteration_{iteration:03d}", work_dir, parent, score, tuple(candidates))
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "ahe_refine.json").write_text(json.dumps({
        "refinement_id": batch.refinement_id, "parent_workspace": str(parent),
        "parent_score": score, "candidate_ids": [c.candidate_id for c in candidates],
        "work_dir": str(work_dir),
        "token_usage_report": _evolve_token_report(
            evolve_usage, debugger_invoked=debugger_invoked,
            source_path=output / "ahe_refine.json",
        ),
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
        "Check which tasks flipped (fail→pass) and which regressed (pass→fail). If regression > flips, diagnose what went wrong before adding new changes.":
            "Compare per-task score changes and investigate regressions before adding new changes.",
        "The analysis reports are pre-built summaries of all task failures with root causes already identified. They save you significant time — do NOT skip them to read raw traces directly.":
            "The analysis files summarize available benchmark evidence; they may not contain a root-cause diagnosis. Read the linked raw trace when the summary is insufficient, and distinguish observation from hypothesis.",
        "this is your primary information source. It contains pre-analyzed root causes, failure patterns, and strategies for every task":
            "this points to the benchmark score, evaluator feedback, and available trace for each task",
        "detailed per-task analysis with specific failure points and successful strategies":
            "per-task evidence and any available diagnosis",
        "Only fall back to reading raw `nexau_in_memory_tracer.cleaned.json` when analysis is missing or insufficient for a specific question — this should be rare":
            "Read the linked raw `nexau_in_memory_tracer.cleaned.json` whenever the summary does not establish a root cause",
        "After creating or modifying middleware**, read at least one `agent/nexau.txt` from a failed task — it contains runtime logs (middleware init errors, warnings, crashes) that static validation cannot catch":
            "After creating or modifying middleware**, inspect available runtime logs or traces for init errors, warnings, and crashes",
    }
    for original, adapted in replacements.items():
        if original not in prompt:
            raise RuntimeError(f"Cannot adapt an unexpected AHE prompt: {original[:60]}")
        prompt = prompt.replace(original, adapted, 1)
    prompt_path.write_text(f"{marker}\n{prompt}", encoding="utf-8")


def _prepare_evolve_runtime(source_dir: Path, runtime_dir: Path) -> Path:
    """Create a candidate-local Evolve Agent compatible with host NexAU.

    Upstream AHE pins NexAU 0.3.9, while the benchmark environment currently
    uses 0.4.1. Its bundled ContextCompactionMiddleware imports an AG-UI event
    absent from this environment. Context compaction is a long-run safeguard,
    not part of candidate editing, so omit only that middleware in this local
    runtime copy. LongToolOutputMiddleware and all evolution tools remain.
    """

    import yaml

    shutil.copytree(source_dir, runtime_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _adapt_score_prompt(runtime_dir / "evolve_prompt.md")
    config_path = runtime_dir / "evolve_agent.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    middlewares = config.get("middlewares", [])
    incompatible = "middleware.context_compaction:ContextCompactionMiddleware"
    retained = [item for item in middlewares if item.get("import") != incompatible]
    if len(retained) == len(middlewares):
        raise RuntimeError("AHE Evolve config has no expected context-compaction middleware")
    config["middlewares"] = retained
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    return config_path


def _worker(request_path: Path, result_path: Path) -> None:
    """Extract upstream ``run_evolve_agent_on_variant`` without Harbor calls."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    run_dir = Path(request["run_dir"])
    evolve_source = Path(request["evolve_agent_dir"])
    evolve_dir = request_path.parent / "evolve_agent_runtime"
    config_path = _prepare_evolve_runtime(evolve_source, evolve_dir)
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
    raw = agent.run(
        message=request["query"],
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
