"""
Explore-Agent Runner

Run two sub-agents in parallel (source exploration + web research),
write the distilled knowledge into skill files for the evolve agent.

Main entry point: run_explore_agent(config, exp_dir)
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

AUTO_DIR = Path(__file__).resolve().parent.parent.parent

ML_SKILL_NAMES = [
    "nexau-framework-internals",
    "coding-agent-sota-research",
]

SOURCE_AGENT_SKILLS = [
    "nexau-framework-internals",
]

WEB_AGENT_SKILLS = [
    "coding-agent-sota-research",
]

CODE_SOURCES_DIR = AUTO_DIR / ".code_sources"

_HISTORY_FLUSH_INTERVAL_SEC = 30


def _dump_agent_to_disk(agent: "Agent | None", output_dir: Path) -> None:  # noqa: F821
    """Persist the agent history and tracer to disk."""
    if agent is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        history_snapshot = list(agent.history)
        history_messages = [msg.model_dump(mode="json") for msg in history_snapshot]
        history_path = output_dir / "explore_agent_history.json"
        tmp_path = str(history_path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(history_messages, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(history_path))
    except Exception as e:
        sys.stderr.write(f"[explore-agent] failed to flush history: {e}\n")

    try:
        from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
        tracers_snapshot = list(agent.config.tracers)
        for tracer in tracers_snapshot:
            if isinstance(tracer, InMemoryTracer):
                tracer_path = output_dir / "explore_agent_tracer.json"
                tmp_path = str(tracer_path) + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(tracer.dump_traces(), f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, str(tracer_path))
                break
    except Exception as e:
        sys.stderr.write(f"[explore-agent] failed to flush tracer: {e}\n")


def _periodic_flush(agent_ref: list, output_dir: Path, stop_event: threading.Event) -> None:
    """Background thread: periodically flush history and tracer to disk."""
    while not stop_event.wait(_HISTORY_FLUSH_INTERVAL_SEC):
        agent = agent_ref[0] if agent_ref else None
        if agent is not None:
            _dump_agent_to_disk(agent, output_dir)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge two dicts, with overlay overriding base."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def prepare_code_sources(config: dict) -> dict[str, Path]:
    """Prepare source directories and return a {name: local_path} mapping.

    Resolution priority:
    1. local_fallback (an existing local repo, e.g. ../nexau)
    2. .code_sources/ cache (previously cloned)
    3. git clone into .code_sources/ (first use)
    """
    sources = config["explore_agent"]["code_sources"]
    paths = {}

    for name, src_cfg in sources.items():
        if src_cfg["type"] == "local":
            path = Path(src_cfg["path"])
            if not path.is_absolute():
                path = (AUTO_DIR / path).resolve()
            if path.exists():
                paths[name] = path
                print(f"[explore-agent] using local path {name} → {path}")
            else:
                print(f"[explore-agent] ⚠️ local path does not exist: {path}")

        elif src_cfg["type"] == "git":
            local_fallback = src_cfg.get("local_fallback")
            if local_fallback:
                fallback_path = (AUTO_DIR / local_fallback).resolve()
                if fallback_path.exists():
                    paths[name] = fallback_path
                    print(f"[explore-agent] using local fallback {name} → {fallback_path}")
                    continue

            ref = src_cfg.get("ref", "main")
            clone_dir = CODE_SOURCES_DIR / f"{name}@{ref}"
            if not clone_dir.exists():
                CODE_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[explore-agent] cloning {name}@{ref}...")
                subprocess.run(
                    [
                        "git", "clone", "--depth", "1",
                        "--branch", ref,
                        src_cfg["url"], str(clone_dir),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"[explore-agent] cloned {name}@{ref} → {clone_dir}")
            else:
                print(f"[explore-agent] using cache {name}@{ref} → {clone_dir}")
            paths[name] = clone_dir

    return paths


def _build_source_agent_query(
    nexau_path: Path,
    output_skill_dir: Path,
) -> str:
    """Build the initial query for the source-exploration agent."""
    return "\n".join([
        "Start the NexAU framework source exploration task.",
        "",
        "## Source location (read-only)",
        f"- NexAU framework: `{nexau_path}`",
        "",
        "## Output location",
        f"- Skill output directory: `{output_skill_dir}/`",
        "",
        "## Your deliverables",
        "",
        "1. `nexau-framework-internals/SKILL.md`",
        "",
        "## Instructions",
        "Following the system prompt, explore the NexAU framework source and write the skill files.",
        "Use read_file (with offset/limit) rather than run_shell_command to read files.",
        "Call complete_task when done.",
    ])


def _build_web_agent_query(
    output_skill_dir: Path,
    web_sources: list[dict],
) -> str:
    """Build the initial query for the web-research agent."""
    lines = [
        "Start the SOTA research task.",
        "",
        "## Output location",
        f"- Skill output directory: `{output_skill_dir}/`",
        "",
        "## Your deliverables",
        "",
        "1. `coding-agent-sota-research/SKILL.md` -- architecture, benchmarks, techniques, ablations",
        "",
    ]

    if web_sources:
        lines.append("## Required web pages")
        for ws in web_sources:
            lines.append(f"- **{ws['url']}** — {ws.get('focus', '')}")
        lines.append("")

    lines.extend([
        "## Instructions",
        "1. First use WebFetch to read every URL listed above",
        "2. Right after reading, use write_file to produce a first draft of the skill file",
        "3. Then use WebSearch for autonomous deep research (15-20 searches)",
        "4. Update the skill file with the new findings",
        "5. Call complete_task to finish",
        "",
        "⚠️ Key rules:",
        "- Write the first draft before expanding; do not wait until research is complete",
        "- Record exact figures and cited URLs",
        "- Target 400-800 lines for coding-agent-sota-research",
        "- Write architecture/middleware/tools/ablations and similar content into coding-agent-sota-research",
    ])

    return "\n".join(lines)


def _prepare_agent_config(
    base_config_path: Path,
    agent_patch: dict,
) -> Path:
    """Read the base agent yaml, apply the patch, and write a temp file in the same directory.

    Use tempfile to generate a unique name and avoid concurrent experiments racing on the same file.
    The temp file sits next to the base so relative paths inside the yaml still resolve.
    If there is no patch, return base_config_path directly (the caller compares to decide whether cleanup is needed).
    """
    if not agent_patch:
        return base_config_path

    with open(base_config_path, encoding="utf-8") as f:
        agent_yaml = yaml.safe_load(f)
    patched = _deep_merge(agent_yaml, agent_patch)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(base_config_path.parent),
        suffix=".yaml",
        prefix="_patched_",
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.dump(patched, f, default_flow_style=False, allow_unicode=True)
    return Path(tmp_path)


def _run_single_agent(
    agent_name: str,
    config_path: Path,
    query: str,
    context: dict,
    output_dir: Path,
) -> bool:
    """Run a single explore-agent sub-agent. Called from its own thread; does not modify sys.stdout.

    Returning True means the agent completed normally (not necessarily producing every skill).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "explore_agent.log"

    agent_ref: list = [None]
    stop_flush = threading.Event()
    flush_thread: threading.Thread | None = None
    log_fh = None

    try:
        from nexau import Agent

        log_fh = open(log_file_path, "w", encoding="utf-8")

        def _log(msg: str) -> None:
            line = f"[{agent_name}] {msg}\n"
            sys.stderr.write(line)
            try:
                log_fh.write(line)
                log_fh.flush()
            except (ValueError, OSError):
                pass

        _log(f"starting agent (config={config_path})")

        agent = Agent.from_yaml(config_path=config_path)
        agent_ref[0] = agent

        flush_thread = threading.Thread(
            target=_periodic_flush,
            args=(agent_ref, output_dir, stop_flush),
            daemon=True,
        )
        flush_thread.start()

        agent.run(message=query, context=context)
        _log("agent run complete")
        return True

    except Exception as e:
        sys.stderr.write(f"[{agent_name}] ❌ agent raised an exception: {e}\n")
        return False

    finally:
        stop_flush.set()
        if flush_thread and flush_thread.is_alive():
            flush_thread.join(timeout=5)
        _dump_agent_to_disk(agent_ref[0], output_dir)
        if log_fh:
            log_fh.close()


def run_explore_agent(config: dict, exp_dir: Path) -> bool:
    """Run two explore-agent sub-agents in parallel and write the produced skills to exp_dir/skills/.

    - agent_source: scan the NexAU framework source -> nexau-framework-internals
    - agent_web:    web research on SOTA coding-agent architectures -> coding-agent-sota-research

    Return True if all skills were produced successfully, False on failure or partial production.
    """
    from concurrent.futures import ThreadPoolExecutor

    ml_config = config.get("explore_agent", {})
    if not ml_config.get("enabled", False):
        return False

    skills_dir = exp_dir / "evolve_agent" / "skills"
    start_time = time.monotonic()

    for skill_name in ML_SKILL_NAMES:
        (skills_dir / skill_name).mkdir(parents=True, exist_ok=True)

    try:
        code_paths = prepare_code_sources(config)
    except subprocess.CalledProcessError as e:
        print(f"[explore-agent] ❌ source preparation failed: {e}")
        return False

    if "nexau" not in code_paths:
        print("[explore-agent] ⚠️ missing NexAU source; skipping explore-agent")
        return False

    ml_agent_patch = config.get("explore_agent_patch", {})
    ml_dir = AUTO_DIR / "agents" / "explore_agent"

    source_base = (ml_dir / "source_agent" / "agent.yaml").resolve()
    web_base = (ml_dir / "web_agent" / "agent.yaml").resolve()

    source_config_path = _prepare_agent_config(source_base, ml_agent_patch)
    web_config_path = _prepare_agent_config(web_base, ml_agent_patch)

    temp_configs = [p for p in (source_config_path, web_config_path)
                    if p not in (source_base, web_base)]

    source_query = _build_source_agent_query(
        nexau_path=code_paths["nexau"],
        output_skill_dir=skills_dir,
    )
    web_query = _build_web_agent_query(
        output_skill_dir=skills_dir,
        web_sources=ml_config.get("web_sources", []),
    )

    source_context = {
        "nexau_path": str(code_paths["nexau"]),
        "output_skill_dir": str(skills_dir),
    }
    today_str = datetime.now().strftime("%Y-%m-%d")
    web_context = {
        "output_skill_dir": str(skills_dir),
        "web_sources": ml_config.get("web_sources", []),
        "date": today_str,
    }

    source_output_dir = exp_dir / "explore_agent_output" / "source"
    web_output_dir = exp_dir / "explore_agent_output" / "web"

    print(f"[explore-agent] 🚀 launching 2 sub-agents in parallel:")
    print(f"  agent_source → Skills: {SOURCE_AGENT_SKILLS}")
    print(f"  agent_web    → Skills: {WEB_AGENT_SKILLS}")
    print(f"  source workspace: {source_output_dir}")
    print(f"  web workspace:    {web_output_dir}")

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml") as pool:
        source_future = pool.submit(
            _run_single_agent,
            "agent_source", source_config_path, source_query,
            source_context, source_output_dir,
        )
        web_future = pool.submit(
            _run_single_agent,
            "agent_web", web_config_path, web_query,
            web_context, web_output_dir,
        )

        source_ok = source_future.result()
        web_ok = web_future.result()

    for tmp in temp_configs:
        tmp.unlink(missing_ok=True)

    elapsed = time.monotonic() - start_time
    produced = [s for s in ML_SKILL_NAMES if (skills_dir / s / "SKILL.md").exists()]
    success = len(produced) == len(ML_SKILL_NAMES)

    print(f"\n[explore-agent] === parallel execution complete ({elapsed:.0f}s) ===")
    print(f"  agent_source: {'✅' if source_ok else '❌'}")
    print(f"  agent_web:    {'✅' if web_ok else '❌'}")

    if success:
        print(f"[explore-agent] ✅ all {len(ML_SKILL_NAMES)} skills were produced")
    else:
        missing_skills = [s for s in ML_SKILL_NAMES if s not in produced]
        print(f"[explore-agent] ⚠️ produced {len(produced)}/{len(ML_SKILL_NAMES)} skills")
        print(f"[explore-agent] missing: {missing_skills}")

    _save_explore_agent_log(exp_dir, code_paths, produced, elapsed, success)

    return success


def _save_explore_agent_log(
    exp_dir: Path,
    code_paths: dict[str, Path],
    produced: list[str],
    elapsed: float,
    success: bool,
) -> None:
    """Save the explore-agent execution log."""
    log = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "success": success,
        "code_sources": {k: str(v) for k, v in code_paths.items()},
        "produced_skills": produced,
        "missing_skills": [s for s in ML_SKILL_NAMES if s not in produced],
    }
    log_path = exp_dir / "explore_agent_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def register_explore_agent_skills(exp_dir: Path) -> None:
    """Register the skills produced by explore-agent into the experiment copy of evolve_agent.yaml,
    and append skill guidance to the experiment copy of evolve_prompt.md."""
    evolve_yaml_path = exp_dir / "evolve_agent" / "evolve_agent.yaml"
    if not evolve_yaml_path.exists():
        print("[explore-agent] ⚠️ evolve_agent.yaml does not exist; skipping registration")
        return

    with open(evolve_yaml_path, encoding="utf-8") as f:
        evolve_config = yaml.safe_load(f)

    existing_skills = evolve_config.get("skills", [])
    registered = []

    for skill_name in ML_SKILL_NAMES:
        skill_path = f"./skills/{skill_name}"
        skill_md = exp_dir / "evolve_agent" / "skills" / skill_name / "SKILL.md"
        if skill_md.exists() and skill_path not in existing_skills:
            existing_skills.append(skill_path)
            registered.append(skill_name)

    if not registered:
        print("[explore-agent] no new skill to register")
        return

    evolve_config["skills"] = existing_skills

    with open(evolve_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(evolve_config, f, default_flow_style=False, allow_unicode=True)

    print(f"[explore-agent] registered {len(registered)} skills into evolve_agent.yaml: {registered}")

    prompt_path = exp_dir / "evolve_agent" / "evolve_prompt.md"
    _ML_PROMPT_MARKER = "### Explore-Agent Skills (auto-generated)"
    if prompt_path.exists():
        existing_content = prompt_path.read_text(encoding="utf-8")
        if _ML_PROMPT_MARKER in existing_content:
            print("[explore-agent] evolve_prompt.md already contains skill guidance; skipping append")
            return

        ml_skill_guide = f"""

{_ML_PROMPT_MARKER}

The following skills are auto-generated by explore-agent and are your **core knowledge source**. Load them via LoadSkill:

| Skill | Coverage | When to use |
|-------|---------|---------|
| `nexau-framework-internals` | Comprehensive reference for the NexAU framework: config schema, component creation patterns (middleware/tool/skill/sub-agent), exact executor loop sequence, hook boundary behavior, token accounting, sub-agent lifecycle, and undocumented gotchas | When creating or modifying any framework component, and when encountering unexpected runtime behavior |
| `coding-agent-sota-research` | State-of-the-art coding-agent architecture patterns, design principles, exact benchmark scores, real prompt text, middleware code, ablation results, negative experimental results, and a gap-analysis framework | When a decision needs concrete figures or code references, or when architecture design guidance is required |
"""
        with open(prompt_path, "a", encoding="utf-8") as f:
            f.write(ml_skill_guide)
        print("[explore-agent] appended skill guidance to the evolve_prompt.md copy")
