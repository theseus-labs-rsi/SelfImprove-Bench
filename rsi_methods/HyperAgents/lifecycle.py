from __future__ import annotations

from dataclasses import dataclass
import json
import os
import posixpath
from pathlib import Path
import shutil
import uuid
from typing import Any, Callable, Mapping

from .archive_utils import (
    RUNTIME_IMAGE_NAME,
    apply_diffs_checked,
    check_candidate_compilation,
    describe_patch,
    patch_paths_for_metadata,
)
from .candidate import CandidateSnapshot, with_lineage
from .result_store import result_is_valid, save_result
from .runtime_agent import RuntimeAgent
from .network_preflight import check_model_network, container_model_environment
from .hyperagents.utils.common import file_exist_and_not_empty
from .hyperagents.utils.constants import REPO_NAME
from .hyperagents.utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    safe_log,
    setup_logger,
)


@dataclass(frozen=True)
class HyperAgentRuntime:
    docker_client: Any
    meta_model: Mapping[str, Any]
    task_model: Mapping[str, Any]
    llm_environment: Mapping[str, str]


@dataclass
class InstantiatedAgent:
    """One candidate loaded into an isolated, running container."""

    snapshot: CandidateSnapshot
    runtime: HyperAgentRuntime
    container: Any
    commit_hash: str
    artifact_dir: Path

    def task_agent(self) -> RuntimeAgent:
        model = self.runtime.task_model
        return RuntimeAgent(
            container=self.container,
            agent_path=f"/{REPO_NAME}/task_agent.py",
            repo_path=f"/{REPO_NAME}",
            model=model["name"],
            reasoning_effort=model.get("reasoning_effort"),
            backend=model["backend"],
            api_mode=model["api_mode"],
            api_key_env=model["api_key_env"],
            base_url_env=model["base_url_env"],
            generation_id=self.snapshot.candidate_id,
            artifact_dir=str(self.artifact_dir),
            max_output_tokens=model.get("max_output_tokens", 16384),
            llm_environment=self.runtime.llm_environment,
        )

    def close(self) -> None:
        try:
            reset = self.container.exec_run(
                ["git", "reset", "--hard", self.snapshot.root_commit],
                workdir=f"/{REPO_NAME}",
            )
            log_container_output(reset)
            clean = self.container.exec_run(
                ["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
            )
            log_container_output(clean)
        finally:
            cleanup_container(self.container)


@dataclass(frozen=True)
class RefineOutput:
    agent: InstantiatedAgent
    changed: bool
    patch_path: Path | None
    result: Any
    metadata: Mapping[str, Any]


def new_generation_metadata(
    output_dir: Path,
    candidate_id: str | int,
    parent_id: str | int | None,
    *,
    run_baseline: str | None,
    refine_requested: bool,
) -> dict[str, Any]:
    return {
        "gen_output_dir": str(output_dir),
        "current_genid": candidate_id,
        "parent_genid": parent_id,
        "run_baseline": run_baseline,
        "prev_patch_files": [],
        "curr_patch_files": [],
        "refine_attempted": refine_requested,
        "meta_agent_command_successful": None,
        "patch_produced": False,
        "compilation_successful": False if refine_requested else None,
        "refine_successful": False if refine_requested else None,
        "run_eval": not refine_requested,
    }


def instantiate_agent(
    snapshot: CandidateSnapshot,
    *,
    runtime: HyperAgentRuntime,
    run_id: str,
    artifact_dir: str | Path,
    preserve_unmodified_copy: bool = False,
) -> InstantiatedAgent:
    """Load one persistent candidate into a fresh reusable container."""
    resolved_artifacts = Path(artifact_dir).resolve()
    resolved_artifacts.parent.mkdir(parents=True, exist_ok=True)
    setup_logger(str(resolved_artifacts.parent / "container.log"))
    try:
        runtime.docker_client.ping()
    except Exception as exc:
        raise RuntimeError(
            "Docker daemon is unavailable. Check DOCKER_HOST, socket permissions, "
            "and whether your Docker daemon is running."
        ) from exc
    image_name = f"{REPO_NAME}:run-{run_id}"
    container_name = f"{REPO_NAME}-gl-container-{run_id}"
    container = build_container(
        runtime.docker_client,
        str(snapshot.workspace),
        image_name,
        container_name,
        runtime_image_name=RUNTIME_IMAGE_NAME,
    )
    if container is None:
        raise RuntimeError("Docker container creation failed.")
    try:
        container.start()
        check_model_network(container, runtime)
        if preserve_unmodified_copy:
            copied = container.exec_run(
                ["cp", "-r", f"/{REPO_NAME}", f"/DONOTTOUCH_{REPO_NAME}"],
                workdir="/",
            )
            log_container_output(copied)
        agent = InstantiatedAgent(
            snapshot=snapshot,
            runtime=runtime,
            container=container,
            commit_hash=snapshot.root_commit,
            artifact_dir=resolved_artifacts,
        )
        return reconstruct_agent(agent, snapshot)
    except Exception:
        try:
            container.remove(force=True)
        except Exception:
            pass
        raise


def reconstruct_agent(
    agent: InstantiatedAgent,
    snapshot: CandidateSnapshot,
    *,
    artifact_dir: str | Path | None = None,
) -> InstantiatedAgent:
    """Deterministically load a candidate into an existing container."""
    repo_path = f"/{REPO_NAME}"
    # Bind mounts retain host ownership, which can differ from the container user.
    _run_candidate_command(
        agent.container,
        [
            "git", "config", "--global", "--replace-all",
            "safe.directory", repo_path, f"^{repo_path}$",
        ],
        "/",
        "Candidate repository trust configuration",
    )
    _run_candidate_command(
        agent.container,
        ["git", "reset", "--hard", snapshot.root_commit],
        repo_path,
        "Candidate reset",
    )
    _run_candidate_command(
        agent.container,
        ["git", "clean", "-fd"],
        repo_path,
        "Candidate clean",
    )
    from .runtime_safety import install_runtime_safety
    # Transport fixes belong to the seed, not to each evolved candidate.
    install_runtime_safety(agent.container, snapshot.root_commit, repo_path)
    agent.commit_hash = apply_diffs_checked(
        agent.container, list(snapshot.lineage_patches)
    )
    agent.snapshot = snapshot
    if artifact_dir is not None:
        agent.artifact_dir = Path(artifact_dir).resolve()
    return agent


def _run_candidate_command(
    container: Any,
    command: list[str],
    workdir: str,
    description: str,
) -> None:
    result = container.exec_run(command, workdir=workdir)
    if result.exit_code != 0:
        output = result.output.decode("utf-8", errors="replace")
        raise RuntimeError(f"{description} failed: {output}")
    log_container_output(result)


def evaluate_agent(
    agent: InstantiatedAgent,
    play: Callable[[RuntimeAgent], Any],
    *,
    output_dir: str | Path,
) -> Any:
    """Evaluate one already-instantiated candidate and persist its Result."""
    resolved_output = Path(output_dir).resolve()
    agent.artifact_dir = resolved_output / "artifacts"
    agent.artifact_dir.mkdir(parents=True, exist_ok=True)
    result = play(agent.task_agent())
    save_result(str(resolved_output), result)
    return result


def refine(
    agent: InstantiatedAgent,
    results: Any,
    *,
    play: Callable[[RuntimeAgent], Any] | None = None,
    output_dir: str | Path,
    iterations_left: int,
    metadata: dict[str, Any] | None = None,
    history_path: str | None = None,
    evidence_source_dir: str | Path | None = None,
    run_workdir: str | None = None,
    search_archive: Any | None = None,
) -> RefineOutput:
    """Improve one candidate; legacy callers may request immediate evaluation."""
    resolved_output = Path(output_dir).resolve()
    metadata = dict(metadata or {})
    if history_path is None:
        history_path = _stage_direct_evidence(
            agent, results, resolved_output,
            metadata=metadata,
            evidence_source_dir=evidence_source_dir,
        )
        if search_archive is not None:
            archive_history = resolved_output / "archive_history"
            if archive_history.exists():
                shutil.rmtree(archive_history)
            search_archive.stage_history(archive_history)
            copy_to_container(agent.container, archive_history, f"{history_path}/archive")
    agent_output = posixpath.join("/tmp", "agent_output")
    cleared = agent.container.exec_run(
        ["sh", "-lc", f"rm -rf {agent_output} && mkdir -p {agent_output}"],
        workdir="/",
    )
    log_container_output(cleared)
    history_file = posixpath.join(agent_output, "meta_agent_chat_history.md")
    command = _meta_agent_command(
        agent,
        history_path=history_path,
        output_path=agent_output,
        history_file=history_file,
        iterations_left=iterations_left,
    )
    execution = agent.container.exec_run(
        command,
        workdir=run_workdir or f"/{REPO_NAME}",
        environment=container_model_environment(agent.runtime.llm_environment),
    )
    metadata["meta_agent_command_successful"] = execution.exit_code == 0
    usage = _read_refinement_usage(agent, agent_output)
    if usage is not None:
        metadata["refinement_usage"] = usage
    if usage and usage.get("tool_budget_exhausted"):
        metadata["tool_budget_exhausted"] = True
    if execution.exit_code != 0:
        output = execution.output.decode("utf-8", errors="replace")
        raise RuntimeError(f"MetaAgent command failed: {output}")
    log_container_output(execution)
    transition = _collect_refinement(agent, resolved_output, agent_output, metadata)
    result = None
    if transition.changed:
        _checkpoint_agent(agent)
        if play is not None:
            result = evaluate_agent(agent, play, output_dir=resolved_output)
    return RefineOutput(
        agent=agent,
        changed=transition.changed,
        patch_path=transition.patch_path,
        result=result,
        metadata=transition.metadata,
    )


def _checkpoint_agent(agent: InstantiatedAgent) -> None:
    """Commit one accepted in-container refinement for the next round."""
    repo_path = f"/{REPO_NAME}"
    staged = agent.container.exec_run(["git", "add", "--all"], workdir=repo_path)
    log_container_output(staged)
    status = agent.container.exec_run(
        ["git", "status", "--porcelain"], workdir=repo_path
    )
    log_container_output(status)
    if status.output.decode("utf-8", errors="replace").strip():
        committed = agent.container.exec_run(
            [
                "git", "-c", "user.name=user", "-c", "user.email=you@example.com",
                "commit", "-m", f"Checkpoint {agent.snapshot.candidate_id}",
            ],
            workdir=repo_path,
        )
        log_container_output(committed)
    revision = agent.container.exec_run(
        ["git", "rev-parse", "HEAD"], workdir=repo_path
    )
    log_container_output(revision)
    agent.commit_hash = revision.output.decode("utf-8", errors="replace").strip()


def _stage_feedback_screenshots(result: dict[str, Any], local_history: Path) -> list[str]:
    """Bundle GameWorld feedback images without changing the original feedback."""

    staged: list[str] = []
    folder = local_history / "artifacts" / "bench_screenshots"
    for case_index, case in enumerate(result.get("cases", [])):
        trajectory = (case.get("feedback") or {}).get("trajectory") or []
        case_screenshots: list[dict[str, Any]] = []
        for step_index, step in enumerate(trajectory):
            for image_index, image in enumerate(step.get("image_records") or []):
                source_name = image.get("path")
                if not isinstance(source_name, str) or not source_name:
                    raise ValueError("Feedback screenshot has no source path")
                source = Path(source_name).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"Feedback screenshot is missing: {source}")
                if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    raise ValueError(f"Unsupported feedback image: {source}")
                if source.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError(f"Feedback image exceeds 20 MiB: {source}")
                folder.mkdir(parents=True, exist_ok=True)
                filename = f"case_{case_index:04d}_step_{step_index:04d}_image_{image_index:02d}{source.suffix.lower()}"
                target = folder / filename
                shutil.copyfile(source, target)
                relative = f"artifacts/bench_screenshots/{filename}"
                image["path"] = relative
                case_screenshots.append({"step_index": step.get("step_index"), "path": relative})
                staged.append(relative)
        if case_screenshots:
            case.setdefault("artifacts", {})["game_screenshots"] = case_screenshots
    return staged


def _stage_direct_evidence(
    agent: InstantiatedAgent,
    results: Any,
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
    evidence_source_dir: str | Path | None = None,
) -> str:
    local_history = output_dir / "evidence"
    if local_history.exists():
        shutil.rmtree(local_history)
    local_history.mkdir(parents=True, exist_ok=True)
    artifact_refs = results.get("artifacts", {}) if isinstance(results, Mapping) else {}
    source_root = Path(evidence_source_dir).resolve() if evidence_source_dir else None
    container_history = "/tmp/evaluation_history"
    if source_root is not None:
        original_result = source_root / "result.json"
        if not original_result.is_file():
            raise FileNotFoundError(f"Evidence result not found: {original_result}")
        staged_result = json.loads(original_result.read_text(encoding="utf-8"))
    else:
        staged_result = json.loads(json.dumps(results, ensure_ascii=False, default=str))
        if artifact_refs:
            raise ValueError("Artifact-bearing evidence requires evidence_source_dir.")
    screenshots = _stage_feedback_screenshots(staged_result, local_history)
    source_result = local_history / "result.json"
    source_result.write_text(json.dumps(staged_result, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "evidence_kind": "rsi_evaluation_evidence",
        "metadata": dict(metadata),
        "result_schema": {
            "selection_score": "Scalar benchmark score; higher is better.",
            "cases": "List of per-case results; each case carries its own metrics, feedback, and artifacts.",
            "case_id": "Stable identifier for one evaluated case.",
        },
        "artifact_resolution": (
            "During MetaAgent execution, artifact paths in result.json are relative "
            "to /tmp/evaluation_history. Game screenshots are copied to "
            "artifacts/bench_screenshots; a bounded sample is attached directly "
            "to the MetaAgent editing request."
        ),
        "game_screenshot_count": len(screenshots),
        "canonical_source": (
            os.path.relpath(source_root, agent.snapshot.run_dir) if source_root else None
        ),
    }
    (local_history / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    source_artifacts = source_root / "artifacts" if source_root else None
    if source_artifacts is not None:
        if artifact_refs and not source_artifacts.is_dir():
            raise FileNotFoundError(f"Evidence artifacts not found: {source_artifacts}")
    cleared = agent.container.exec_run(["rm", "-rf", container_history], workdir="/")
    log_container_output(cleared)
    created = agent.container.exec_run(["mkdir", "-p", container_history], workdir="/")
    log_container_output(created)
    copy_to_container(agent.container, source_result, f"{container_history}/result.json")
    copy_to_container(
        agent.container,
        local_history / "evidence_manifest.json",
        f"{container_history}/evidence_manifest.json",
    )
    if source_artifacts is not None and source_artifacts.is_dir():
        copy_to_container(
            agent.container, source_artifacts, f"{container_history}/artifacts"
        )
    staged_artifacts = local_history / "artifacts"
    if staged_artifacts.is_dir():
        copy_to_container(agent.container, staged_artifacts, f"{container_history}/artifacts")
    return container_history


_USAGE_RECORDING_BOOTSTRAP = """
import json
import runpy
import sys
from pathlib import Path
import meta_agent

usage_path = Path(sys.argv[1])
sys.argv = ["run_meta_agent.py", *sys.argv[2:]]
original_chat = meta_agent.chat_with_agent

def tracked_chat(*args, **kwargs):
    history = original_chat(*args, **kwargs)
    last = history[-1] if history else {}
    usage = dict(last.get("usage") or {})
    if last.get("tool_budget_exhausted"):
        usage["tool_budget_exhausted"] = True
        usage["tool_calls_executed"] = last["tool_calls_executed"]
        usage["tool_call_limit"] = last["tool_call_limit"]
    if usage.get("total_tokens", 0) > 0 or usage.get("tool_budget_exhausted"):
        usage_path.write_text(json.dumps(usage), encoding="utf-8")
    return history

meta_agent.chat_with_agent = tracked_chat
runpy.run_path("run_meta_agent.py", run_name="__main__")
"""


def _meta_agent_command(
    agent: InstantiatedAgent,
    *,
    history_path: str,
    output_path: str,
    history_file: str,
    iterations_left: int,
) -> list[str]:
    model = agent.runtime.meta_model
    command = [
        "timeout", "21600", "python", "-c", _USAGE_RECORDING_BOOTSTRAP,
        posixpath.join(output_path, "meta_agent_usage.json"),
        "--model", model["name"],
    ]
    if model.get("reasoning_effort"):
        command.extend(["--reasoning_effort", model["reasoning_effort"]])
    command.extend([
        "--backend", model["backend"],
        "--api_mode", model["api_mode"],
        "--api_key_env", model["api_key_env"],
        "--base_url_env", model["base_url_env"],
        "--chat_history_file", history_file,
        "--repo_path", f"/{REPO_NAME}/",
        "--evals_folder", history_path,
        "--git_dir", f"/{REPO_NAME}",
        "--base_commit", agent.commit_hash,
        "--outdir", output_path,
        "--iterations_left", str(iterations_left),
    ])
    return command


def _collect_refinement(
    agent: InstantiatedAgent,
    output_dir: Path,
    container_output: str,
    metadata: dict[str, Any],
) -> RefineOutput:
    usage = _read_refinement_usage(agent, container_output)
    if usage is not None:
        metadata["refinement_usage"] = usage
    local_output = output_dir / "agent_output"
    copy_from_container(
        agent.container,
        source_path=container_output,
        dest_path=str(local_output),
    )
    patch_path = local_output / "model_patch.diff"
    changed = file_exist_and_not_empty(str(patch_path))
    metadata["patch_produced"] = changed
    metadata["run_eval"] = changed
    if changed:
        _validate_patch_scope(patch_path)
        check_candidate_compilation(agent.container)
        metadata.setdefault("curr_patch_files", [])
        metadata["curr_patch_files"] += patch_paths_for_metadata(
            str(output_dir), [str(patch_path)]
        )
        candidate_id = metadata.get("current_genid", agent.snapshot.candidate_id)
        parent_id = agent.snapshot.candidate_id
        agent.snapshot = with_lineage(
            agent.snapshot,
            candidate_id=candidate_id,
            parent_id=parent_id,
            lineage_patches=[*agent.snapshot.lineage_patches, patch_path],
        )
        metadata["change"] = describe_patch(str(patch_path), str(output_dir))
        metadata["compilation_successful"] = True
        metadata["refine_successful"] = True
    else:
        metadata["error"] = "MetaAgent completed without producing a code diff."
    return RefineOutput(
        agent=agent,
        changed=changed,
        patch_path=patch_path if changed else None,
        result=None,
        metadata=metadata,
    )


def _read_refinement_usage(
    agent: InstantiatedAgent, container_output: str,
) -> dict[str, Any] | None:
    path = posixpath.join(container_output, "meta_agent_usage.json")
    result = agent.container.exec_run(["cat", path], workdir="/")
    if result.exit_code != 0:
        return None
    try:
        raw = json.loads(result.output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = {
        "prompt_num_tokens": "input_tokens",
        "response_num_tokens": "output_tokens",
        "reasoning_num_tokens": "reasoning_tokens",
        "total_num_tokens": "total_tokens",
        "cached_prompt_num_tokens": "cached_input_tokens",
    }
    if not isinstance(raw, dict):
        return None
    valid_usage = all(
        type(raw.get(key)) is int and raw[key] >= 0 for key in fields.values()
    ) and raw.get("total_tokens", 0) > 0
    exhausted = raw.get("tool_budget_exhausted") is True
    if not valid_usage and not exhausted:
        return None
    usage = {name: raw[key] for name, key in fields.items()} if valid_usage else {}
    if exhausted:
        usage["tool_budget_exhausted"] = True
        usage["tool_calls_executed"] = raw.get("tool_calls_executed")
        usage["tool_call_limit"] = raw.get("tool_call_limit")
    return usage


def _validate_patch_scope(patch_path: Path) -> None:
    """Prevent candidate refinement from modifying backend infrastructure."""
    protected_paths = ("benchmark/backend/", "benchmark/contracts.py")
    for line in patch_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("--- a/", "+++ b/")):
            continue
        changed_path = line[6:]
        if changed_path.startswith(protected_paths):
            raise RuntimeError(
                f"MetaAgent patch modifies protected framework code: {changed_path}"
            )


def finalize_generation(
    output_dir: str | Path,
    metadata: dict[str, Any],
    result: Any,
) -> None:
    metadata["evaluation_successful"] = result_is_valid(result)
    metadata["valid_parent"] = metadata["run_eval"] and metadata["evaluation_successful"]
    path = Path(output_dir) / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_generation_log(output_dir: str | Path) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    setup_logger(str(path / "generate.log"))
    return path
