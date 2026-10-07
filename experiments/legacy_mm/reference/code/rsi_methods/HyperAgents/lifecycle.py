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
            "Docker daemon is unavailable. Start Docker Desktop and retry."
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
    container.start()
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


def reconstruct_agent(
    agent: InstantiatedAgent,
    snapshot: CandidateSnapshot,
    *,
    artifact_dir: str | Path | None = None,
) -> InstantiatedAgent:
    """Deterministically load a candidate into an existing container."""
    repo_path = f"/{REPO_NAME}"
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
    agent.commit_hash = apply_diffs_checked(
        agent.container, list(snapshot.lineage_patches)
    )
    from .runtime_safety import install_runtime_safety
    agent.commit_hash = install_runtime_safety(agent.container, agent.commit_hash, repo_path)
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
    # The candidate repository is bind-mounted into Docker. Put a SECOND,
    # append-only ledger inside its .git directory so the host budget watcher can
    # see every paid MetaAgent response while exec_run is still blocking.
    live_name = f"hyperagents_meta_{uuid.uuid4().hex}.jsonl"
    live_path = f"/{REPO_NAME}/.git/{live_name}"
    live_host = agent.snapshot.workspace / ".git" / live_name
    if not live_host.parent.is_dir():
        raise RuntimeError("HyperAgents live ledger repository is missing")
    env = dict(agent.runtime.llm_environment)
    env["HYPERAGENTS_META_LIVE_LEDGER"] = live_path
    env["HYPERAGENTS_META_MAX_TOKENS"] = os.environ.get("HYPERAGENTS_META_MAX_TOKENS", "360000")
    if not 0 < int(env["HYPERAGENTS_META_MAX_TOKENS"]) <= 500000:
        raise ValueError("HyperAgents MetaAgent token cap must be 1..500000")
    metadata["meta_live_ledger"] = str(live_host)
    metadata["meta_token_cap"] = int(env["HYPERAGENTS_META_MAX_TOKENS"])
    execution = agent.container.exec_run(
        command,
        workdir=run_workdir or f"/{REPO_NAME}",
        environment=env,
    )
    metadata["meta_agent_command_successful"] = execution.exit_code == 0
    if execution.exit_code != 0:
        # Preserve the per-call ledger even when the MetaAgent crashes after
        # receiving billable model responses. This is evidence, not a score.
        try:
            copy_from_container(
                agent.container, source_path=agent_output,
                dest_path=str(resolved_output / "agent_output_failed"),
            )
        except Exception:
            pass  # keep the original exception; usage completeness remains unknown
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
            "selection_score": "Normalized scalar in [0, 1]; higher is better.",
            "cases": "List of per-case results; each case carries its own metrics, feedback, and artifacts.",
            "case_id": "Stable identifier for one evaluated case.",
        },
        "analysis_instruction": (
            "Inspect result.json and its cases and artifacts. Infer recurring "
            "failure patterns across cases; do not optimize for one case_id or "
            "one stochastic trace."
        ),
        "artifact_resolution": (
            "During MetaAgent execution, artifact paths in result.json are relative "
            "to /tmp/evaluation_history. Game screenshots are copied to "
            "artifacts/bench_screenshots and also passed to a multimodal inspection call."
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
    if screenshots:
        created = agent.container.exec_run(
            ["mkdir", "-p", f"{container_history}/artifacts"], workdir="/"
        )
        if created.exit_code != 0:
            raise RuntimeError("Cannot prepare screenshot evidence directory in container")
        copy_to_container(
            agent.container, local_history / "artifacts" / "bench_screenshots",
            f"{container_history}/artifacts/bench_screenshots",
        )
    return container_history


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
        "timeout", "21600", "python", "run_meta_agent.py",
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
