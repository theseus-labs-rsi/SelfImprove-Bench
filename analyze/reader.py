"""Load and organize the raw improvement artifacts of each RSI method."""

import difflib
import filecmp
import json
import math
import warnings
from pathlib import Path
from typing import Any


# ===== Helpers =====

def _read_utf8_text(path: str | Path) -> str:
    """Read UTF-8 text and include the file location on encoding errors."""
    source = Path(path)
    try:
        return source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Cannot read file as UTF-8: {source}") from exc


def _read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object and reject unexpected top-level types."""
    source = Path(path)
    try:
        content = json.loads(_read_utf8_text(source))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {source}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"JSON top level must be an object: {source}")
    return content


def _workspace_files(root: Path) -> dict[Path, Path]:
    """List workspace files, excluding Git metadata bundled with candidates."""
    return {
        path.relative_to(root): path
        for path in root.rglob("*")
        if path.is_file()
        and not {".git", "__pycache__"}.intersection(path.relative_to(root).parts)
        and path.suffix.lower() not in {".pyc", ".pyo"}
    }


def _read_workspace_text(path: Path | None) -> str:
    """Read a workspace file; a missing path indicates an added or deleted file."""
    return "" if path is None else _read_utf8_text(path)


def _read_ahe_workspace_diff(candidate: Path) -> str:
    """Compare the same-round baseline and AHE candidate workspace without relying on Git history."""
    iteration = candidate.parent.name
    state = candidate.parents[2]
    baseline = state / "runs" / iteration / "input" / "workspace"
    evolved = candidate / "workspace"
    for root in (baseline, evolved):
        if not root.is_dir():
            raise FileNotFoundError(f"AHE workspace does not exist: {root}")

    old_files = _workspace_files(baseline)
    new_files = _workspace_files(evolved)
    chunks: list[str] = []
    for relative in sorted(old_files.keys() | new_files.keys()):
        old_path, new_path = old_files.get(relative), new_files.get(relative)
        if old_path and new_path and filecmp.cmp(old_path, new_path, shallow=False):
            continue
        from_name = f"a/{relative.as_posix()}" if old_path else "/dev/null"
        to_name = f"b/{relative.as_posix()}" if new_path else "/dev/null"
        try:
            old_text = _read_workspace_text(old_path)
            new_text = _read_workspace_text(new_path)
        except ValueError:
            chunks.append(f"Binary files {from_name} and {to_name} differ\n")
            continue
        chunks.extend(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=from_name,
            tofile=to_name,
        ))
    return "".join(chunks)


# ===== Single-artifact readers =====

def read_ace_operations(path: str | Path) -> list[dict[str, Any]]:
    """Read the raw operation records from the given ACE JSONL file."""
    source = Path(path)
    records: list[dict[str, Any]] = []

    # Parse line by line without rewriting or normalizing record fields.
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                operation = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid ACE JSONL at {source}:{line_number}") from exc
            if not isinstance(operation, dict):
                raise ValueError(f"ACE operation must be an object at {source}:{line_number}")
            records.append(operation)

    return records


def _read_ace_curator_operations(task_dir: Path) -> list[dict[str, Any]] | None:
    """Read this round's operations from the trace task's curator response; return None if the file is missing."""
    sources = sorted(task_dir.glob("curator_refine_*.json"))
    if not sources:
        return None
    if len(sources) != 1:
        raise ValueError(f"ACE curator response file is not unique: {task_dir}")

    record = _read_json_object(sources[0])
    response = record.get("response")
    if not isinstance(response, str):
        raise ValueError(f"ACE curator response must be a string: {sources[0]}")
    try:
        content = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ACE curator response is not valid JSON: {sources[0]}") from exc
    operations = content.get("operations") if isinstance(content, dict) else None
    if not isinstance(operations, list) or not all(isinstance(item, dict) for item in operations):
        raise ValueError(f"ACE curator operations must be a list of objects: {sources[0]}")
    return operations


def read_hyperagent_patch(path: str | Path) -> str:
    """Read the raw patch from the given HyperAgent model_patch.diff."""
    return _read_utf8_text(path)


def read_ahe_candidate(candidate_dir: str | Path) -> dict[str, str]:
    """Read the summary and detailed workspace diff of the given AHE candidate."""
    candidate = Path(candidate_dir)
    return {
        "evolve_result": _read_utf8_text(candidate / "evolve_result.json"),
        "workspace_diff": _read_ahe_workspace_diff(candidate),
    }


# ===== Method-level organizers =====

def _score_fields(
    report: dict[str, Any] | None, candidate_id: str | None = None,
) -> dict[str, Any]:
    """Read the actual parent and candidate scores; keep missing or non-finite values empty."""
    report = report or {}
    parent = report.get("parent_score")
    scores = report.get("candidate_scores", {})
    current = scores.get(candidate_id) if isinstance(scores, dict) else None
    if "candidate_score" in report and (
        candidate_id is None or report.get("selected_candidate_id") == candidate_id
    ):
        current = report["candidate_score"]
    valid = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    parent = parent if valid(parent) else None
    current = current if valid(current) else None
    return {
        "parent_score": parent, "current_score": current,
        "score_delta": current - parent if current is not None and parent is not None else None,
    }


def collect_ace_scores(experiment_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read per-round selection results by task without changing the ACE artifact reading interface."""
    root = Path(experiment_dir).resolve()
    tasks = root.glob("task_*") if (root / "batch_manifest.json").is_file() else root.glob("rsi/ace/*/task_*")
    return {task.name: _score_fields(_read_hyper_selection(task)) for task in tasks if task.is_dir()}


def _evaluation_scores(
    candidate_dir: Path, root: Path, fallback: dict[str, Any],
) -> dict[str, Any]:
    """Prefer the candidate result and match the same-round parent result by parent_run_id."""
    path = candidate_dir / "evaluation" / "result.json"
    if not path.is_file():
        return fallback
    result = _read_json_object(path)
    current = result.get("relative_score") if result.get("status") == "ok" else None
    parent = fallback["parent_score"]
    run_id = result.get("parent_run_id")
    if isinstance(run_id, str):
        matches = [path for path in (root / "runs" / "train").glob("*.json") if path.stem.partition("_")[2] == run_id]
        if len(matches) > 1:
            raise ValueError(f"Parent evaluation result is not unique: {run_id}")
        if matches:
            record = _read_json_object(matches[0])
            parent = record.get("relative_score") if record.get("status") == "ok" else None
    return _score_fields({"parent_score": parent, "candidate_score": current})

def collect_ace_artifacts(
    experiment_dir: str | Path,
) -> list[tuple[str, list[dict[str, Any]] | None]]:
    """Read the raw improvement records of each task from the ACE experiment directory."""
    source_root = Path(experiment_dir).resolve()
    if (source_root / "batch_manifest.json").is_file():
        task_dirs = sorted(path for path in source_root.iterdir() if _is_task_dir(path))
        if not task_dirs:
            raise FileNotFoundError(f"ACE trace task directory does not exist: {source_root}")
        return [
            (task_dir.name, _read_ace_curator_operations(task_dir))
            for task_dir in task_dirs
        ]

    ace_root = source_root / "rsi" / "ace"
    if not ace_root.is_dir():
        raise FileNotFoundError(f"ACE run directory does not exist: {ace_root}")

    run_dirs = sorted(path for path in ace_root.iterdir() if path.is_dir())
    if len(run_dirs) != 1:
        names = [path.name for path in run_dirs]
        raise ValueError(f"ACE run directory must be unique, found {len(run_dirs)}: {names}")

    task_dirs = sorted(
        path for path in run_dirs[0].iterdir()
        if path.is_dir() and path.name.startswith("task_")
    )
    if not task_dirs:
        raise FileNotFoundError(f"ACE task directory does not exist: {run_dirs[0]}")

    artifacts: list[tuple[str, list[dict[str, Any]] | None]] = []
    for task_dir in task_dirs:
        candidates = task_dir / "candidates"
        sources = sorted(candidates.glob("*/curator_operations_diff.jsonl"))
        legacy = candidates / "curator_operations_diff.jsonl"
        if legacy.is_file():
            sources.append(legacy)
        if len(sources) > 1:
            raise ValueError(f"ACE operations file is not unique: {task_dir}")
        content = read_ace_operations(sources[0]) if sources else None
        artifacts.append((task_dir.name, content))
    return artifacts


def collect_hyperagent_artifacts(
    experiment_dir: str | Path,
) -> list[dict[str, Any]]:
    """Read the single valid HyperAgent run and order candidates by explicit parent_id."""
    return _collect_hyper_tree_artifacts(
        experiment_dir, method_dir="hyperagents", method_name="HyperAgent",
    )


def _collect_hyper_tree_artifacts(
    experiment_dir: str | Path,
    *,
    method_dir: str,
    method_name: str,
) -> list[dict[str, Any]]:
    """Read a HyperAgent-style candidate tree from a method directory."""
    source_root = Path(experiment_dir).resolve()
    if (source_root / "batch_manifest.json").is_file():
        return _read_hyper_trace_batch(source_root, method_name)

    root = source_root / "rsi" / method_dir
    if not root.is_dir():
        raise FileNotFoundError(f"{method_name} run directory does not exist: {root}")

    run_dirs = [path for path in root.iterdir() if _has_hyper_tasks(path)]
    if len(run_dirs) != 1:
        names = [path.name for path in run_dirs]
        raise ValueError(f"{method_name} must have a unique valid run, found {len(run_dirs)}: {names}")

    nodes: list[dict[str, Any]] = []
    task_dirs = sorted(path for path in run_dirs[0].iterdir() if _is_task_dir(path))
    for task_dir in task_dirs:
        nodes.extend(_read_hyper_task(task_dir, run_dirs[0].name, method_name))
    return _order_hyper_nodes(nodes, method_name)


def _read_hyper_trace_batch(root: Path, method_name: str) -> list[dict[str, Any]]:
    """Rebuild the parent-child chain from a HyperAgent trace batch with a single selected candidate."""
    manifest = _read_json_object(root / "batch_manifest.json")
    run_id = manifest.get("batch_id")
    if not isinstance(run_id, str):
        raise ValueError(f"{method_name} batch is missing batch_id: {root}")

    nodes: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in root.iterdir() if _is_task_dir(path)):
        report = _read_json_object(task_dir / "refinement_report.json")
        metadata = report.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid HyperAgent refinement metadata: {task_dir}")
        nodes.append(_build_hyper_trace_node(task_dir, run_id, report, metadata))
    if not nodes:
        raise FileNotFoundError(f"{method_name} trace task directory does not exist: {root}")
    return _order_hyper_nodes(nodes, method_name)


def _build_hyper_trace_node(
    task_dir: Path,
    run_id: str,
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Convert a single-candidate task report into a unified HyperAgent node."""
    candidate_id = report.get("candidate_id")
    raw_parent = metadata.get("parent_genid")
    if not isinstance(candidate_id, str):
        raise ValueError(f"HyperAgent trace is missing candidate_id: {task_dir}")
    if raw_parent not in (None, "initial") and not isinstance(raw_parent, str):
        raise ValueError(f"Invalid HyperAgent trace parent_genid: {candidate_id}")

    patch_path = task_dir / "agent_output" / "model_patch.diff"
    accepted = metadata.get("refine_successful")
    return {
        "node_id": candidate_id,
        **_evaluation_scores(task_dir / "agent_output", task_dir.parent,
                             _score_fields(_read_hyper_selection(task_dir), candidate_id)),
        "parent_id": None if raw_parent in (None, "initial") else raw_parent,
        "task_id": task_dir.name,
        "run_id": run_id,
        "changed": bool(report.get("changed")),
        "selected": True,
        "accepted": accepted if isinstance(accepted, bool) else None,
        "artifact_path": str(patch_path),
        "artifact": read_hyperagent_patch(patch_path) if patch_path.is_file() else None,
    }


def _is_task_dir(path: Path) -> bool:
    """Return whether a directory is a method-generated task directory."""
    return path.is_dir() and path.name.startswith("task_")


def _has_hyper_tasks(path: Path) -> bool:
    """Empty runs are excluded from the uniqueness check."""
    return path.is_dir() and any(_is_task_dir(child) for child in path.iterdir())


def _read_hyper_task(
    task_dir: Path,
    run_id: str,
    method_name: str,
) -> list[dict[str, Any]]:
    """Read all candidates and their selection state within one task."""
    batch_path = task_dir / "candidate_batch.json"
    if not batch_path.is_file():
        warnings.warn(
            f"Skipping {method_name} task missing candidate_batch.json: {task_dir}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    batch = _read_json_object(batch_path)
    candidates = batch.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"HyperAgent candidates must be a list: {batch_path}")

    selection = _read_hyper_selection(task_dir)
    return [
        _build_hyper_node(task_dir, run_id, candidate, selection)
        for candidate in candidates
    ]


def _read_hyper_selection(task_dir: Path) -> dict[str, Any] | None:
    """Read the task's candidate selection result; leave the state unknown when missing."""
    paths = sorted(task_dir.glob("selection_stage_*/selection.json"))
    if len(paths) > 1:
        raise ValueError(f"HyperAgent selection file is not unique: {task_dir}")
    return _read_json_object(paths[0]) if paths else None


def _build_hyper_node(
    task_dir: Path,
    run_id: str,
    candidate: Any,
    selection: dict[str, Any] | None,
) -> dict[str, Any]:
    """Rebuild candidate paths from local directories to avoid stale absolute paths in metadata."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
        raise ValueError(f"HyperAgent candidate is missing candidate_id: {task_dir}")
    candidate_id = candidate["candidate_id"]
    raw_parent = candidate.get("parent_id")
    parent_id = None if raw_parent in (None, "initial") else raw_parent
    if parent_id is not None and not isinstance(parent_id, str):
        raise ValueError(f"Invalid HyperAgent parent_id: {candidate_id}")

    patch_path = task_dir / "candidates" / candidate_id / "agent_output" / "model_patch.diff"
    selected_id = selection.get("selected_candidate_id") if selection else None
    selected = candidate_id == selected_id if selected_id else None
    accepted = selection.get("accepted") if selected is True else (False if selected_id else None)
    return {
        "node_id": candidate_id,
        **_evaluation_scores(patch_path.parent, task_dir.parents[3],
                             _score_fields(selection, candidate_id)),
        "parent_id": parent_id,
        "task_id": task_dir.name,
        "run_id": run_id,
        "changed": bool(candidate.get("changed")),
        "selected": selected,
        "accepted": accepted,
        "artifact_path": str(patch_path),
        "artifact": read_hyperagent_patch(patch_path) if patch_path.is_file() else None,
    }


def _order_hyper_nodes(
    nodes: list[dict[str, Any]],
    method_name: str = "HyperAgent",
) -> list[dict[str, Any]]:
    """Topologically sort by parent_id while checking for duplicates, missing parents, and cycles."""
    by_id = {node["node_id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError(f"{method_name} contains duplicate candidate_id")
    ordered: list[dict[str, Any]] = []
    states: dict[str, int] = {}

    def visit(node_id: str) -> None:
        state = states.get(node_id, 0)
        if state == 1:
            raise ValueError(f"{method_name} parent_id forms a cycle: {node_id}")
        if state == 2:
            return
        states[node_id] = 1
        parent_id = by_id[node_id]["parent_id"]
        if parent_id is not None:
            if parent_id not in by_id:
                raise ValueError(f"{method_name} parent node does not exist: {node_id} -> {parent_id}")
            visit(parent_id)
        states[node_id] = 2
        ordered.append(by_id[node_id])

    for node in nodes:
        visit(node["node_id"])
    return ordered


def collect_ahe_artifacts(
    experiment_dir: str | Path,
) -> list[dict[str, Any]]:
    """Read AHE candidates and follow the harness retained by each selection."""
    root = Path(experiment_dir).resolve() / "rsi" / "ahe"
    if not root.is_dir():
        raise FileNotFoundError(f"AHE run directory does not exist: {root}")

    run_dirs = [path for path in root.iterdir() if _has_ahe_candidates(path)]
    if len(run_dirs) != 1:
        names = [path.name for path in run_dirs]
        raise ValueError(f"AHE must have a unique valid run, found {len(run_dirs)}: {names}")

    candidates_root = run_dirs[0] / "state" / "candidates"
    iteration_dirs = sorted(
        path for path in candidates_root.iterdir()
        if path.is_dir() and path.name.startswith("iteration_")
    )
    if not iteration_dirs:
        raise FileNotFoundError(f"AHE iteration directory does not exist: {candidates_root}")

    nodes: list[dict[str, Any]] = []
    parent_id: str | None = None
    reports = {}
    for path in run_dirs[0].glob("task_*/selection_stage_*/ahe_selection.json"):
        report = _read_json_object(path)
        reports.setdefault(report.get("refinement_id"), []).append(report)
    for iteration_dir in iteration_dirs:
        variant_dir = _single_ahe_variant(iteration_dir)
        node_id = f"{iteration_dir.name}:{variant_dir.name}"
        matching = reports.get(iteration_dir.name, [])
        if not matching:
            raise ValueError(
                f"AHE selection result is missing; cannot reconstruct retained parents: {iteration_dir.name}"
            )
        if len(matching) > 1:
            raise ValueError(f"AHE score result is not unique: {iteration_dir.name}")
        report = matching[0]
        status = report.get("status")
        committed_id = report.get("committed_candidate_id")
        if status == "committed" and committed_id == variant_dir.name:
            accepted = True
        elif status == "unchanged" and committed_id is None:
            accepted = False
        else:
            raise ValueError(f"Invalid AHE selection result: {iteration_dir.name}")
        nodes.append({
            "node_id": node_id,
            **_evaluation_scores(variant_dir, Path(experiment_dir).resolve(),
                                 _score_fields(report, variant_dir.name)),
            "parent_id": parent_id,
            "accepted": accepted,
            "selection_status": status,
            "iteration": iteration_dir.name,
            "variant": variant_dir.name,
            "run_id": run_dirs[0].name,
            "artifact_dir": str(variant_dir),
            "artifact": read_ahe_candidate(variant_dir),
        })
        if accepted:
            parent_id = node_id
    return nodes


def _has_ahe_candidates(path: Path) -> bool:
    """Return whether the run contains an AHE candidates directory."""
    candidates = path / "state" / "candidates"
    return candidates.is_dir() and any(candidates.iterdir())


def _single_ahe_variant(iteration_dir: Path) -> Path:
    """Require the single-variant layout used by the published AHE configurations."""
    variants = sorted(
        path for path in iteration_dir.iterdir()
        if path.is_dir() and path.name.startswith("variant_")
    )
    if len(variants) != 1:
        names = [path.name for path in variants]
        raise ValueError(
            f"AHE iteration is missing a unique variant; cannot determine the parent-child chain: {iteration_dir} -> {names}"
        )
    return variants[0]
