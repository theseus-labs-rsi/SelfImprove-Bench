"""Durable, conservative RSI state checkpoints for benchmark resume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
from .checkpoint_io import sync_directory


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _path(raw: str, root: Path, *, directory: bool = False) -> Path:
    path = Path(raw).resolve()
    if not path.is_relative_to(root.resolve()) or not (path.is_dir() if directory else path.is_file()):
        raise RuntimeError(f"Missing/unsafe resume artifact: {path}")
    return path


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_json(path: Path) -> object | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _restore_optional_json(path: Path, value: object | None) -> None:
    if value is None:
        path.unlink(missing_ok=True)
    else:
        atomic_json(path, value)


def export(session: object) -> dict:
    method, adaptor = session.method, session.adaptor
    state: dict = {"method": method, "batch_id": session.batch_id,
                   "selection_stage": session._selection_stage}
    if method == "ace":
        state.update(workspace=str(adaptor.workspace_dir), playbook=adaptor.playbook,
                     workspace_selection=_optional_json(adaptor.workspace_dir / "selection.json"),
                     version=adaptor._version, next_global_id=adaptor._next_global_id,
                     refinement_round=adaptor._refinement_round,
                     processed_feedback_count=adaptor._processed_feedback_count)
    elif method == "ahe":
        state.update(run_dir=str(adaptor.run_dir), workspace=str(adaptor.state.current_workspace),
                     refinement_round=adaptor._refinement_round,
                     observed_training_tasks=adaptor._observed_training_tasks,
                     ahe_policy=adaptor.checkpoint_policy(),
                     workspace_config_digest=_file_digest(adaptor.state.current_workspace / "code_agent.yaml"),
                     evolve_config_digest=_file_digest(adaptor.state.evolve_agent_dir / "evolve_agent.yaml"))
    elif method == "hyperagents":
        from rsi_methods.HyperAgents.HyperAgentsAdaptor import _snapshot_to_dict
        snapshot = _snapshot_to_dict(adaptor.snapshot)
        snapshot["patch_digests"] = [_file_digest(Path(p)) for p in snapshot["lineage_patches"]]
        state.update(workspace_dir=str(adaptor.workspace_dir), snapshot=snapshot,
                     active_snapshot=_optional_json(adaptor.workspace_dir / "active_snapshot.json"),
                     workspace_selection=_optional_json(adaptor.workspace_dir / "selection.json"),
                     refinement_round=adaptor._refinement_round)
    else:
        raise RuntimeError(f"Resume unsupported for RSI method {method}")
    return state


def restore(session: object, state: dict) -> None:
    if state.get("method") != session.method or state.get("batch_id") != session.batch_id:
        raise RuntimeError("Resume RSI session identity mismatch")
    adaptor, root = session.adaptor, session.artifact_dir
    if session.method == "ahe" and state.get("ahe_policy") != adaptor.checkpoint_policy():
        raise RuntimeError(
            "AHE checkpoint predates or differs from XBR-visual policy; "
            "do not mix old and new evolution schedules in one lineage"
        )
    session._selection_stage = int(state["selection_stage"])
    if session.method == "ace":
        # Configure from the original snapshot; never start a new lineage.
        import tomllib
        raw = (root / "method_config.toml").read_bytes()
        if raw != adaptor.config_path.read_bytes():
            raise RuntimeError("ACE method configuration changed since original run")
        adaptor._config_snapshot = raw
        adaptor._configure(tomllib.loads(raw.decode()))
        adaptor._validate_config()
        from rsi_methods.ACE.ACEAdaptor import Reflector, Curator
        adaptor._reflector = Reflector(adaptor.dispatcher, adaptor.reflector_backend_name,
                                       adaptor.reflector_model, adaptor.refine_max_output_tokens)
        adaptor._curator = Curator(adaptor.dispatcher, adaptor.curator_backend_name,
                                   adaptor.curator_model, adaptor.refine_max_output_tokens)
        adaptor.workspace_dir = _path(state["workspace"], adaptor.workspace_root, directory=True)
        adaptor._playbook = state["playbook"]
        adaptor._persist_playbook(adaptor.workspace_dir, adaptor._playbook)
        _restore_optional_json(adaptor.workspace_dir / "selection.json", state["workspace_selection"])
        adaptor._version = int(state["version"])
        adaptor._next_global_id = int(state["next_global_id"])
        adaptor._refinement_round = int(state["refinement_round"])
        adaptor._processed_feedback_count = int(state["processed_feedback_count"])
        adaptor._initialized = True
    elif session.method == "ahe":
        from rsi_methods.AHE.init import AHEState
        run_dir = _path(state["run_dir"], root, directory=True)
        workspace = _path(state["workspace"], run_dir, directory=True)
        if _file_digest(workspace / "code_agent.yaml") != state["workspace_config_digest"]:
            raise RuntimeError("AHE committed workspace changed since checkpoint")
        evolve_dir = _path(str(run_dir / "evolve_agent"), run_dir, directory=True)
        if _file_digest(evolve_dir / "evolve_agent.yaml") != state["evolve_config_digest"]:
            raise RuntimeError("AHE Evolve Agent config changed since checkpoint")
        observed = state.get("observed_training_tasks")
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise RuntimeError("Invalid AHE observed training task count in checkpoint")
        adaptor.state = AHEState(run_dir, workspace, evolve_dir)
        adaptor._refinement_round = int(state["refinement_round"])
        adaptor._observed_training_tasks = observed
    elif session.method == "hyperagents":
        import tomllib
        from rsi_methods.HyperAgents.candidate import CandidateSnapshot
        raw = (root / "method_config.toml").read_bytes()
        if raw != adaptor.config_path.read_bytes():
            raise RuntimeError("HyperAgents method configuration changed since original run")
        adaptor._config_snapshot = raw
        adaptor._configure(tomllib.loads(raw.decode()))
        adaptor.workspace_dir = _path(state["workspace_dir"], root, directory=True)
        _restore_optional_json(adaptor.workspace_dir / "selection.json", state["workspace_selection"])
        _restore_optional_json(adaptor.workspace_dir / "active_snapshot.json", state["active_snapshot"])
        data = state["snapshot"]
        patches = [_path(p, root) for p in data["lineage_patches"]]
        if [_file_digest(p) for p in patches] != data["patch_digests"]:
            raise RuntimeError("HyperAgents committed lineage patches changed")
        adaptor._snapshot = CandidateSnapshot(data["candidate_id"],
            _path(data["run_dir"], root, directory=True),
            _path(data["workspace"], root, directory=True), data["root_commit"],
            data["parent_id"], tuple(patches))
        adaptor._refinement_round = int(state["refinement_round"])
    else:
        raise RuntimeError(f"Resume unsupported for RSI method {session.method}")
