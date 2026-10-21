"""Method-owned checkpoints for the three bundled RSI adapters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from benchmark.interfaces import MethodContext
from benchmark.resume_state import atomic_json


class MethodCheckpointMixin:
    """Expose public persistence hooks without benchmark access to method internals."""

    checkpoint_method: str

    def bind_context(self, context: MethodContext | None) -> None:
        if context is not None and context.options:
            raise ValueError("Bundled adapters use their config.toml; adapter_options must be empty")
        self._method_context = context

    def export_state(self) -> dict:
        if self._method_context is None:
            raise RuntimeError("Checkpoint hooks require a MethodContext")
        return export_state(self, self.checkpoint_method)

    def restore_state(self, state: dict) -> None:
        if self._method_context is None:
            raise RuntimeError("Checkpoint hooks require a MethodContext")
        restore_state(self, self.checkpoint_method, self._method_context.artifact_dir, state)


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


def export_state(adaptor: object, method: str) -> dict:
    state: dict = {}
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
                     evolution_history=_optional_json(adaptor.run_dir / "evolution_history.json") or [],
                     workspace_config_digest=_file_digest(adaptor.state.current_workspace / "code_agent.yaml"),
                     evolve_config_digest=_file_digest(adaptor.state.evolve_agent_dir / "evolve_agent.yaml"))
    elif method == "hyperagents":
        from rsi_methods.HyperAgents.HyperAgentsAdaptor import _snapshot_to_dict
        snapshot = _snapshot_to_dict(adaptor.snapshot)
        snapshot["patch_digests"] = [_file_digest(Path(p)) for p in snapshot["lineage_patches"]]
        state.update(workspace_dir=str(adaptor.workspace_dir), snapshot=snapshot,
                     active_snapshot=_optional_json(adaptor.workspace_dir / "active_snapshot.json"),
                     workspace_selection=_optional_json(adaptor.workspace_dir / "selection.json"),
                     refinement_round=adaptor._refinement_round,
                     search_archive=adaptor._archive.to_dict())
    else:
        raise RuntimeError(f"Resume unsupported for RSI method {method}")
    return state


def restore_state(adaptor: object, method: str, root: Path, state: dict) -> None:
    if method == "ahe":
        adaptor._configure(adaptor._load_config())
    if method == "ahe" and state.get("ahe_policy") != adaptor.checkpoint_policy():
        raise RuntimeError(
            "AHE checkpoint predates or differs from the active-lineage policy; "
            "do not mix old and new evolution schedules in one lineage"
        )
    if method == "ace":
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
    elif method == "ahe":
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
        from rsi_methods.AHE.history import save_history
        save_history(run_dir, state.get("evolution_history") or [])
    elif method == "hyperagents":
        import tomllib
        from rsi_methods.HyperAgents.candidate import CandidateSnapshot
        from rsi_methods.HyperAgents.search_archive import SearchArchive
        if "search_archive" not in state:
            raise RuntimeError("HyperAgents checkpoint predates archive restoration; use the historical code for that run")
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
        adaptor._archive = SearchArchive.from_dict(state["search_archive"], root=root)
        archived = adaptor._archive.nodes.get(str(adaptor.snapshot.candidate_id))
        if archived is None or archived["snapshot"] != adaptor.snapshot:
            raise RuntimeError("HyperAgents active snapshot does not match its archive")
        adaptor._persist_search(adaptor.workspace_dir)
    else:
        raise RuntimeError(f"Resume unsupported for RSI method {method}")
