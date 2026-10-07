"""Runtime utilities for GameWorld."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import colorlog

from agents import BaseClient, create_client, get_config_for_model
from catalog import load_model
from runtime.runtime_config import RuntimeConfig

LOGGER = logging.getLogger(__name__)

MODEL_LOG_LEVEL = 25
GAME_LOG_LEVEL = 26
TASK_LOG_LEVEL = 27
ENV_LOG_LEVEL = 28

_CUSTOM_LOG_LEVELS = {
    "model": MODEL_LOG_LEVEL,
    "game": GAME_LOG_LEVEL,
    "task": TASK_LOG_LEVEL,
    "env": ENV_LOG_LEVEL,
}

_LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "white",
    "WARNING": "yellow",
    "ERROR": "red,bg_white",
    "CRITICAL": "red,bg_white",
    "MODEL": "blue",
    "GAME": "red",
    "TASK": "green",
    "ENV": "purple",
}

RERUN_POLICIES = ("auto", "never", "always")


@dataclass(frozen=True)
class RerunResolution:
    log_root: str | None
    skip: bool = False
    reason: str | None = None
    metadata: dict[str, Any] | None = None


def _build_level_logger(level: int):
    def _log(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    return _log


def _install_custom_logger_methods() -> None:
    for method_name, level in _CUSTOM_LOG_LEVELS.items():
        logging.addLevelName(level, method_name.upper())
        if not hasattr(logging.Logger, method_name):
            setattr(logging.Logger, method_name, _build_level_logger(level))


def _build_stream_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)s: %(message)s",
            log_colors=_LOG_COLORS,
        )
    )
    return handler


def setup_logging(level: int = logging.INFO) -> None:
    """Configure colorful logging with custom levels (MODEL, GAME, TASK, ENV)."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()
    root_logger.addHandler(_build_stream_handler())
    root_logger.setLevel(level)
    logging.getLogger("client").setLevel(logging.DEBUG)


def _default_run_dir(runtime_config: RuntimeConfig) -> Path:
    results_dir = Path(__file__).resolve().parent / "results"
    if len(runtime_config.model_ids) == 1:
        model_spec = runtime_config.model_ids[0]
    else:
        model_spec = "-".join(runtime_config.model_ids)
    run_name = (
        f"run_{runtime_config.log_session_id}_"
        f"{runtime_config.game_id}_{runtime_config.task_id}_{model_spec}"
    )
    return results_dir / run_name


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _last_task_evaluation(run_dir: Path) -> dict[str, Any] | None:
    for path in sorted(run_dir.glob("agent_*/interactions.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            evaluation = payload.get("task_evaluation")
            if isinstance(evaluation, dict):
                return evaluation
    return None


def _run_has_finalized_evaluation(run_dir: Path) -> bool:
    for summary_path in run_dir.glob("agent_*/evaluation/summary.json"):
        summary = _read_json_file(summary_path)
        if summary.get("finalized") is True:
            return True

    evaluation = _last_task_evaluation(run_dir)
    return bool(evaluation and evaluation.get("finalized") is True)


def _resolve_cmdline_path(value: str, cwd: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or cwd is None:
        return path.expanduser().resolve()
    return (cwd / path).expanduser().resolve()


def _process_uses_log_root(pid_dir: Path, target: Path) -> bool:
    try:
        raw_cmdline = (pid_dir / "cmdline").read_bytes()
    except Exception:
        return False
    if not raw_cmdline:
        return False

    args = [part.decode(errors="ignore") for part in raw_cmdline.split(b"\0") if part]
    try:
        cwd = (pid_dir / "cwd").resolve()
    except Exception:
        cwd = None

    for idx, arg in enumerate(args):
        if arg == "--log-root" and idx + 1 < len(args):
            try:
                if _resolve_cmdline_path(args[idx + 1], cwd) == target:
                    return True
            except Exception:
                continue
        if arg.startswith("--log-root="):
            try:
                if _resolve_cmdline_path(arg.split("=", 1)[1], cwd) == target:
                    return True
            except Exception:
                continue
    return False


def _run_has_live_process(run_dir: Path) -> bool:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return False

    target = run_dir.expanduser().resolve()
    current_pid = os.getpid()
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == current_pid:
            continue
        if _process_uses_log_root(pid_dir, target):
            return True
    return False


def _rerun_chain_root(source_dir: Path) -> Path:
    """Return the stable root directory for a ``*_rerunN`` chain."""
    match = re.match(r"^(?P<root>.+)_rerun(?P<index>[1-9][0-9]*)$", source_dir.name)
    if not match:
        return source_dir
    return source_dir.parent / match.group("root")


def _rerun_chain(root_dir: Path) -> list[tuple[int, Path]]:
    """List existing attempts in chronological rerun-index order."""
    attempts: list[tuple[int, Path]] = []
    if root_dir.exists():
        attempts.append((0, root_dir))
    pattern = re.compile(rf"^{re.escape(root_dir.name)}_rerun(?P<index>[1-9][0-9]*)$")
    for candidate in root_dir.parent.glob(f"{root_dir.name}_rerun*"):
        match = pattern.match(candidate.name)
        if match and candidate.is_dir():
            attempts.append((int(match.group("index")), candidate))
    return sorted(attempts, key=lambda item: item[0])


def _next_rerun_dir(source_dir: Path) -> tuple[Path, Path, int, list[Path]]:
    """Resolve the next target and the immediately preceding attempt.

    The old implementation always linked every rerun back to the base run.
    This keeps a true lineage instead: base -> rerun1 -> rerun2. Each attempt
    remains immutable and the game itself still starts from a fresh reset.
    """
    root_dir = _rerun_chain_root(source_dir)
    chain = _rerun_chain(root_dir)
    previous_dir = chain[-1][1] if chain else source_dir
    next_index = chain[-1][0] + 1 if chain else 1
    return (
        root_dir.parent / f"{root_dir.name}_rerun{next_index}",
        previous_dir,
        next_index,
        [path for _, path in chain],
    )


def resolve_rerun_log_root(
    log_root: str | None,
    *,
    rerun_policy: str,
    config_preset: str | None = None,
) -> RerunResolution:
    """Resolve the log root for a run, creating a rerun target when needed.

    This is not a persisted game-state restore. It is a safe rerun mechanism:
    incomplete historical artifacts stay immutable, and the new run records the
    source directory it replaced.
    """
    if rerun_policy not in RERUN_POLICIES:
        raise ValueError(f"Invalid rerun policy: {rerun_policy}")
    if log_root is None or rerun_policy == "never":
        return RerunResolution(log_root=log_root)

    requested_dir = Path(log_root).expanduser()
    if not requested_dir.exists():
        return RerunResolution(log_root=log_root)

    requested_dir = requested_dir.resolve()
    rerun_dir, source_dir, rerun_index, chain = _next_rerun_dir(requested_dir)
    source_meta_path = source_dir / "run_meta.json"
    source_meta = _read_json_file(source_meta_path)
    source_preset = source_meta.get("preset")
    if config_preset and source_preset and source_preset != config_preset:
        raise ValueError(
            "Refusing to attach rerun to a different experiment cell: "
            f"previous preset={source_preset!r}, requested preset={config_preset!r}. "
            "Use a new --log-root for a different model/task configuration."
        )
    source_status = source_meta.get("status")
    source_finalized = _run_has_finalized_evaluation(source_dir)
    last_eval = _last_task_evaluation(source_dir)

    if rerun_policy == "auto" and _run_has_live_process(source_dir):
        return RerunResolution(
            log_root=str(source_dir),
            skip=True,
            reason="source_run_is_still_active",
            metadata={
                "source_run": str(source_dir),
                "source_run_id": source_dir.name,
                "source_status": source_status,
                "source_finalized": source_finalized,
            },
        )

    if rerun_policy == "auto" and source_finalized:
        return RerunResolution(
            log_root=str(source_dir),
            skip=True,
            reason="source_run_already_finalized",
            metadata={
                "source_run": str(source_dir),
                "source_run_id": source_dir.name,
                "source_status": source_status,
                "source_finalized": source_finalized,
            },
        )

    return RerunResolution(
        log_root=str(rerun_dir),
        metadata={
            "rerun_policy": rerun_policy,
            "rerun_index": rerun_index,
            "rerun_reason": (
                "forced_rerun" if rerun_policy == "always" else "incomplete_previous_run"
            ),
            "source_run": str(source_dir),
            "source_run_id": source_dir.name,
            "chain_root_run": str(_rerun_chain_root(requested_dir)),
            "previous_run": str(source_dir),
            "prior_attempts": [str(path) for path in chain],
            "continuation_mode": "fresh_environment_linked_history",
            "source_status": source_status,
            "source_finalized": source_finalized,
            "source_last_step": last_eval.get("step") if isinstance(last_eval, dict) else None,
            "source_last_task_status": (
                last_eval.get("task_status") if isinstance(last_eval, dict) else None
            ),
        },
    )


def prepare_run_artifacts(
    runtime_config: RuntimeConfig,
    *,
    config_preset: str,
    port: int | None,
    log_root: str | None = None,
    rerun_metadata: dict[str, Any] | None = None,
) -> None:
    if log_root is not None:
        runtime_config.log_root = log_root

    run_dir = (
        Path(runtime_config.log_root)
        if runtime_config.log_root
        else _default_run_dir(runtime_config)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_config.log_root = str(run_dir)

    from tools.monitor import run_meta_path, write_run_meta

    meta_fields = dict(
        run_id=run_dir.name,
        preset=config_preset,
        game_id=runtime_config.game_id,
        task_id=runtime_config.task_id,
        model_spec=",".join(runtime_config.model_ids),
        port=port,
        session_id=runtime_config.log_session_id,
        return_code=None,
        ended_at=None,
        status="starting",
    )
    if rerun_metadata:
        meta_fields["rerun"] = rerun_metadata
    if not run_meta_path(run_dir).is_file():
        meta_fields["mode"] = "standalone"

    write_run_meta(run_dir, **meta_fields)


def mark_run_running(runtime_config: RuntimeConfig) -> None:
    if not runtime_config.log_root:
        return

    from tools.monitor import write_run_meta

    write_run_meta(runtime_config.log_root, status="running")


def finalize_run_metadata(
    runtime_config: RuntimeConfig,
    *,
    return_code: int | None,
    status: str,
) -> None:
    if not runtime_config.log_root:
        return

    from tools.monitor import write_run_meta

    write_run_meta(
        runtime_config.log_root,
        return_code=return_code,
        status=status,
        ended_at=datetime.now().isoformat(),
    )


def _validate_runtime_fields(
    runtime_config: RuntimeConfig,
    agent_ids: Sequence[str],
) -> None:
    expected = runtime_config.agent_count
    actual_counts = {
        "agent_ids": len(agent_ids),
        "model_ids": len(runtime_config.model_ids),
        "system_prompts": len(runtime_config.system_prompts),
        "enable_memory": len(runtime_config.enable_memory),
        "role_controls_maps": len(runtime_config.role_controls_maps),
        "semantic_controls_maps": len(runtime_config.semantic_controls_maps),
        "semantic_controls_specs": len(runtime_config.semantic_controls_specs),
    }
    mismatches = [
        f"{field}={count}"
        for field, count in actual_counts.items()
        if count != expected
    ]
    if mismatches:
        raise ValueError(
            f"RuntimeConfig.agent_count={expected} is inconsistent with runtime fields: "
            + ", ".join(mismatches)
        )


def _apply_model_profile_overrides(model_config: Any, config_overrides: dict[str, Any]) -> None:
    for key, value in config_overrides.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
            continue
        LOGGER.debug("Ignoring unknown model config override: %s", key)


def _build_runtime_overrides(
    runtime_config: RuntimeConfig,
    idx: int,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "system_prompt": runtime_config.system_prompts[idx],
        "enable_memory": runtime_config.enable_memory[idx],
        "memory_rounds": runtime_config.memory_rounds,
        "memory_format": runtime_config.memory_format,
        "log_session_id": runtime_config.log_session_id,
    }
    if runtime_config.log_root:
        overrides["log_root"] = runtime_config.log_root
    return overrides


def _prepare_client_config(
    runtime_config: RuntimeConfig,
    idx: int,
    model_id: str,
):
    model_profile = load_model(model_id)
    model_config = get_config_for_model(model_profile.model_name)
    if model_profile.config_overrides:
        _apply_model_profile_overrides(model_config, model_profile.config_overrides)

    client_config = model_config.with_overrides(
        **_build_runtime_overrides(runtime_config, idx)
    )
    return model_profile, client_config


def build_agent_clients(
    runtime_config: RuntimeConfig,
    agent_ids: list[str],
) -> list[BaseClient]:
    """Build clients for game agents (supports mixed model ids)."""
    _validate_runtime_fields(runtime_config, agent_ids)

    clients: list[BaseClient] = []

    for idx, model_id in enumerate(runtime_config.model_ids):
        model_profile, client_config = _prepare_client_config(
            runtime_config,
            idx,
            model_id,
        )
        clients.append(
            create_client(
                model_profile.model_name,
                client_config,
                semantic_controls_specs=runtime_config.semantic_controls_specs[idx],
            )
        )

    return clients


_install_custom_logger_methods()
