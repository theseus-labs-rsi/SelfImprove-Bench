"""Durable session envelopes; method state belongs to the adapter hooks."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from .checkpoint_io import sync_directory
from .methods import BUILTIN_ADAPTERS


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


def _adapter_spec(session: object) -> str:
    return getattr(session.agent_config, "adapter", None) or BUILTIN_ADAPTERS[session.method]


def export(session: object) -> dict:
    payload = session.adaptor.export_state()
    if not isinstance(payload, Mapping):
        raise TypeError("RSI export_state() must return a JSON object")
    return {
        "method": session.method, "batch_id": session.batch_id,
        "selection_stage": session._selection_stage,
        "adapter": _adapter_spec(session),
        "adapter_state": json.loads(json.dumps(dict(payload), allow_nan=False)),
    }


def restore(session: object, state: dict) -> None:
    if state.get("method") != session.method or state.get("batch_id") != session.batch_id:
        raise RuntimeError("Resume RSI session identity mismatch")
    if "adapter_state" in state:
        if state.get("adapter") != _adapter_spec(session):
            raise RuntimeError("Resume RSI adapter identity mismatch")
        payload = state["adapter_state"]
    elif session.method in BUILTIN_ADAPTERS:
        # Pre-interface checkpoints kept bundled method fields in the envelope.
        payload = state
    else:
        raise RuntimeError("Missing RSI adapter checkpoint state")
    session.adaptor.restore_state(payload)
    session._selection_stage = int(state["selection_stage"])
