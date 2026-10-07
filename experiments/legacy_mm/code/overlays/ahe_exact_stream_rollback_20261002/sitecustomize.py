"""Task-scoped AHE retry support with lossless conversation rollback.

This overlay is active only for the clean ``mc_ahe/task_05`` supplement
episode.  A provider ``response.failed`` can append the current user turn to
NexAU history before raising.  Before exposing that failure to the existing
bounded action retry, restore the exact pre-call history so the retry does not
duplicate a game observation.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback


episode = Path(os.environ.get("SUPPLEMENT_EPISODE", ""))
if tuple(episode.parts[-2:]) == ("mc_ahe", "task_05"):
    project = Path(__file__).resolve().parents[2]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))

    from benches.gameworld import network_retry
    from rsi_methods.AHE import agent as agent_module

    _original_classifier = network_retry.is_transient_connection_error
    _pairs = (
        ('"message": "Unable to forward upstream stream"', '"type": "upstream_error"'),
        ('"message": "stream_read_error"', '"code": "stream_read_error"'),
        ('"message": "An error occurred while processing the request."', '"type": "server_error"'),
    )

    def _exact_provider_transient(error: BaseException | str) -> bool:
        text = str(error)
        return '"event": "response.failed"' in text and any(
            message in text and kind in text for message, kind in _pairs
        )

    def _ahe_transient_connection_error(error: Exception) -> bool:
        return _original_classifier(error) or _exact_provider_transient(error)

    network_retry.is_transient_connection_error = _ahe_transient_connection_error

    def _serve_actions_with_rollback(connection, agent, sandbox_dir, log) -> None:
        while True:
            request = connection.recv()
            if request.get("op") == "close":
                return
            before_history = None
            before_messages = None
            try:
                if request.get("op") == "snapshot_session":
                    connection.send(
                        {"ok": True, "history": agent_module.dump_nexau_history(agent)}
                    )
                    continue
                if request.get("op") == "restore_session":
                    history = request.get("history")
                    if not isinstance(history, list) or agent.history:
                        raise RuntimeError("NexAU worker is not a fresh empty session")
                    from nexau.core.messages import Message

                    validated = [Message.model_validate(item) for item in history]
                    agent.history.replace_all(validated)
                    agent.history.flush()
                    restored = agent_module.dump_nexau_history(agent)
                    if restored != history:
                        raise RuntimeError("NexAU session history changed during restore")
                    connection.send({"ok": True, "history": restored})
                    continue
                if request.get("op") != "act":
                    raise RuntimeError("Unknown NexAU worker operation")

                if request.get("image_paths"):
                    removed = agent_module.compact_prior_game_turns(agent)
                    if removed:
                        (Path(request["call_dir"]) / "history_compaction.json").write_text(
                            json.dumps(
                                {"removed_messages": removed, "keep_recent_turns": 8}
                            ),
                                encoding="utf-8",
                        )
                agent_module.compact_prior_screenshots(agent)
                # Keep deep copies of the exact Message objects after
                # deterministic compaction. ``agent.run`` replaces the system
                # message and its error path appends/flushed user+error
                # messages, so shallow references are not a stable rollback
                # snapshot. JSON validation is also unsuitable because it can
                # regenerate ``created_at`` at microsecond precision.
                before_messages = [
                    message.model_copy(deep=True) for message in agent.history
                ]
                before_history = agent_module.dump_nexau_history(agent)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                raw = agent.run(
                    message=agent_module.make_nexau_message(
                        request["prompt"], request.get("image_paths") or []
                    ),
                    context={
                        "date": now,
                        "username": os.environ.get("USER") or os.environ.get("USERNAME"),
                        "working_directory": str(sandbox_dir),
                        "env_content": {
                            "date": now,
                            "working_directory": str(sandbox_dir),
                        },
                    },
                )
                answer = raw[0] if isinstance(raw, tuple) else raw
                trace_path = agent_module.AHEAgent._save_trace(
                    agent, Path(request["call_dir"])
                )
                connection.send(
                    {"ok": True, "text": str(answer or ""), "trace_path": str(trace_path)}
                )
            except BaseException as exc:
                traceback.print_exc(file=log)
                try:
                    agent_module.AHEAgent._save_trace(
                        agent, Path(request.get("call_dir", sandbox_dir))
                    )
                except BaseException:
                    traceback.print_exc(file=log)
                rolled_back = False
                rollback_error = None
                if before_history is not None and _exact_provider_transient(exc):
                    try:
                        agent.history.replace_all(before_messages)
                        agent.history.flush()
                        # NexAU serializes a missing ``created_at`` as
                        # ``datetime.now()`` without assigning it back to the
                        # Message, so two JSON dumps of the same exact object
                        # can differ by microseconds. Compare the actual
                        # in-memory Pydantic values instead.
                        if list(agent.history) != before_messages:
                            raise RuntimeError("history differs after rollback")
                        rolled_back = True
                    except BaseException as restore_exc:
                        rollback_error = f"{type(restore_exc).__name__}: {restore_exc}"
                        traceback.print_exc(file=log)
                error_text = str(exc)
                if rollback_error:
                    error_text += f"; AHE history rollback failed: {rollback_error}"
                connection.send(
                    {
                        "ok": False,
                        "error": error_text,
                        "error_type": (
                            "ConnectionError"
                            if rolled_back
                            else type(exc).__name__
                        ),
                        "status_code": getattr(exc, "status_code", None),
                        "history_rolled_back": rolled_back,
                    }
                )

    agent_module.AHEAgent._serve_actions = staticmethod(_serve_actions_with_rollback)
