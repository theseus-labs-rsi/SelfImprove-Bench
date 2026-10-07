"""Replay utilities for building HTML and video artifacts from agent logs."""

from __future__ import annotations

import atexit
import signal
import threading
from pathlib import Path


def trigger_auto_replays(run_dir: str | Path, reason: str | None = None) -> None:
    from .html import trigger_html_replayer
    from .video import trigger_video_replayer

    trigger_html_replayer(run_dir, reason=reason)
    trigger_video_replayer(run_dir, reason=reason)


def _handle_shutdown_signal(signum, _frame) -> None:
    raise KeyboardInterrupt(f"signal-{signum}")


def register_auto_replay_hooks(run_dir: str | Path) -> None:
    resolved_run_dir = Path(run_dir)
    atexit.register(lambda: trigger_auto_replays(resolved_run_dir, "exit"))
    # Supplemental evaluations may run several GameWorld tasks concurrently in
    # worker threads. Python only permits signal handlers to be installed by
    # the main thread; atexit replay generation remains safe for worker runs.
    if threading.current_thread() is not threading.main_thread():
        return
    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGQUIT", None)):
        if sig is None:
            continue
        signal.signal(sig, _handle_shutdown_signal)


__all__ = [
    "register_auto_replay_hooks",
    "trigger_auto_replays",
]
