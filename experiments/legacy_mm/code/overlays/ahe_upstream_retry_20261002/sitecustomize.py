"""Narrow AHE-only classifier amendment for an exact provider stream failure."""
from __future__ import annotations

import os
from pathlib import Path
import sys


episode = os.environ.get("SUPPLEMENT_EPISODE", "")
parts = set(Path(episode).parts)
if {"mc_ahe", "astray_ahe"} & parts:
    project = Path(__file__).resolve().parents[2]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from benches.gameworld import network_retry

    _original = network_retry.is_transient_connection_error
    _exact = '"message": "Unable to forward upstream stream"'

    def _ahe_transient_connection_error(error: Exception) -> bool:
        return _original(error) or _exact in str(error)

    network_retry.is_transient_connection_error = _ahe_transient_connection_error
