"""Bounded retries for connection failures while requesting one game action.

Only retry transport and transient upstream errors. Invalid requests (notably
AHE's image-limit 400), authentication errors, and agent coding errors are not
retried. A failed action has not yet been delivered to GameWorld.
"""

from __future__ import annotations

import time
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")
# 520-524: Cloudflare in front of the model gateway (e.g. 524 origin timeout).
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def is_transient_connection_error(error: Exception) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status in _RETRYABLE_STATUS
        name = type(current).__name__
        if name in {"APIConnectionError", "APITimeoutError", "ConnectionError", "ProxyError", "ConnectError", "ConnectTimeout"}:
            return True
        current = current.__cause__ or current.__context__
    # HyperAgents runs its TaskAgent in Docker and wraps the original proxy
    # traceback in a plain RuntimeError; match the explicit exception marker,
    # never a bare "502" which could be an unrelated agent message.
    text = str(error)
    wrapped_markers = (
        "ProxyError: 502 Bad Gateway",
        "APIConnectionError: Connection error.",
        "openai.APITimeoutError: Request timed out.",
        "httpx.ConnectTimeout:",
        "httpx2.ConnectTimeout:",
        "httpcore.ConnectTimeout:",
        "httpcore2.ConnectTimeout:",
        # The SDK raises InternalServerError only for 5xx responses.
        "openai.InternalServerError: Error code: 5",
    )
    return any(marker in text for marker in wrapped_markers)


def call_with_connection_retry(
    call: Callable[[], T], *, max_attempts: int = 4, initial_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    audit_path: Path | None = None, invalidate_usage_on_retry: bool = False,
) -> T:
    if max_attempts < 1 or initial_delay < 0:
        raise ValueError("Invalid connection retry settings")
    def record(attempt: int, status: str, usage=None, error: Exception | None = None) -> None:
        if audit_path is None:
            return
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"attempt": attempt, "status": status,
               "error_type": type(error).__name__ if error else None,
               "status_code": getattr(error, "status_code", None) if error else None,
               "usage": asdict(usage) if is_dataclass(usage) else usage}
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    for attempt in range(1, max_attempts + 1):
        try:
            response = call()
            record(attempt, "completed", getattr(response, "usage", None))
            if attempt > 1 and invalidate_usage_on_retry and is_dataclass(response):
                # No provider usage is available for the earlier failed requests.
                response = replace(response, usage=None, metadata={**(response.metadata or {}),
                    "token_coverage": "unknown_prior_failed_attempts"})
            return response
        except Exception as exc:
            record(attempt, "failed_usage_unknown", error=exc)
            if attempt == max_attempts or not is_transient_connection_error(exc):
                raise
            delay = min(initial_delay * 2 ** (attempt - 1), 15.0)
            print(f"[gameworld] transient model connection failure; retrying action "
                  f"{attempt + 1}/{max_attempts} after {delay:.0f}s ({type(exc).__name__})", flush=True)
            sleep(delay)
    raise AssertionError("unreachable")
