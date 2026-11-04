"""Provider-neutral baseline calls and JSON retry handling.

Game packages own prompts and output validation.  This module owns all external
Codex, Claude Code and DeepSeek Harness invocation details, so adding a game no
longer duplicates process management or telemetry code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import rsi_session
from .methods import RSI_METHOD_IDENTITIES


BASELINE_PROVIDERS = ("codex", "claude", "deepseek", "openai")
PROVIDER_CHOICES = BASELINE_PROVIDERS + RSI_METHOD_IDENTITIES
DSH_INSTALL_HELP = (
    "DeepSeek Harness (dsh) is not installed. Install it with:\n"
    "  npm.cmd install --global @deepseek-ai/dsh\n"
    "Then set DEEPSEEK_API_KEY and run dsh --version."
)

# The static baseline (dsh) must be write-free and tool-free: pin the sandbox to read-only
# and disable every tool-* plugin
# (values and plugin ids come from the real output of `dsh --profile headless --dump-config`)
DSH_READ_ONLY_TOOL_IDS = (
    "tool-bash",
    "tool-pwsh",
    "tool-jobs",
    "tool-fs",
    "tool-fs-search",
    "tool-skill",
    "tool-subagent-control",
    "tool-subagent-list-agents",
    "tool-subagent",
    "tool-subagent-fork",
    "tool-workflow",
    "tool-result-pruner",
    "tool-todo",
    "tool-goal",
    "tool-ralph",
    "tool-str-replace-editor",
    "tool-web",
)

_NESTED_CODEX_ENV_VARS = (
    "CODEX_CI",
    "CODEX_PERMISSION_PROFILE",
    "CODEX_SANDBOX_NETWORK_DISABLED",
    "CODEX_SESSION_ID",
    "CODEX_THREAD_ID",
)
_NESTED_CLAUDE_ENV_VARS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")


class LLMClientError(RuntimeError):
    """An agent call failed or returned an invalid JSON action."""


def _unknown_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": None,
        "reasoning_tokens": None,
        "answer_tokens": None,
        "total_tokens": None,
        "method": "unavailable",
        "includes_all_attempts": True,
    }


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(value: Any, prompt: str = "", answer: str = "") -> dict[str, Any]:
    def field(name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    input_tokens = _integer(field("input_tokens"))
    if input_tokens is None:
        input_tokens = _integer(field("prompt_tokens"))
    output_tokens = _integer(field("output_tokens"))
    if output_tokens is None:
        output_tokens = _integer(field("completion_tokens"))
    details = field("output_tokens_details") or field("completion_tokens_details")
    reasoning = None
    if details is not None:
        reasoning = _integer(details.get("reasoning_tokens") if isinstance(details, Mapping) else getattr(details, "reasoning_tokens", None))
    if input_tokens is None or output_tokens is None:
        return _unknown_usage()
    reasoning = reasoning or 0
    total = _integer(field("total_tokens"))
    if total is None:
        total = input_tokens + output_tokens
    return {
        "prompt_tokens": input_tokens,
        "reasoning_tokens": reasoning,
        "answer_tokens": max(0, output_tokens - reasoning),
        "total_tokens": total,
        "method": "provider_reported",
        "includes_all_attempts": True,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise LLMClientError("agent response does not contain a JSON object")


def _timeout_tail(exc: subprocess.TimeoutExpired, limit: int = 500) -> str:
    """Truncated tail of the subprocess stdout/stderr in a timeout exception, for error messages."""
    parts: list[str] = []
    for name in ("stdout", "stderr"):
        value = getattr(exc, name, None)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value:
            parts.append(f"{name} tail: {str(value).strip()[-limit:]!r}")
    return " | ".join(parts) if parts else "child produced no output before the timeout"


def _run(
    args: list[str],
    *,
    prompt: str | None,
    timeout: float,
    environment: Mapping[str, str],
    show_agent_output: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "text": True,
        "encoding": "utf-8",
        "timeout": timeout,
        "check": False,
        "env": environment,
    }
    if show_agent_output:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": None})
    else:
        kwargs["capture_output"] = True
    if prompt is not None:
        kwargs["input"] = prompt
    return runner(args, **kwargs)


def call_codex(
    prompt: str,
    *,
    model: str | None = None,
    command: str | None = None,
    timeout: float = 300,
    clean_environment: bool = True,
    show_agent_output: bool = False,
    telemetry: dict[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    executable = command or shutil.which("codex") or shutil.which("codex.cmd")
    if not executable:
        raise LLMClientError("codex executable was not found on PATH")
    args = [executable, "exec", "--ephemeral", "--skip-git-repo-check", "--color", "never"]
    if model:
        args.extend(["--model", model])
    args.append("-")
    environment = os.environ.copy()
    if clean_environment:
        for name in _NESTED_CODEX_ENV_VARS:
            environment.pop(name, None)
    try:
        completed = _run(
            args,
            prompt=prompt,
            timeout=timeout,
            environment=environment,
            show_agent_output=show_agent_output,
            runner=runner,
        )
    except subprocess.TimeoutExpired as exc:
        partial = getattr(exc, "stdout", None) or getattr(exc, "output", None) or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        if partial:
            try:
                extract_json_object(str(partial))
            except LLMClientError:
                pass
            else:
                if telemetry is not None:
                    telemetry["tokens"] = _unknown_usage()
                return str(partial)
        raise LLMClientError(
            f"Codex invocation timed out after {timeout:g}s; check that Codex is logged in."
        ) from exc
    except OSError as exc:
        raise LLMClientError(f"Codex invocation failed: {exc}") from exc
    answer = completed.stdout or ""
    if telemetry is not None:
        telemetry["tokens"] = _unknown_usage()
    if completed.returncode != 0:
        raise LLMClientError(
            f"Codex exited with code {completed.returncode}: "
            f"{(completed.stderr or answer).strip()}"
        )
    return answer


def call_claude(
    prompt: str,
    *,
    model: str | None = None,
    command: str | None = None,
    timeout: float = 300,
    clean_environment: bool = True,
    show_agent_output: bool = False,
    telemetry: dict[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    executable = command or shutil.which("claude") or shutil.which("claude.cmd")
    if not executable:
        raise LLMClientError("claude executable was not found on PATH")
    output_format = "stream-json" if show_agent_output else "json"
    args = [executable, "-p", "--no-session-persistence", "--output-format", output_format, "--tools="]
    if show_agent_output:
        args.extend(["--include-partial-messages", "--verbose"])
    if model:
        args.extend(["--model", model])
    environment = os.environ.copy()
    if clean_environment:
        for name in _NESTED_CLAUDE_ENV_VARS:
            environment.pop(name, None)
    try:
        completed = _run(
            args,
            prompt=prompt,
            timeout=timeout,
            environment=environment,
            show_agent_output=show_agent_output,
            runner=runner,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMClientError(
            f"Claude invocation timed out after {timeout:g}s; check that Claude Code is authenticated."
            f" | {_timeout_tail(exc)}"
        ) from exc
    except OSError as exc:
        raise LLMClientError(f"Claude invocation failed: {exc}") from exc
    if completed.returncode != 0:
        raise LLMClientError(
            f"Claude exited with code {completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '').strip()}"
        )
    try:
        envelope = json.loads(completed.stdout or "")
    except json.JSONDecodeError as exc:
        envelope = None
        for line in (completed.stdout or "").splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping) and isinstance(candidate.get("result"), str):
                envelope = candidate
        if envelope is None:
            raise LLMClientError("Claude returned an invalid JSON result envelope") from exc
    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("result"), str):
        raise LLMClientError("Claude JSON result did not contain a text result")
    answer = str(envelope["result"])
    if telemetry is not None:
        raw = envelope.get("usage")
        if isinstance(raw, Mapping):
            prompt_count = sum(
                _integer(raw.get(name)) or 0
                for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
            output_count = _integer(raw.get("output_tokens"))
            if output_count is not None:
                telemetry["tokens"] = {
                    "prompt_tokens": prompt_count,
                    "reasoning_tokens": None,
                    "answer_tokens": output_count,
                    "total_tokens": prompt_count + output_count,
                    "method": "claude_cli_reported",
                    "includes_all_attempts": True,
                    "input_token_details": {
                        "uncached_input_tokens": _integer(raw.get("input_tokens")) or 0,
                        "cache_creation_input_tokens": _integer(raw.get("cache_creation_input_tokens")) or 0,
                        "cache_read_input_tokens": _integer(raw.get("cache_read_input_tokens")) or 0,
                    },
                }
        model_usage = envelope.get("modelUsage")
        if isinstance(model_usage, Mapping) and (
            not isinstance(raw, Mapping)
            or sum(_integer(raw.get(name)) or 0 for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")) == 0
        ):
            prompt_count = 0
            output_count = 0
            for usage_record in model_usage.values():
                if not isinstance(usage_record, Mapping):
                    continue
                prompt_count += sum(
                    _integer(usage_record.get(name)) or 0
                    for name in ("inputTokens", "cacheCreationInputTokens", "cacheReadInputTokens")
                )
                output_count += _integer(usage_record.get("outputTokens")) or 0
            telemetry["tokens"] = {
                "prompt_tokens": prompt_count,
                "reasoning_tokens": None,
                "answer_tokens": output_count,
                "total_tokens": prompt_count + output_count,
                "method": "claude_cli_model_usage_all_models",
                "includes_all_attempts": True,
            }
        telemetry.setdefault("tokens", _unknown_usage())
    return answer


def call_deepseek_harness(
    prompt: str,
    *,
    model: str | None = None,
    command: str | None = None,
    timeout: float = 300,
    show_agent_output: bool = False,
    telemetry: dict[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    task_label: str = "game-task",
    projection_directory: Path | None = None,
    reasoning_effort: str | None = None,
    read_only: bool = True,
) -> str:
    executable = command or shutil.which("dsh") or shutil.which("dsh.cmd")
    if not executable:
        raise LLMClientError(DSH_INSTALL_HELP)
    environment = os.environ.copy()
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="game-benchmark-dsh-", encoding="utf-8", delete=False
    )
    patch_path = Path(handle.name)
    with handle:
        if read_only:
            # The static baseline must not modify the environment: read-only sandbox
            # plus all tools disabled (matches codex read_only/deny_all and claude --tools=).
            handle.write("- id: sandbox-policy\n  config:\n    mode: read-only\n")
            for tool_id in DSH_READ_ONLY_TOOL_IDS:
                handle.write(f"- id: {tool_id}\n  disabled: true\n")
        defaults: list[str] = []
        if model:
            defaults.append(f"    model: {json.dumps(model)}")
        if reasoning_effort:
            # The dsh adapter only accepts off|low|high|max; an invalid value raises
            # UNSUPPORTED_REASONING_EFFORT at call time.
            defaults.append(f"    reasoningEffort: {json.dumps(reasoning_effort)}")
        if defaults:
            handle.write(
                "- id: agent-default-model\n  config:\n"
                "    provider: deepseek-official\n" + "\n".join(defaults) + "\n"
            )
        handle.write(
            "- id: headless-runner\n  config:\n"
            f"    task: {json.dumps(prompt)}\n"
        )
    projection_before = set(projection_directory.glob("*.json")) if projection_directory else set()
    args = [executable, "--profile", "headless", "--patch", str(patch_path), task_label]
    try:
        completed = _run(
            args,
            prompt=None,
            timeout=timeout,
            environment=environment,
            show_agent_output=show_agent_output,
            runner=runner,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMClientError(
            f"DeepSeek Harness timed out after {timeout:g}s; check DEEPSEEK_API_KEY and connectivity."
            f" | {_timeout_tail(exc)}"
        ) from exc
    except OSError as exc:
        raise LLMClientError(f"DeepSeek Harness invocation failed: {exc}") from exc
    finally:
        patch_path.unlink(missing_ok=True)
    answer = completed.stdout or ""
    if telemetry is not None:
        telemetry["tokens"] = _unknown_usage()
        if projection_directory is not None:
            for projection in sorted(set(projection_directory.glob("*.json")) - projection_before):
                try:
                    payload = json.loads(projection.read_text(encoding="utf-8"))
                    totals = payload["record"]["rows"]["tokenUsage"]["val"]["totals"]
                    prompt_tokens = sum(
                        _integer(totals.get(name)) or 0
                        for name in ("uncachedInputTokens", "cacheReadTokens", "cacheWriteTokens")
                    )
                    output_tokens = _integer(totals.get("outputTokens")) or 0
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                telemetry["tokens"] = {
                    "prompt_tokens": prompt_tokens,
                    "reasoning_tokens": None,
                    "answer_tokens": output_tokens,
                    "total_tokens": prompt_tokens + output_tokens,
                    "method": "dsh_session_projection_provider_reported",
                    "includes_all_attempts": True,
                }
                break
    if completed.returncode != 0:
        raise LLMClientError(
            f"DeepSeek Harness exited with code {completed.returncode}: "
            f"{(completed.stderr or answer).strip()}"
        )
    return answer


def call_openai(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    client: Any = None,
    telemetry: dict[str, Any] | None = None,
    show_agent_output: bool = False,
) -> str:
    del show_agent_output
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError("OpenAI SDK is not installed; run: pip install openai") from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMClientError("OPENAI_API_KEY is not set")
        options: dict[str, Any] = {"api_key": key}
        if base_url:
            options["base_url"] = base_url
        if timeout is not None:
            options["timeout"] = timeout
        client = OpenAI(**options)
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        if hasattr(client, "responses"):
            response = client.responses.create(model=selected_model, input=prompt)
            answer = getattr(response, "output_text", None)
            if answer:
                if telemetry is not None:
                    telemetry["tokens"] = _usage(getattr(response, "usage", None), prompt, answer)
                return answer
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content or ""
        if telemetry is not None:
            telemetry["tokens"] = _usage(getattr(response, "usage", None), prompt, answer)
        return answer
    except Exception as exc:
        raise LLMClientError(f"OpenAI request failed: {exc}") from exc


def load_rsi_integration() -> Any:
    return rsi_session


def _call_rsi(
    prompt: str,
    *,
    method: str,
    rsi_agent: Any,
    telemetry: dict[str, Any],
) -> str:
    if rsi_agent is None or not callable(getattr(rsi_agent, "act", None)):
        raise LLMClientError(f"Provider {method} requires an active RSI agent")
    bridge = load_rsi_integration()
    response = rsi_agent.act(prompt)
    telemetry["tokens"] = bridge.rsi_token_metrics(response)
    telemetry["rsi_agent_call"] = bridge.backend_response_record(method, prompt, response)
    return bridge.extract_rsi_answer(method, response)


def _aggregate_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("prompt_tokens", "reasoning_tokens", "answer_tokens", "total_tokens")
    result: dict[str, Any] = {}
    for field in fields:
        values = [attempt.get("tokens", {}).get(field) for attempt in attempts]
        result[field] = (
            sum(values)
            if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            else None
        )
    methods = {attempt.get("tokens", {}).get("method", "unavailable") for attempt in attempts}
    result["method"] = methods.pop() if len(methods) == 1 else "attempt_sum"
    result["includes_all_attempts"] = True
    return result


def request_json(
    task: Mapping[str, Any],
    *,
    prompt: str,
    validator: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
    provider: str = "codex",
    model: str | None = None,
    timeout: float = 300,
    max_attempts: int = 3,
    show_agent_output: bool = False,
    metrics: dict[str, Any] | None = None,
    rsi_agent: Any | None = None,
    callers: Mapping[str, Callable[..., str]] | None = None,
    **provider_options: Any,
) -> dict[str, Any]:
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= 3:
        raise LLMClientError("max_attempts must be an integer from 1 to 3")
    normalized = provider.strip().lower()
    if normalized not in PROVIDER_CHOICES:
        raise LLMClientError(f"Unsupported provider: {provider}")
    available = dict(callers or {})
    available.setdefault("codex", call_codex)
    available.setdefault("claude", call_claude)
    available.setdefault("deepseek", call_deepseek_harness)
    available.setdefault("openai", call_openai)
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    current_prompt = prompt
    if metrics is not None:
        metrics.clear()
        metrics.update({"provider": normalized, "attempts": attempts})
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()
        telemetry: dict[str, Any] = {}
        if show_agent_output:
            print(
                f"[agent] provider={normalized} attempt={attempt}/{max_attempts} starting",
                file=sys.stderr,
                flush=True,
            )
        try:
            if normalized in RSI_METHOD_IDENTITIES:
                answer = _call_rsi(
                    current_prompt,
                    method=normalized,
                    rsi_agent=rsi_agent,
                    telemetry=telemetry,
                )
            else:
                answer = available[normalized](
                    current_prompt,
                    model=model,
                    timeout=timeout,
                    show_agent_output=show_agent_output,
                    telemetry=telemetry,
                    **provider_options,
                )
        except Exception as exc:
            record = {
                "attempt": attempt,
                "status": "provider_error",
                "duration_seconds": round(time.perf_counter() - attempt_started, 6),
                "tokens": telemetry.get("tokens", _unknown_usage()),
                "error": str(exc),
            }
            if "rsi_agent_call" in telemetry:
                record["rsi_agent_call"] = telemetry["rsi_agent_call"]
            attempts.append(record)
            if metrics is not None:
                metrics.update(
                    duration_seconds=round(time.perf_counter() - started, 6),
                    attempt_count=len(attempts),
                    tokens=_aggregate_attempts(attempts),
                )
            raise
        try:
            value = validator(extract_json_object(answer), task)
        except (LLMClientError, ValueError) as exc:
            record = {
                "attempt": attempt,
                "status": "validation_error",
                "duration_seconds": round(time.perf_counter() - attempt_started, 6),
                "tokens": telemetry.get("tokens", _unknown_usage()),
                "error": str(exc),
            }
            if "rsi_agent_call" in telemetry:
                record["rsi_agent_call"] = telemetry["rsi_agent_call"]
            attempts.append(record)
            from .methods import runtime_policy
            non_retryable = runtime_policy(normalized).is_fatal_error(answer)
            if attempt == max_attempts or non_retryable:
                if metrics is not None:
                    metrics.update(
                        duration_seconds=round(time.perf_counter() - started, 6),
                        attempt_count=len(attempts),
                        tokens=_aggregate_attempts(attempts),
                    )
                preview = answer.strip()
                if len(preview) > 8000:
                    preview = f"<truncated; showing last 8000 characters>\n{preview[-8000:]}"
                raise LLMClientError(
                    f"Invalid action after {attempt} attempts: {exc}\nLast response:\n{preview or '<empty response>'}"
                ) from exc
            current_prompt = (
                f"{prompt}\n\nYour previous response failed validation ({exc}). "
                "Return only one corrected JSON object."
            )
            continue
        record = {
            "attempt": attempt,
            "status": "ok",
            "duration_seconds": round(time.perf_counter() - attempt_started, 6),
            "tokens": telemetry.get("tokens", _unknown_usage()),
        }
        if "rsi_agent_call" in telemetry:
            record["rsi_agent_call"] = telemetry["rsi_agent_call"]
        attempts.append(record)
        if metrics is not None:
            metrics.update(
                duration_seconds=round(time.perf_counter() - started, 6),
                attempt_count=len(attempts),
                tokens=_aggregate_attempts(attempts),
            )
        if show_agent_output:
            print(
                f"[agent] provider={normalized} attempt={attempt}/{max_attempts} completed",
                file=sys.stderr,
                flush=True,
            )
        return value
    raise AssertionError("unreachable")
