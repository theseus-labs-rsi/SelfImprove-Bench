"""Thin compatibility API; provider calls live in RSI/rsi_methods/Baseline."""

from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RSI.rsi_methods.Baseline.client import (  # noqa: E402
    DSH_INSTALL_HELP, LLMClientError, PROVIDER_CHOICES, RSI_PROVIDERS,
    call_claude, call_codex, call_deepseek_harness as _call_deepseek_harness, call_openai,
    extract_json_object, load_rsi_integration, request_json,
)

DEFAULT_PROMPT_FILE = Path(__file__).parent / "prompts" / "farm_planning_prompt.md"

def _dsh_projection_directory(environment: Mapping[str, str] | None = None) -> Path | None:
    value = (environment or os.environ).get("DSH_PROJECTION_DIR")
    return Path(value) if value else None

def call_deepseek_harness(prompt: str, **kwargs: Any) -> str:
    command = kwargs.get("command") or shutil.which("dsh") or shutil.which("dsh.cmd")
    if not command:
        raise LLMClientError(DSH_INSTALL_HELP)
    kwargs["command"] = command
    kwargs.setdefault("task_label", "farm-planning-task")
    kwargs.setdefault("projection_directory", _dsh_projection_directory())
    return _call_deepseek_harness(prompt, **kwargs)

def load_prompt(prompt_file: str | Path = DEFAULT_PROMPT_FILE) -> str:
    return Path(prompt_file).read_text(encoding="utf-8")

def build_prompt(task: Mapping[str, Any], prompt_file: str | Path = DEFAULT_PROMPT_FILE) -> str:
    visible = {k: v for k, v in task.items() if k not in {"reference_daily_profits", "reference_profit", "reference_method"}}
    return f"{load_prompt(prompt_file).rstrip()}\n\n## Current task\n```json\n{json.dumps(visible, ensure_ascii=False, indent=2)}\n```\n\nThis is a non-interactive API call. Return only the plan JSON object.\n"

def _validate_plan(plan: dict[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("task_id") not in (None, task.get("task_id")):
        raise LLMClientError("LLM plan task_id does not match the task")
    if not isinstance(plan.get("days"), list):
        raise LLMClientError("LLM plan must contain a days array")
    return plan

def request_plan(task: Mapping[str, Any], *, provider: str = "codex", prompt_file: str | Path = DEFAULT_PROMPT_FILE, model: str | None = None, timeout: float = 300, max_attempts: int = 3, show_agent_output: bool = False, metrics: dict[str, Any] | None = None, rsi_session: Any | None = None, **provider_options: Any) -> dict[str, Any]:
    return request_json(task, prompt=build_prompt(task, prompt_file), validator=_validate_plan, provider=provider, model=model, timeout=timeout, max_attempts=max_attempts, show_agent_output=show_agent_output, metrics=metrics, rsi_session=rsi_session, callers={"codex": call_codex, "claude": call_claude, "deepseek": call_deepseek_harness, "openai": call_openai}, **provider_options)
