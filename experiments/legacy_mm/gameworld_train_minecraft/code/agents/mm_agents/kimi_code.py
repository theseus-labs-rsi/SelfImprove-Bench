"""Kimi Code CLI agent implementation."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..harness.function_calling_utils import build_openai_action_tools
from ..harness.memory import memory_entries_to_text
from .base.base_client import BaseClientConfig
from .base.generalist_agent import GeneralistAgent


LOGGER = logging.getLogger(__name__)


@dataclass
class KimiCodeConfig(BaseClientConfig):
    """Configuration for the official Kimi Code CLI."""

    model: str = "kimi-k3"
    model_type: str = "generalist"
    kimi_bin: str = "/usr/local/bin/kimi"
    provider_type: str = "openai"
    endpoint: str | None = "https://free-router.opendatalab.com/v1"
    max_context_size: int = 262144
    capabilities: str = "thinking,tool_use"
    disable_telemetry: bool = True
    request_timeout: float = 180.0
    retry_backoff_sec: float = 2.0
    retry_max_backoff_sec: float = 60.0


class KimiCodeAgent(GeneralistAgent):
    """Run one Kimi Code CLI prompt and parse its final assistant message."""

    def __init__(self, config: BaseClientConfig, **shared_tools):
        super().__init__(config, **shared_tools)
        self._model_name = config.model or "kimi-k3"
        self._kimi_bin = str(getattr(config, "kimi_bin", "/usr/local/bin/kimi"))
        self._provider_type = str(getattr(config, "provider_type", "openai"))
        self._base_url = str(
            getattr(config, "endpoint", None) or "https://free-router.opendatalab.com/v1"
        ).rstrip("/")
        self._max_context_size = int(getattr(config, "max_context_size", 262144))
        self._capabilities = str(getattr(config, "capabilities", "thinking,tool_use"))
        self._disable_telemetry = bool(getattr(config, "disable_telemetry", True))
        self._current_game_state: dict[str, Any] | None = None

        auth_path = Path("/root/.codex/auth.json")
        try:
            auth_payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            auth_payload = {}
        api_key = auth_payload.get("OPENAI_API_KEY")
        self._api_key = self._resolve_api_key(api_key, env_vars=("OPENAI_API_KEY",))

        base_home = Path(config.log_root or ".") / "kimi_code_home"
        self._kimi_code_home = base_home / f"client_{uuid.uuid4().hex}"
        self._kimi_code_home.mkdir(parents=True, exist_ok=True)

    def set_game_state(self, state: dict[str, Any] | None) -> None:
        self._current_game_state = state

    def _build_prompt_text(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[object],
    ) -> str:
        sections: list[str] = []
        if system_prompt and system_prompt.strip():
            sections.append(system_prompt.strip())

        memory_text = memory_entries_to_text(memory_entries) or ""
        if memory_text.strip():
            sections.append("# Recent interaction memory\n" + memory_text.strip())

        if user_prompt and user_prompt.strip():
            sections.append(user_prompt.strip())

        state_json = json.dumps(
            self._current_game_state,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        action_contract = []
        for tool in build_openai_action_tools(self._semantic_controls_specs):
            action_contract.append(
                {
                    "tool_name": tool["name"],
                    "description": tool.get("description", ""),
                    "arguments_schema": tool.get("parameters", {}),
                }
            )

        sections.append("# Current game state\n```json\n" + state_json + "\n```")
        sections.append(
            "# Legal actions and argument formats\n```json\n"
            + json.dumps(action_contract, ensure_ascii=False, separators=(",", ":"))
            + "\n```"
        )
        sections.append(
            "Choose exactly one legal action. Return only one JSON object in this form: "
            '{"tool_name":"<legal action id>","arguments":{"reasoning":"<short rationale>"}}. '
            "Include every additional argument required by that action's arguments_schema. "
            "Do not call shell or file tools and do not include Markdown fences."
        )
        return "\n\n".join(sections)

    def build_request_payload(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[object],
        tools: list[dict[str, object]],
        screenshot_path: Path,
    ) -> dict[str, object]:
        del tools
        return {
            "prompt": self._build_prompt_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                memory_entries=memory_entries,
            ),
            "screenshot_path": str(screenshot_path),
        }

    @staticmethod
    def _last_assistant_content(stream_output: str) -> str:
        last_content = ""
        for raw_line in stream_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("role") != "assistant":
                continue
            content = event.get("content")
            if isinstance(content, str) and content.strip():
                last_content = content.strip()
        return last_content

    @staticmethod
    def _parse_action_payload(text: str) -> dict[str, object] | None:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text or ""):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            tool_name = payload.get("tool_name")
            arguments = payload.get("arguments")
            if isinstance(tool_name, str) and tool_name.strip():
                return {
                    "tool_name": tool_name.strip(),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
        return None

    @staticmethod
    def _is_retryable_failure(message: str) -> bool:
        text = message.lower()
        patterns = (
            "timed out",
            "timeout",
            "status_code=429",
            "status code: 429",
            "http 429",
            "rate limit",
            "too many requests",
            "额度预扣失败",
            "额度不足",
            "insufficient quota",
            "insufficient balance",
            "overloaded",
            "service unavailable",
        )
        return any(pattern in text for pattern in patterns) or bool(
            re.search(r"(?:status(?:_code)?[=: ]+|http\s+|\b)(?:500|502|503|504)\b", text)
        )

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "KIMI_MODEL_NAME": self._model_name,
                "KIMI_MODEL_API_KEY": self._api_key,
                "KIMI_MODEL_PROVIDER_TYPE": self._provider_type,
                "KIMI_MODEL_BASE_URL": self._base_url,
                "KIMI_MODEL_MAX_CONTEXT_SIZE": str(self._max_context_size),
                "KIMI_MODEL_CAPABILITIES": self._capabilities,
                "KIMI_DISABLE_TELEMETRY": "1" if self._disable_telemetry else "0",
                "KIMI_CODE_HOME": str(self._kimi_code_home),
            }
        )
        return env

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]")

    def send_request(self, request_payload: dict[str, object]) -> object:
        prompt = str(request_payload.get("prompt") or "")
        cmd = [self._kimi_bin, "--output-format", "stream-json", "-p", prompt]
        backoff = max(0.1, float(getattr(self.config, "retry_backoff_sec", 2.0)))
        max_backoff = max(backoff, float(getattr(self.config, "retry_max_backoff_sec", 60.0)))
        attempt = 0

        while True:
            attempt += 1
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(Path.cwd()),
                    env=self._build_env(),
                    capture_output=True,
                    text=True,
                    timeout=self.config.request_timeout,
                    check=False,
                )
                stdout = self._redact(proc.stdout)
                stderr = self._redact(proc.stderr)
                combined = f"{stdout}\n{stderr}"
                if proc.returncode == 0:
                    return {
                        "returncode": proc.returncode,
                        "stdout": stdout,
                        "stderr": stderr,
                        "assistant_content": self._last_assistant_content(stdout),
                        "attempts": attempt,
                    }
                if not self._is_retryable_failure(combined):
                    return {
                        "returncode": proc.returncode,
                        "stdout": stdout,
                        "stderr": stderr,
                        "assistant_content": self._last_assistant_content(stdout),
                        "attempts": attempt,
                    }
                LOGGER.warning(
                    "Kimi Code transient failure on attempt %d; retrying in %.1fs: %s",
                    attempt,
                    backoff,
                    stderr.strip() or "no stderr",
                )
            except subprocess.TimeoutExpired:
                LOGGER.warning(
                    "Kimi Code timed out after %.1fs on attempt %d; retrying in %.1fs",
                    self.config.request_timeout,
                    attempt,
                    backoff,
                )
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2.0)

    def extract_tool_call(self, response: object) -> dict[str, object] | None:
        if not isinstance(response, dict):
            return None
        return self._parse_action_payload(str(response.get("assistant_content") or ""))

    def extract_reasoning(self, response: object) -> str | None:
        action = self.extract_tool_call(response)
        if not action:
            return None
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            return None
        reasoning = arguments.get("reasoning")
        return reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None

    def extract_error(self, response: object) -> str | None:
        if not isinstance(response, dict):
            return "Invalid Kimi Code response"
        returncode = response.get("returncode")
        if returncode not in (0, None):
            stderr = str(response.get("stderr") or "").strip()
            return f"Kimi Code failed with returncode={returncode}: {stderr or 'no stderr'}"
        if not self.extract_tool_call(response):
            return "No valid action in final Kimi Code assistant content"
        return None


__all__ = ["KimiCodeAgent", "KimiCodeConfig"]
