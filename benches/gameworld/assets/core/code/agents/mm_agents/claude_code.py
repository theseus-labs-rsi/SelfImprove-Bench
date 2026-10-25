"""Claude Code CLI agent implementation."""

from __future__ import annotations

import base64
import fcntl
import json
import logging
import mimetypes
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..harness.function_calling_utils import build_openai_action_tools
from ..harness.memory import memory_entries_to_text
from .base.base_client import BaseClientConfig
from .base.generalist_agent import GeneralistAgent


LOGGER = logging.getLogger(__name__)


@dataclass
class ClaudeCodeConfig(BaseClientConfig):
    """Configuration for the Claude Code CLI backend."""

    model: str = "claude-opus-5-5"
    model_type: str = "generalist"
    api_key: str | None = None
    claude_bin: str = "claude"
    endpoint: str | None = None
    request_timeout: float = 180.0
    retry_interval_sec: float = 30.0
    max_attempts: int = 120
    lock_path: str = "/tmp/claude-opus55.lock"


class ClaudeCodeAgent(GeneralistAgent):
    """Run one stateless Claude Code prompt for each semantic game action."""

    def __init__(self, config: BaseClientConfig, **shared_tools):
        super().__init__(config, **shared_tools)
        self._model_name = config.model or "claude-opus-5-5"
        self._claude_bin = str(getattr(config, "claude_bin", "claude"))
        self._base_url = str(getattr(config, "endpoint", None) or "").rstrip("/")
        self._api_key = str(
            getattr(config, "api_key", None) or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        )
        # Allow independent benchmark workers to use separate CLI locks while
        # preserving the catalog default for single-process runs.
        self._lock_path = Path(
            os.environ.get(
                "CLAUDE_CODE_LOCK_PATH",
                str(getattr(config, "lock_path", "/tmp/claude-opus55.lock")),
            )
        )
        self._current_game_state: dict[str, Any] | None = None

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
            "Do not call tools and do not include Markdown fences or free-form text."
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

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._base_url:
            env["ANTHROPIC_BASE_URL"] = self._base_url
        else:
            env.pop("ANTHROPIC_BASE_URL", None)
        if self._api_key:
            env["ANTHROPIC_AUTH_TOKEN"] = self._api_key
        else:
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
        return env

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]") if self._api_key else text

    def send_request(self, request_payload: dict[str, object]) -> object:
        prompt = str(request_payload.get("prompt") or "")
        screenshot_path = Path(str(request_payload.get("screenshot_path") or ""))
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if screenshot_path.is_file():
            media_type = mimetypes.guess_type(screenshot_path.name)[0] or "image/png"
            image_data = base64.b64encode(screenshot_path.read_bytes()).decode("ascii")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            })
        input_message = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": content},
        }, ensure_ascii=False)
        cmd = [
            self._claude_bin,
            "--bare",
            "--print",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
            "--model",
            self._model_name,
        ]
        retry_interval = max(
            0.0,
            float(getattr(self.config, "retry_interval_sec", 30.0)),
        )
        max_attempts = max(1, int(getattr(self.config, "max_attempts", 120)))
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, max_attempts + 1):
            timed_out = False
            with self._lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(Path.cwd()),
                        env=self._build_env(),
                        input=input_message + "\n",
                        capture_output=True,
                        text=True,
                        timeout=self.config.request_timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout = self._redact(str(exc.stdout or ""))
                    stderr = self._redact(str(exc.stderr or ""))
                    returncode = None
                else:
                    stdout = self._redact(proc.stdout)
                    stderr = self._redact(proc.stderr)
                    returncode = proc.returncode
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

            if returncode == 0:
                raw_stream = stdout
                extracted_text = ""
                usage: dict[str, object] = {}
                for line in raw_stream.splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "result":
                        result_text = event.get("result")
                        if isinstance(result_text, str) and result_text.strip():
                            extracted_text = result_text.strip()
                        event_usage = event.get("usage")
                        if isinstance(event_usage, dict):
                            usage = event_usage
                    elif event.get("type") == "assistant":
                        message = event.get("message")
                        if isinstance(message, dict):
                            message_content = message.get("content")
                            if isinstance(message_content, list):
                                text_parts = [
                                    item.get("text", "")
                                    for item in message_content
                                    if isinstance(item, dict)
                                    and isinstance(item.get("text"), str)
                                ]
                                if text_parts and not extracted_text:
                                    extracted_text = "\n".join(text_parts).strip()
                return {
                    "returncode": returncode,
                    "stdout": extracted_text or stdout,
                    "stderr": stderr,
                    "raw_stream": raw_stream,
                    "attempts": attempt,
                    "usage": usage,
                }

            if attempt >= max_attempts:
                return {
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "attempts": attempt,
                    "timed_out": timed_out,
                }

            reason = "request timeout" if timed_out else stderr.strip() or "no stderr"
            LOGGER.warning(
                "Claude Code request failed on attempt %d/%d; retrying in %.1fs: %s",
                attempt,
                max_attempts,
                retry_interval,
                reason,
            )
            time.sleep(retry_interval)

        raise AssertionError("unreachable")

    def extract_tool_call(self, response: object) -> dict[str, object] | None:
        if not isinstance(response, dict):
            return None
        return self._parse_action_payload(str(response.get("stdout") or ""))

    def extract_reasoning(self, response: object) -> str | None:
        action = self.extract_tool_call(response)
        if not action:
            return None
        arguments = action.get("arguments")
        if isinstance(arguments, dict):
            reasoning = arguments.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
        return None

    def extract_error(self, response: object) -> str | None:
        if not isinstance(response, dict):
            return "Claude Code returned an invalid response object"
        returncode = response.get("returncode")
        if returncode not in (0, None):
            stderr = str(response.get("stderr") or "").strip()
            return f"Claude Code failed with returncode={returncode}: {stderr or 'no stderr'}"
        if response.get("timed_out"):
            return "Claude Code request timed out after exhausting retries"
        if not self.extract_tool_call(response):
            return "Claude Code returned no valid semantic action"
        return None


__all__ = ["ClaudeCodeAgent", "ClaudeCodeConfig"]
