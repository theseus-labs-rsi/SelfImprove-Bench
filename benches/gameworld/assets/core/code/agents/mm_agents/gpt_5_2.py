"""GPT-5.2 agent implementation."""

from __future__ import annotations

import os
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..harness.function_calling_utils import build_openai_action_tools
from .base.base_client import BaseClientConfig
from .base.generalist_agent import GeneralistAgent


@dataclass
class GPT52Config(BaseClientConfig):
    """Configuration for GPT-5.2."""

    model: str = "gpt-5.2"
    model_type: str = "generalist"
    api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    reasoning_effort: str = "medium"


class GPT52Agent(GeneralistAgent):
    """GPT-5.2 agent using the OpenAI Responses API."""

    def __init__(self, config: BaseClientConfig, **shared_tools):
        super().__init__(config, **shared_tools)
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            auth_path = Path("/root/.codex/auth.json")
            if auth_path.is_file():
                try:
                    auth_payload = json.loads(auth_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    auth_payload = {}
                api_key = auth_payload.get("OPENAI_API_KEY")
        self._api_key = self._resolve_api_key(api_key, env_vars=("OPENAI_API_KEY",))
        self._base_url = (config.endpoint or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._request_timeout = config.request_timeout
        self._model_name = config.model

    def build_tools(self) -> list[dict[str, object]]:
        return build_openai_action_tools(self._semantic_controls_specs)

    def build_request_payload(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[object],
        tools: list[dict[str, object]],
        screenshot_path: Path,
    ) -> dict[str, object]:
        input_content = self._build_user_content(
            memory_entries=memory_entries,
            append_user_text=lambda text: {"type": "input_text", "text": text},
            append_user_image=lambda image_file: {
                "type": "input_image",
                "image_url": self._build_data_url(image_file),
            },
            user_prompt=user_prompt,
            screenshot_path=screenshot_path,
        )
        request_payload: dict[str, object] = {
            "model": self._model_name,
            "max_output_tokens": self.config.max_tokens,
            "reasoning": {"effort": self.config.reasoning_effort},
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {"type": "message", "role": "user", "content": input_content},
            ],
        }
        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = "required"
        return request_payload

    def send_request(self, request_payload: dict[str, object]) -> object:
        url = f"{self._base_url}/responses"
        body = json.dumps(request_payload).encode("utf-8")
        # Keep the current browser/game episode alive across transient gateway
        # failures. Retrying after the process exits would start a new game.
        max_attempts = 10
        for attempt in range(max_attempts):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._request_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                    self._logger.warning(
                        "Transient Responses API HTTP error on attempt %d/%d: HTTP %d; retrying",
                        attempt + 1, max_attempts, exc.code,
                    )
                    time.sleep(min(30, 2**attempt))
                    continue
                raise RuntimeError(
                    f"Responses API request failed: HTTP {exc.code}: {error_body}"
                ) from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Responses API request failed after {max_attempts} transient-error attempts"
                    ) from exc
                self._logger.warning(
                    "Transient Responses API error on attempt %d/%d: %s; retrying",
                    attempt + 1, max_attempts,
                    exc,
                )
                time.sleep(min(30, 2**attempt))

        raise AssertionError("unreachable")

    def extract_tool_call(self, response: object) -> dict[str, object] | None:
        return self._extract_tool_call_from_output_items(
            self._extract_response_output_items(response)
        )

__all__ = ["GPT52Agent", "GPT52Config"]
