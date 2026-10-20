"""GameWorld transport for any adapter implementing the public acting protocol."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agents.mm_agents.base.base_client import BaseClient

from .actor_input import USER_PROMPT, build_actor_input
from .network_retry import call_with_connection_retry


class RSIGameClient(BaseClient):
    def __init__(self, config: Any, rsi_agent: Any, **shared_tools: Any) -> None:
        super().__init__(config, **shared_tools)
        self._rsi_agent = rsi_agent

    def _parse_action(self, text: str) -> dict[str, Any] | None:
        candidates = [text.strip()]
        candidates.extend(match.group(1).strip() for match in re.finditer(
            r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE))
        decoder = json.JSONDecoder()
        for candidate in candidates:
            for index, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(candidate[index:])
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                name = payload.get("tool_name") or payload.get("name") or payload.get("action")
                if isinstance(name, str) and name.strip() in self._action_tool_names:
                    arguments = payload.get("arguments")
                    return {"tool_name": name.strip(),
                            "arguments": arguments if isinstance(arguments, dict) else {}}
        stripped = text.strip().strip("`\"'")
        return {"tool_name": stripped, "arguments": {}} if stripped in self._action_tool_names else None

    def get_action(self, screenshot_path: Path) -> dict[str, object] | None:
        system_prompt, memory_entries, prompt, image_paths = build_actor_input(self, screenshot_path)
        response = call_with_connection_retry(
            lambda: self._rsi_agent.act(prompt, image_paths=image_paths),
            max_attempts=int(os.environ.get("GAMEWORLD_CONNECTION_MAX_ATTEMPTS", "4")),
            audit_path=(Path(os.environ["GAMEWORLD_ACTION_AUDIT_PATH"])
                        if os.environ.get("GAMEWORLD_ACTION_AUDIT_PATH")
                        else screenshot_path.parent.parent / "token_attempts.jsonl"),
            invalidate_usage_on_retry=True,
        )
        action = self._parse_action(response.text)
        usage = None if response.usage is None else {
            name: getattr(response.usage, name) for name in (
                "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
        }
        return self._complete_action(
            screenshot_path=screenshot_path, raw_message_sent=prompt, raw_response=response.text,
            system_prompt=system_prompt, user_prompt=USER_PROMPT, memory_entries=memory_entries,
            tool_call=action, reasoning=response.reasoning,
            error=None if action is not None else "RSI agent returned no valid action JSON", usage=usage,
        )
