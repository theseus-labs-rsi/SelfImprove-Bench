"""GameWorld client wrapper for a frozen HyperAgents RuntimeAgent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.mm_agents.base.base_client import BaseClient, BaseClientConfig


TRANSPORT_NOTE = """
# HyperAgents Action Transport
The registered actions above are available through a JSON transport rather than
native function calls. Return exactly one JSON object in this form:
{"tool_name":"<registered action id>","arguments":{"reasoning":"<short rationale>"}}
Do not use markdown fences or output any additional text.
"""


class HyperAgentsGameClient(BaseClient):
    def __init__(self, config: BaseClientConfig, runtime_agent: Any, **shared_tools: Any) -> None:
        super().__init__(config, **shared_tools)
        self._runtime_agent = runtime_agent

    def _parse_action(self, text: str) -> dict[str, Any] | None:
        candidates = [text.strip()]
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(
                r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE
            )
        )
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
                tool_name = (
                    payload.get("tool_name")
                    or payload.get("name")
                    or payload.get("action")
                )
                if isinstance(tool_name, str) and tool_name.strip() in self._action_tool_names:
                    arguments = payload.get("arguments")
                    return {
                        "tool_name": tool_name.strip(),
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    }
        stripped = text.strip().strip("`\"'")
        if stripped in self._action_tool_names:
            return {"tool_name": stripped, "arguments": {}}
        return None

    def get_action(self, screenshot_path: Path) -> dict[str, object] | None:
        system_prompt = self.config.system_prompt or ""
        user_prompt = "Game screen:"
        memory_entries = self._collect_memory_context()
        text_parts = [system_prompt, TRANSPORT_NOTE.strip()]
        image_paths: list[Path] = []
        if memory_entries:
            text_parts.append("# Recent Action History")
        for entry in memory_entries:
            if entry.type == "text":
                value = self._format_memory_text_entry(entry, as_action_history=True)
                if value:
                    text_parts.append(value)
            elif entry.type == "image":
                image_file = entry.image_file()
                if image_file is not None and image_file.exists():
                    image_paths.append(image_file)
        text_parts.extend(("# Current Observation", user_prompt))
        image_paths.append(screenshot_path)
        prompt = "\n\n".join(part for part in text_parts if part)
        response = self._runtime_agent.act(prompt, image_paths=image_paths)
        action = self._parse_action(response.text)
        error = None if action is not None else "HyperAgents returned no valid action JSON"
        usage = None
        if response.usage is not None:
            usage = {
                name: getattr(response.usage, name)
                for name in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                )
            }
        return self._complete_action(
            screenshot_path=screenshot_path,
            raw_message_sent=prompt,
            raw_response=response.text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            memory_entries=memory_entries,
            tool_call=action,
            reasoning=response.reasoning,
            error=error,
            usage=usage,
        )


__all__ = ["HyperAgentsGameClient", "TRANSPORT_NOTE"]
