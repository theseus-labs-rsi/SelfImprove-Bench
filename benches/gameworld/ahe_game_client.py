"""GameWorld client adapter for the RSI-Bench CSD AHE agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.mm_agents.base.base_client import BaseClient


TRANSPORT_NOTE = """
# AHE Action Transport
Return exactly one JSON object for every step:
{"tool_name":"<registered action id>","arguments":{"reasoning":"<short rationale>"}}
Do not use markdown fences or output additional text.
""".strip()


class AHEGameClient(BaseClient):
    """Preserve the GameWorld prompt/action contract around ``AHEAgent.act``.

    The CSD AHE implementation currently rejects image input.  We still pass
    the screenshot explicitly so the incompatibility is surfaced as a clear
    runtime error rather than silently evaluating a text-only variant.
    """

    def __init__(self, config: Any, ahe_agent: Any, **shared_tools: Any) -> None:
        super().__init__(config, **shared_tools)
        self._ahe_agent = ahe_agent

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
                tool_name = payload.get("tool_name") or payload.get("name") or payload.get("action")
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
        text_parts = [system_prompt, TRANSPORT_NOTE]
        memory_entries = self._collect_memory_context()
        if memory_entries:
            text_parts.append("# Recent Action History")
        for entry in memory_entries:
            if entry.type == "text":
                value = self._format_memory_text_entry(entry, as_action_history=True)
                if value:
                    text_parts.append(value)
        text_parts.extend(("# Current Observation", "Game screen:"))
        prompt = "\n\n".join(part for part in text_parts if part)

        # Keep image handling at the framework boundary. CSD AHE raises a
        # deliberate NotImplementedError here until its NexAU image mapping is
        # verified; swallowing it would invalidate the experiment.
        response = self._ahe_agent.act(prompt, image_paths=[screenshot_path])
        action = self._parse_action(response.text)
        error = None if action is not None else "AHE returned no valid action JSON"
        return self._complete_action(
            screenshot_path=screenshot_path,
            raw_message_sent=prompt,
            raw_response=response.text,
            system_prompt=system_prompt,
            user_prompt="Game screen:",
            memory_entries=memory_entries,
            tool_call=action,
            reasoning=response.reasoning,
            error=error,
            usage=None,
        )


__all__ = ["AHEGameClient", "TRANSPORT_NOTE"]
