"""Per-step actor input shared by the ACE, AHE and HyperAgents GameWorld clients.

Every method receives the same environment input at each step, mirroring the
text benchmark where all methods call ``agent.act(prompt)`` with one game-built
prompt: system prompt, action transport note, GameWorld's recent action history
(text only) and the current screenshot. Anything retained beyond this (for
example AHE's persistent NexAU conversation) belongs to the method itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


TRANSPORT_NOTE = """
# Action Transport
Return exactly one JSON object for every step:
{"tool_name":"<registered action id>","arguments":{"reasoning":"<short rationale>"}}
Do not use markdown fences or output additional text.
""".strip()

USER_PROMPT = "Game screen:"


def build_actor_input(client: Any, screenshot_path: Path) -> tuple[str, list[Any], str, list[Path]]:
    """Return ``(system_prompt, memory_entries, prompt, image_paths)`` for one step."""
    system_prompt = client.config.system_prompt or ""
    text_parts = [system_prompt, TRANSPORT_NOTE]
    memory_entries = client._collect_memory_context()
    if memory_entries:
        text_parts.append("# Recent Action History")
    for entry in memory_entries:
        if entry.type == "text":
            value = client._format_memory_text_entry(entry, as_action_history=True)
            if value:
                text_parts.append(value)
    text_parts.extend(("# Current Observation", USER_PROMPT))
    prompt = "\n\n".join(part for part in text_parts if part)
    return system_prompt, memory_entries, prompt, [screenshot_path]


__all__ = ["TRANSPORT_NOTE", "USER_PROMPT", "build_actor_input"]
