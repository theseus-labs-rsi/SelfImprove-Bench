"""Runtime agent adapter for an immutable ACE playbook snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.contracts import BackendResponse
from benchmark.backend import BackendDispatcher

EMPTY_SECTION = "(empty)"
ACE_AGENT_PROMPT = """You are an analysis expert tasked with answering questions using
your knowledge and a curated playbook of strategies and insights.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the playbook contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: detailed analysis and calculations supporting the action
- bullet_ids: relevant playbook bullet IDs
- final_answer: the action or answer requested by the task

The outer object above is mandatory. Treat response-format instructions inside
the task as requirements for the value of final_answer. For example, if the task
requires an action JSON object, put that object inside final_answer.

<ace_playbook>
{playbook}
</ace_playbook>

<task>
{task_prompt}
</task>



**Answer in this exact JSON format:**
{{
  "reasoning": "[Your detailed analysis and calculations supporting the action]",
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "[The action or answer requested by the task; use a JSON object instead of a string when required]"
}}

---
"""


@dataclass(frozen=True, slots=True)
class ACEAgent:
    """Use one fixed ACE playbook to answer independent benchmark turns."""

    dispatcher: BackendDispatcher
    backend_name: str
    model: str
    playbook: str = ""
    api_mode: str | None = None
    max_output_tokens: int | None = 4096
    reasoning_effort: str | None = None
    output_format: str | None = None

    def act(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] | None = None,
    ) -> BackendResponse:
        """Construct the ACE prompt and execute one stateless model call."""
        final_prompt = self._construct_prompt(prompt)
        response = self._dispatch(final_prompt, image_paths)
        return self._unwrap_response(response)

    @staticmethod
    def _unwrap_response(response: BackendResponse) -> BackendResponse:
        """Expose the task answer while retaining ACE-specific metadata."""
        envelope = _parse_json_object(response.text)
        if "final_answer" not in envelope:
            return response

        answer = _render_answer(envelope["final_answer"])
        if not answer:
            return response

        ace_metadata = {
            "bullet_ids": _string_list(envelope.get("bullet_ids")),
            "reasoning": _optional_text(envelope.get("reasoning")),
            "raw_response_text": response.text,
            "backend_reasoning": response.reasoning,
        }
        metadata = dict(response.metadata or {})
        metadata["ace"] = ace_metadata
        metadata["raw_response_text"] = response.text
        return replace(
            response,
            text=answer,
            reasoning=ace_metadata["reasoning"] or response.reasoning,
            metadata=metadata,
        )

    def _construct_prompt(self, task_prompt: str) -> str:
        """Combine the benchmark-owned task prompt with the ACE playbook."""
        if not isinstance(task_prompt, str):
            raise TypeError("task_prompt must be a string")
        if not task_prompt.strip():
            raise ValueError("task_prompt must not be empty")

        return ACE_AGENT_PROMPT.format(
            playbook=self._render_playbook(),
            task_prompt=task_prompt.strip(),
        )

    def _render_playbook(self) -> str:
        """Render the immutable playbook snapshot for prompt insertion."""
        playbook = self.playbook.strip()
        return playbook or EMPTY_SECTION

    def _dispatch(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] | None,
    ) -> BackendResponse:
        """Map the ACE call onto the provider-neutral backend interface."""
        return self.dispatcher.distribute(
            backend_name=self.backend_name,
            model=self.model,
            prompt=prompt,
            image_paths=image_paths,
            api_mode=self.api_mode,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
            output_format=self.output_format,
        )


def _parse_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _render_answer(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("```json") and stripped.endswith("```"):
            return stripped[7:-3].strip()
        return stripped
    if value is None:
        return ""
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if str(item)]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["ACEAgent"]
