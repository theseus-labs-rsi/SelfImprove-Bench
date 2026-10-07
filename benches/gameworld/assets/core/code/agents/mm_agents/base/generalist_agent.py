"""Template flow for semantic-control agents."""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

from .base_client import BaseClient


class GeneralistAgent(BaseClient):
    """Shared request/response flow for semantic-control agents."""

    def prepare_prompt(
        self,
        screenshot_path: Path,
    ) -> tuple[str | None, str, list[Any]]:
        del screenshot_path
        return self._prepare_multimodal_prompt_and_memory()

    def build_tools(self) -> list[dict[str, Any]]:
        return []

    @abstractmethod
    def build_request_payload(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[Any],
        tools: list[dict[str, Any]],
        screenshot_path: Path,
    ) -> dict[str, Any]:
        """Build the provider-specific request payload."""

    @abstractmethod
    def send_request(self, request_payload: dict[str, Any]) -> Any:
        """Send the request payload to the provider."""

    @abstractmethod
    def extract_tool_call(self, response: Any) -> dict[str, object] | None:
        """Extract one semantic tool call from the provider response."""

    def extract_reasoning(self, response: Any) -> str | None:
        del response
        return None

    def extract_error(self, response: Any) -> str | None:
        del response
        return None

    @classmethod
    def _extract_diagnostic_text(cls, response: Any, raw_response: str) -> str:
        """Extract diagnosis text, including OpenAI-compatible proxy responses."""
        message = cls._extract_first_choice_message(response)
        text = cls._extract_message_text(message).strip()
        if text:
            return text
        try:
            payload = json.loads(raw_response)
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return ""
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _build_diagnostic_system_prompt(
        system_prompt: str | None,
        runtime_advisory: str | None,
    ) -> str:
        """Keep task/control constraints available during diagnosis."""
        return (
            f"{(system_prompt or '').rstrip()}\n\n"
            "# Diagnostic Reasoning Stage\n"
            "Do not choose or call an action yet. Compare the current screenshot "
            "with the recent screenshot/action history, while obeying all game, "
            "task, and registered-action constraints above. Return exactly three "
            "concise labeled lines:\n"
            "Evidence: what changed or failed to change.\n"
            "Cause: one likely reason the repeated action failed.\n"
            "Recovery: one next-step objective, naming only a registered action "
            "when an action is named.\n\n"
            "Runtime signal:\n"
            f"{runtime_advisory or ''}"
        )

    def get_action(self, screenshot_path: Path) -> dict[str, object] | None:
        system_prompt, user_prompt, memory_entries = self.prepare_prompt(screenshot_path)
        tools = self.build_tools()
        diagnostic_reasoning: str | None = None
        diagnostic_raw_message_sent: str | None = None
        diagnostic_raw_response: str | None = None
        if self._runtime_reasoning_required:
            diagnostic_system_prompt = self._build_diagnostic_system_prompt(
                system_prompt,
                self._runtime_advisory,
            )
            diagnostic_payload = self.build_request_payload(
                system_prompt=diagnostic_system_prompt,
                user_prompt=user_prompt,
                memory_entries=memory_entries,
                tools=[],
                screenshot_path=screenshot_path,
            )
            diagnostic_raw_message_sent = self._stringify_raw_message_sent(
                diagnostic_payload
            )
            diagnostic_response = self.send_request(diagnostic_payload)
            diagnostic_raw_response = self._stringify_raw_response(diagnostic_response)
            diagnostic_reasoning = self._extract_diagnostic_text(
                diagnostic_response,
                diagnostic_raw_response,
            )
            if diagnostic_reasoning:
                system_prompt = (
                    f"{(system_prompt or '').rstrip()}\n\n"
                    "# Generated Diagnostic Reasoning\n"
                    f"{diagnostic_reasoning}\n\n"
                    "Use this diagnosis to select the next action. Now follow the "
                    "registered action interface and original output format."
                )
        request_payload = self.build_request_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            memory_entries=memory_entries,
            tools=tools,
            screenshot_path=screenshot_path,
        )
        raw_message_sent = self._stringify_raw_message_sent(request_payload)
        response = self.send_request(request_payload)
        raw_response = self._stringify_raw_response(response)

        tool_call: dict[str, object] | None = None
        reasoning: str | None = diagnostic_reasoning
        error: str | None = None
        try:
            tool_call = self.extract_tool_call(response)
            reasoning = self.extract_reasoning(response) or diagnostic_reasoning
            error = self.extract_error(response)
        except Exception as exc:
            error = f"Failed to parse tool call: {exc}"
            self._logger.warning(error)

        return self._complete_action(
            screenshot_path=screenshot_path,
            raw_message_sent=raw_message_sent,
            raw_response=raw_response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            memory_entries=memory_entries,
            tool_call=tool_call,
            reasoning=reasoning,
            diagnostic_raw_message_sent=diagnostic_raw_message_sent,
            diagnostic_raw_response=diagnostic_raw_response,
            error=error,
            usage=self.extract_usage(response),
        )
