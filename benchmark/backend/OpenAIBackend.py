from __future__ import annotations

import base64
import mimetypes
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .Backend import BackendRequest, BackendResponse, TokenUsage, ToolCall


SUPPORTED_IMAGE_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
RESPONSES_MODE = "responses"
CHAT_COMPLETIONS_MODE = "chat_completions"
JSON_OBJECT_FORMAT = "json_object"


class OpenAIBackend:
    """OpenAI adapter for Responses and Chat Completions APIs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        api_mode: str = RESPONSES_MODE,
        max_output_tokens: int = 512,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client_kwargs = {}
            resolved_api_key = api_key or os.getenv(api_key_env)
            resolved_base_url = base_url or os.getenv(base_url_env)
            if resolved_api_key:
                client_kwargs["api_key"] = resolved_api_key
            if resolved_base_url:
                client_kwargs["base_url"] = resolved_base_url
            client = OpenAI(**client_kwargs)

        self._client = client
        self._api_mode = api_mode
        self._max_output_tokens = max_output_tokens

    def generate(self, request: BackendRequest) -> BackendResponse:
        api_mode = request.api_mode or self._api_mode
        if api_mode == RESPONSES_MODE:
            return self._generate_responses(request)
        if api_mode == CHAT_COMPLETIONS_MODE:
            return self._generate_chat_completions(request)
        raise ValueError(f"Unsupported OpenAI API mode: {api_mode!r}")

    def _generate_responses(self, request: BackendRequest) -> BackendResponse:
        if request.messages and request.image_paths:
            raise ValueError("messages and image_paths cannot be combined")
        payload = {
            "model": request.model.removeprefix("openai/"),
            "input": (
                self._build_responses_messages(request.messages)
                if request.messages
                else self._build_responses_input(
                    request.prompt,
                    request.image_paths,
                )
            ),
            "max_output_tokens": (
                request.max_output_tokens or self._max_output_tokens
            ),
            "store": False,
        }
        if request.reasoning_effort is not None:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.output_format is not None:
            self._validate_output_format(request.output_format)
            payload["text"] = {"format": {"type": request.output_format}}
        if request.tools:
            payload["tools"] = self._build_responses_tools(request.tools)
            payload["tool_choice"] = request.tool_choice or "auto"
            payload["parallel_tool_calls"] = request.parallel_tool_calls

        response = self._client.responses.create(**payload)
        return BackendResponse(
            text=(getattr(response, "output_text", "") or "").strip(),
            reasoning=self._extract_reasoning(response),
            usage=self._map_usage(getattr(response, "usage", None)),
            tool_calls=self._extract_responses_tool_calls(response),
            provider="openai",
            model=getattr(response, "model", None) or request.model,
            status=self._as_text(getattr(response, "status", None)),
            finish_reason=self._extract_finish_reason(response),
            response_id=self._as_text(getattr(response, "id", None)),
        )

    def _generate_chat_completions(
        self,
        request: BackendRequest,
    ) -> BackendResponse:
        if request.messages and request.image_paths:
            raise ValueError("messages and image_paths cannot be combined")
        model = request.model.removeprefix("openai/")
        payload = {
            "model": model,
            "messages": (
                [dict(message) for message in request.messages]
                if request.messages
                else self._build_chat_messages(
                    request.prompt,
                    request.image_paths,
                )
            ),
        }
        token_key = "max_completion_tokens" if "gpt-5" in model else "max_tokens"
        payload[token_key] = request.max_output_tokens or self._max_output_tokens
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.temperature is not None and "gpt-5" not in model:
            payload["temperature"] = request.temperature
        if request.output_format is not None:
            self._validate_output_format(request.output_format)
            payload["response_format"] = {"type": request.output_format}
        if request.tools:
            payload["tools"] = [dict(tool) for tool in request.tools]
            payload["tool_choice"] = request.tool_choice or "auto"
            payload["parallel_tool_calls"] = request.parallel_tool_calls

        response = self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message
        return BackendResponse(
            text=self._extract_chat_text(getattr(message, "content", None)),
            reasoning=getattr(message, "reasoning_content", None),
            usage=self._map_usage(getattr(response, "usage", None)),
            tool_calls=self._extract_chat_tool_calls(message),
            provider="openai",
            model=getattr(response, "model", None) or request.model,
            status="completed",
            finish_reason=self._as_text(getattr(choice, "finish_reason", None)),
            response_id=self._as_text(getattr(response, "id", None)),
        )

    @classmethod
    def _build_responses_messages(
        cls,
        messages: tuple[Mapping[str, Any], ...],
    ) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            if message.get("role") == "tool":
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": str(message.get("content", "")),
                    }
                )
                continue
            content = message.get("content", message.get("text", ""))
            if content:
                converted.append({"role": message["role"], "content": content})
            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or ():
                    function = tool_call.get("function", {})
                    converted.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call["id"],
                            "name": function["name"],
                            "arguments": function.get("arguments", "{}"),
                        }
                    )
        return converted

    @staticmethod
    def _build_responses_tools(
        tools: tuple[Mapping[str, Any], ...],
    ) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            function = tool.get("function", tool)
            converted.append(
                {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": function["parameters"],
                }
            )
        return converted

    @classmethod
    def _build_responses_input(
        cls,
        prompt: str,
        image_paths: tuple[Path, ...],
    ) -> str | list[dict[str, Any]]:
        if not image_paths:
            return prompt

        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt}
        ]
        content.extend(
            {
                "type": "input_image",
                "image_url": cls._encode_image(path),
            }
            for path in image_paths
        )
        return [{"role": "user", "content": content}]

    @classmethod
    def _build_chat_messages(
        cls,
        prompt: str,
        image_paths: tuple[Path, ...],
    ) -> list[dict[str, Any]]:
        if not image_paths:
            return [{"role": "user", "content": prompt}]

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": cls._encode_image(path)},
            }
            for path in image_paths
        )
        return [{"role": "user", "content": content}]

    @staticmethod
    def _encode_image(path: Path) -> str:
        resolved_path = path.expanduser().resolve(strict=True)
        if not resolved_path.is_file():
            raise ValueError(f"Image path is not a file: {resolved_path}")

        media_type = mimetypes.guess_type(resolved_path.name)[0]
        if media_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image type: {resolved_path}")

        encoded = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    @staticmethod
    def _extract_chat_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        texts = []
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                texts.append(str(text))
        return "\n".join(texts).strip()

    @classmethod
    def _extract_responses_tool_calls(
        cls,
        response: Any,
    ) -> tuple[ToolCall, ...]:
        calls = []
        for item in getattr(response, "output", ()) or ():
            if cls._get_value(item, "type") != "function_call":
                continue
            calls.append(
                ToolCall(
                    id=cls._as_text(cls._get_value(item, "call_id")),
                    name=str(cls._get_value(item, "name")),
                    arguments=cls._get_value(item, "arguments") or "{}",
                )
            )
        return tuple(calls)

    @classmethod
    def _extract_chat_tool_calls(
        cls,
        message: Any,
    ) -> tuple[ToolCall, ...]:
        calls = []
        for tool_call in cls._get_value(message, "tool_calls") or ():
            function = cls._get_value(tool_call, "function") or {}
            calls.append(
                ToolCall(
                    id=cls._as_text(cls._get_value(tool_call, "id")),
                    name=str(cls._get_value(function, "name")),
                    arguments=cls._get_value(function, "arguments") or "{}",
                )
            )
        return tuple(calls)

    @staticmethod
    def _validate_output_format(output_format: str) -> None:
        if output_format != JSON_OBJECT_FORMAT:
            raise ValueError(
                f"Unsupported OpenAI output format: {output_format!r}"
            )

    @classmethod
    def _extract_reasoning(cls, response: Any) -> str | None:
        summaries: list[str] = []
        for item in getattr(response, "output", ()) or ():
            if cls._get_value(item, "type") != "reasoning":
                continue
            summaries.extend(
                cls._extract_summary_texts(cls._get_value(item, "summary"))
            )
        if summaries:
            return "\n".join(summaries)

        reasoning = cls._get_value(response, "reasoning")
        summaries = cls._extract_summary_texts(
            cls._get_value(reasoning, "summary")
        )
        return "\n".join(summaries) or None

    @staticmethod
    def _extract_summary_texts(summary: Any) -> list[str]:
        if summary is None:
            return []

        parts = summary if isinstance(summary, (list, tuple)) else (summary,)
        texts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                text = part
            elif isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text and str(text).lower() not in {"auto", "concise", "detailed"}:
                texts.append(str(text))
        return texts

    @staticmethod
    def _map_usage(usage: Any) -> TokenUsage | None:
        if usage is None:
            return None

        input_details = OpenAIBackend._get_value(
            usage,
            "input_tokens_details",
            "prompt_tokens_details",
        )
        output_details = OpenAIBackend._get_value(
            usage,
            "output_tokens_details",
            "completion_tokens_details",
        )
        return TokenUsage(
            input_tokens=OpenAIBackend._get_value(
                usage,
                "input_tokens",
                "prompt_tokens",
            ),
            output_tokens=OpenAIBackend._get_value(
                usage,
                "output_tokens",
                "completion_tokens",
            ),
            total_tokens=OpenAIBackend._get_value(usage, "total_tokens"),
            cached_input_tokens=OpenAIBackend._get_value(
                input_details,
                "cached_tokens",
            ),
            reasoning_tokens=OpenAIBackend._get_value(
                output_details,
                "reasoning_tokens",
            ),
        )

    @staticmethod
    def _get_value(source: Any, *names: str) -> Any:
        for name in names:
            value = (
                source.get(name)
                if isinstance(source, dict)
                else getattr(source, name, None)
            )
            if value is not None:
                return value
        return None

    @classmethod
    def _extract_finish_reason(cls, response: Any) -> str | None:
        incomplete = getattr(response, "incomplete_details", None)
        return cls._as_text(getattr(incomplete, "reason", None))

    @staticmethod
    def _as_text(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "value", value))
