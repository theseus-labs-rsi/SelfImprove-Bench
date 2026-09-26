"""Thin message adapter between HyperAgents and its bundled backends."""

from __future__ import annotations

from functools import lru_cache

from benchmark.contracts import BackendResponse
from benchmark.backend import BackendDispatcher, OpenAIBackend


MAX_TOKENS = 16384


def _normalize_messages(msg_history, prompt) -> list[dict]:
    messages = []
    for history_message in msg_history or ():
        converted = dict(history_message)
        if "text" in converted and "content" not in converted:
            converted["content"] = converted.pop("text")
        for metadata_key in ("usage", "reasoning", "status", "finish_reason"):
            converted.pop(metadata_key, None)
        messages.append(converted)

    if isinstance(prompt, list):
        messages.extend(dict(message) for message in prompt)
    elif isinstance(prompt, dict):
        messages.append(dict(prompt))
    else:
        messages.append({"role": "user", "content": str(prompt)})
    return messages


def _restore_history(messages) -> list[dict]:
    history = []
    for message in messages:
        converted = dict(message)
        if "content" in converted:
            converted["text"] = converted.pop("content")
        history.append(converted)
    return history


@lru_cache(maxsize=None)
def _dispatcher(api_key_env: str, base_url_env: str) -> BackendDispatcher:
    return BackendDispatcher(
        {
            "openai": OpenAIBackend(
                api_key_env=api_key_env,
                base_url_env=base_url_env,
            ),
        }
    )


def generate_response(
    prompt,
    *,
    model: str,
    msg_history=None,
    tools=None,
    tool_choice=None,
    reasoning_effort=None,
    backend="openai",
    api_mode="responses",
    api_key_env="OPENAI_API_KEY",
    base_url_env="OPENAI_BASE_URL",
    max_tokens=MAX_TOKENS,
    temperature=0.0,
    parallel_tool_calls=False,
) -> tuple[BackendResponse, list[dict]]:
    """Generate one assistant turn and return its complete local history."""
    messages = _normalize_messages(msg_history, prompt)
    response = _dispatcher(api_key_env, base_url_env).distribute(
        backend,
        "",
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        api_mode=api_mode,
        max_output_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
    assistant_message = {
        "role": "assistant",
        "content": response.text,
        "status": response.status,
        "finish_reason": response.finish_reason,
    }
    if response.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            }
            for tool_call in response.tool_calls
        ]
    if response.reasoning:
        assistant_message["reasoning"] = response.reasoning
    messages.append(assistant_message)
    return response, _restore_history(messages)


__all__ = ["MAX_TOKENS", "generate_response"]
