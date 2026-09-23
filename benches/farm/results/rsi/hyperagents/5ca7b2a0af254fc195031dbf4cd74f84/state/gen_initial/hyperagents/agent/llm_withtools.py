"""Native tool-call loop used by HyperAgents TaskAgent and MetaAgent."""

from __future__ import annotations

import json

from agent.llm import MAX_TOKENS, generate_response
from agent.tools import load_tools


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)


def _native_tools(tool_infos) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": info["name"],
                "description": info.get("description", ""),
                "parameters": info["input_schema"],
            },
        }
        for info in tool_infos
    ]


def _parse_tool_calls(tool_calls) -> list[dict]:
    parsed = []
    for tool_call in tool_calls:
        if hasattr(tool_call, "name"):
            tool_id = getattr(tool_call, "id", None)
            tool_name = tool_call.name
            arguments = tool_call.arguments
        elif isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            tool_id = tool_call.get("id")
            tool_name = function.get("name")
            arguments = function.get("arguments", "{}")
        else:
            raise TypeError(f"Unsupported tool call type: {type(tool_call)!r}")
        try:
            tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            tool_input = None
        parsed.append(
            {
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            }
        )
    return parsed


def _execute_tool(tools, name, tool_input):
    if name not in tools:
        return f"Error: Tool {name!r} not found"
    if not isinstance(tool_input, dict):
        return "Error: Tool input must be a JSON object"
    try:
        return tools[name]["function"](**tool_input)
    except Exception as exc:
        return f"Error executing tool {name!r}: {exc}"


def _merge_usage(total: dict[str, int], usage) -> None:
    for field in _USAGE_FIELDS:
        total[field] += getattr(usage, field) or 0
    total["prompt_tokens"] = total["input_tokens"]
    total["completion_tokens"] = total["output_tokens"]


def chat_with_agent(
    prompt,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=None,
    multiple_tool_calls=False,
    max_tool_calls=40,
    reasoning_effort="medium",
    backend="openai",
    api_mode="responses",
    api_key_env="OPENAI_API_KEY",
    base_url_env="OPENAI_BASE_URL",
    max_tokens=MAX_TOKENS,
):
    """Run one model turn, including any native tool-call round trips."""
    loaded_tools = load_tools(logging=logging, names=tools_available or [])
    tools = {tool["info"]["name"]: tool for tool in loaded_tools}
    tool_schemas = _native_tools(tool["info"] for tool in loaded_tools)
    history = list(msg_history or ())
    usage_total = {field: 0 for field in _USAGE_FIELDS}

    current_prompt = prompt
    if tool_schemas:
        current_prompt = [
            {
                "role": "system",
                "content": (
                    "Use the supplied tools to inspect and edit the repository. "
                    "Do not claim that tool access is unavailable."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    tool_call_count = 0

    while True:
        logging(f"Input: {current_prompt!r}")
        response, history = generate_response(
            current_prompt,
            model=model,
            msg_history=history,
            tools=tool_schemas,
            tool_choice="auto" if tool_schemas else None,
            reasoning_effort=reasoning_effort,
            backend=backend,
            api_mode=api_mode,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            max_tokens=max_tokens,
            parallel_tool_calls=multiple_tool_calls,
        )
        _merge_usage(usage_total, response.usage)
        logging(f"Output: {response.text!r}")

        tool_calls = _parse_tool_calls(response.tool_calls)
        if not tool_calls:
            break
        if max_tool_calls > 0 and tool_call_count + len(tool_calls) > max_tool_calls:
            raise RuntimeError("Maximum number of tool calls reached")

        tool_messages = []
        for tool_call in tool_calls:
            tool_output = _execute_tool(
                tools,
                tool_call["name"],
                tool_call["input"],
            )
            tool_call_count += 1
            logging(
                f"Tool output: {tool_call['name']}({tool_call['input']!r}) "
                f"-> {tool_output!r}"
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_output),
                }
            )
        current_prompt = tool_messages

    if history and history[-1].get("role") == "assistant":
        history[-1]["usage"] = usage_total
    return history


__all__ = ["chat_with_agent"]
