"""Native tool-call loop used by HyperAgents TaskAgent and MetaAgent."""

from __future__ import annotations

import json
import os
from pathlib import Path

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


def _record_model_attempt(usage, *, error=None):
    """Append one provider attempt before tool execution can fail or discard history."""
    destination = os.environ.get("HYPERAGENTS_TOKEN_LEDGER")
    live_destination = os.environ.get("HYPERAGENTS_META_LIVE_LEDGER")
    if not destination and not live_destination:
        return
    data = {"status": "failed_unknown" if error else "completed",
            "error_type": type(error).__name__ if error else None,
            "usage": {field: getattr(usage, field, None) for field in _USAGE_FIELDS} if usage else None}
    for path in (destination, live_destination):
        if not path:
            continue
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


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
    image_paths=None,
):
    """Run one model turn, including any native tool-call round trips."""
    loaded_tools = load_tools(logging=logging, names=tools_available or [])
    tools = {tool["info"]["name"]: tool for tool in loaded_tools}
    tool_schemas = _native_tools(tool["info"] for tool in loaded_tools)
    history = list(msg_history or ())
    usage_total = {field: 0 for field in _USAGE_FIELDS}
    usage_complete = True
    # Budget is enforced inside the container immediately after every model reply,
    # before another tool request can trigger an expensive model round-trip.
    meta_cap = int(os.environ.get("HYPERAGENTS_META_MAX_TOKENS", "0"))
    if meta_cap < 0:
        raise ValueError("HYPERAGENTS_META_MAX_TOKENS cannot be negative")
    # Screenshot inspection and editing are two chat_with_agent invocations in
    # the SAME MetaAgent process. Include their earlier calls in one cap.
    meta_spent = 0
    ledger_path = os.environ.get("HYPERAGENTS_TOKEN_LEDGER")
    if meta_cap and ledger_path and Path(ledger_path).exists():
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            value = (event.get("usage") or {}).get("total_tokens")
            if event.get("status") != "completed" or isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("MetaAgent prior usage is unknown; refusing next paid response")
            meta_spent += value
    if meta_cap and meta_spent >= meta_cap:
        raise RuntimeError("MetaAgent token cap exhausted before next model call")

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
        try:
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
            image_paths=image_paths if not history else None,
        )
        except Exception as exc:
            _record_model_attempt(None, error=exc)
            raise
        _record_model_attempt(response.usage)
        if response.usage is None or not isinstance(response.usage.total_tokens, int):
            usage_complete = False
        _merge_usage(usage_total, response.usage)
        logging(f"Output: {response.text!r}")
        if meta_cap and meta_spent + usage_total["total_tokens"] >= meta_cap:
            logging(f"MetaAgent token cap ({meta_cap}) reached; "
                    "saving current candidate without further tools or model calls.")
            break

        tool_calls = _parse_tool_calls(response.tool_calls)
        if not tool_calls:
            break
        if max_tool_calls > 0 and tool_call_count + len(tool_calls) > max_tool_calls:
            # Do not discard an already-edited candidate just because the model
            # requested one more tool. The caller can still save and evaluate the
            # current repository diff. Do not execute the over-budget calls.
            history[-1]["tool_budget_exhausted"] = True
            history[-1]["tool_calls_executed"] = tool_call_count
            history[-1]["tool_call_limit"] = max_tool_calls
            logging(f"Tool-call limit ({max_tool_calls}) reached; "
                    "saving the current candidate for evaluation without further tools.")
            break

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
        if max_tool_calls > 0 and tool_call_count >= max_tool_calls:
            # Stop before paying for another model turn that would be unable to
            # use tools. run_meta_agent.py persists the diff after this return.
            history[-1]["tool_budget_exhausted"] = True
            history[-1]["tool_calls_executed"] = tool_call_count
            history[-1]["tool_call_limit"] = max_tool_calls
            logging(f"Tool-call limit ({max_tool_calls}) reached; "
                    "saving the current candidate for evaluation without further tools.")
            break

    if history and history[-1].get("role") == "assistant":
        history[-1]["usage"] = usage_total if usage_complete else None
    return history


__all__ = ["chat_with_agent"]
