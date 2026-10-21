"""Upstream text-tool protocol with benchmark transport and usage accounting."""

from __future__ import annotations

import json
import os
import re
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


def get_tooluse_prompt(tool_infos=()):
    if not tool_infos:
        return ""
    tools_available = "\n\n".join(str(info) for info in tool_infos)
    return """Here are the available tools:
```
{tools_available}
```

Use only one tool (if needed) in this format:
<json>
{{
    "tool_name": ...,
    "tool_input": ...
}}
</json>

ONLY USE ONE TOOL PER RESPONSE, AND STRICTLY FOLLOW THE FORMAT OF TOOL_NAME AND TOOL_INPUT ABOVE.
DO NOT HALLUCINATE OR MAKE UP ANYTHING.
""".format(tools_available=tools_available).strip()


def check_for_tool_uses(response):
    parsed = []
    for match in re.findall(r"<json>\s*(\{.*?\})\s*</json>", response, re.DOTALL):
        try:
            call = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and "tool_name" in call and "tool_input" in call:
            parsed.append(call)
    return parsed or None


def should_retry_tool_use(response, tool_uses=None):
    if tool_uses:
        return False
    positions = [response.find(marker) for marker in ("<json>", "tool_name", "tool_input")]
    return positions[0] >= 0 and positions[0] < positions[1] < positions[2] and len(response) >= 2000


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
    system_msg = get_tooluse_prompt([tool["info"] for tool in loaded_tools]) + "\n\n"
    history = list(msg_history or ())
    usage_total = {field: 0 for field in _USAGE_FIELDS}
    usage_complete = True
    # Budget is enforced inside the container immediately after every model reply,
    # before another tool request can trigger an expensive model round-trip.
    meta_cap = int(os.environ.get("HYPERAGENTS_META_MAX_TOKENS", "0"))
    if meta_cap < 0:
        raise ValueError("HYPERAGENTS_META_MAX_TOKENS cannot be negative")
    # Include earlier calls in the process-wide accounting cap.
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

    current_prompt = system_msg + prompt
    tool_call_count = 0

    while True:
        logging(f"Input: {current_prompt!r}")
        try:
            response, history = generate_response(
                current_prompt,
                model=model,
                msg_history=history,
                reasoning_effort=reasoning_effort,
                backend=backend,
                api_mode=api_mode,
                api_key_env=api_key_env,
                base_url_env=base_url_env,
                max_tokens=max_tokens,
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

        tool_calls = check_for_tool_uses(response.text)
        retry_tool_use = should_retry_tool_use(response.text, tool_calls)
        if not tool_calls and not retry_tool_use:
            break
        if max_tool_calls > 0 and tool_call_count >= max_tool_calls:
            history[-1]["tool_budget_exhausted"] = True
            history[-1]["tool_calls_executed"] = tool_call_count
            history[-1]["tool_call_limit"] = max_tool_calls
            logging(f"Tool-call limit ({max_tool_calls}) reached; "
                    "saving the current candidate for evaluation without further tools.")
            break

        tool_messages = []
        for tool_call in (tool_calls or []) if multiple_tool_calls else (tool_calls or [])[:1]:
            tool_output = _execute_tool(
                tools,
                tool_call["tool_name"],
                tool_call["tool_input"],
            )
            tool_call_count += 1
            logging(
                f"Tool output: {tool_call['tool_name']}({tool_call['tool_input']!r}) "
                f"-> {tool_output!r}"
            )
            tool_messages.append(
                f'<json>\n    {{\n        "tool_name": "{tool_call["tool_name"]}",\n'
                f'        "tool_input": {tool_call["tool_input"]},\n'
                f'        "tool_output": "{tool_output}"\n    }}\n</json>'
            )
        if retry_tool_use:
            tool_messages.append("Error: Output context exceeded. Please try again.")
        current_prompt = system_msg + "\n\n".join(tool_messages)

    if history and history[-1].get("role") == "assistant":
        history[-1]["usage"] = usage_total if usage_complete else None
    return history


__all__ = ["chat_with_agent"]
