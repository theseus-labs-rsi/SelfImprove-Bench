"""Strict provider-reported token evidence for RSI agents (never estimate text length)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def nexau_usage_snapshot(raw_traces: Any) -> dict[str, Any]:
    """Extract a cumulative snapshot; one missing LLM span invalidates the sum."""
    from trace_converter import flatten_inmemory_spans, is_llm_span

    spans = flatten_inmemory_spans(raw_traces)
    calls = [span for span in spans if is_llm_span(span)]
    entries = []
    fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")
    for span in calls:
        output = span.get("output") or {}
        usage = output.get("usage") if isinstance(output, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
        values = {
            "input_tokens": _number(usage.get("input_tokens", usage.get("prompt_tokens"))),
            "output_tokens": _number(usage.get("output_tokens", usage.get("completion_tokens"))),
            "total_tokens": _number(usage.get("total_tokens", usage.get("totalTokens"))),
            "cached_input_tokens": _number(usage.get("cached_input_tokens", input_details.get("cached_tokens"))),
            "reasoning_tokens": _number(usage.get("reasoning_tokens", output_details.get("reasoning_tokens"))),
        }
        entries.append({"span_id": span.get("id"), **values, "raw_usage": usage})
    totals = {field: sum(item[field] for item in entries)
              if entries and all(item[field] is not None for item in entries) else None
              for field in fields}
    return {
        "trace_id": raw_traces[0].get("id") if isinstance(raw_traces, list) and raw_traces and isinstance(raw_traces[0], dict) else None,
        "model_calls": len(entries), "calls_with_usage": sum(item["total_tokens"] is not None for item in entries),
        **totals, "calls": entries,
    }



def hyperagents_usage_report(history_path: Path) -> dict[str, Any]:
    """Sum per-provider response records; historical cumulative histories are not enough."""
    ledger_path = history_path.with_name("model_attempts.jsonl")
    result = {"method": "hyperagents_model_attempts", "complete": False,
              "source_path": str(ledger_path), "history_path": str(history_path),
              "total_tokens": None, "calls": []}
    if not history_path.is_file() or not ledger_path.is_file():
        result["reason"] = "missing_history_or_attempt_ledger"
        return result
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError("MetaAgent history must be a list")
    if not any(isinstance(message, dict) and message.get("role") == "assistant" for message in history):
        result["reason"] = "no_completed_meta_agent_history"
        return result
    for index, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        usage = record.get("usage") or {}
        entry = {"attempt": index + 1, "status": record.get("status"),
                 "error_type": record.get("error_type")}
        for field in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
            entry[field] = _number(usage.get(field))
        result["calls"].append(entry)
    entries = result["calls"]
    result["complete"] = bool(entries) and all(e["status"] == "completed" and e["total_tokens"] is not None for e in entries)
    if not result["complete"]:
        result["reason"] = "missing_usage_or_failed_attempt"
    else:
        result["total_tokens"] = sum(e["total_tokens"] for e in entries)
    for target, field in (("prompt_num_tokens", "input_tokens"), ("response_num_tokens", "output_tokens"),
                          ("reasoning_num_tokens", "reasoning_tokens")):
        result[target] = sum(e[field] for e in entries) if result["complete"] and all(e[field] is not None for e in entries) else None
    return result
