"""Parse and validate one batch of sweepmine induction answers."""

import json
import re
from collections.abc import Sequence
from typing import Any


def _validate_board(board: Any, answer_map: Sequence[Sequence[int | str]], task_id: int) -> None:
    if type(board) is not list or len(board) != len(answer_map):
        raise ValueError(f"Task {task_id}: board row count does not match")
    for row_index, (row, answer_row) in enumerate(zip(board, answer_map)):
        if type(row) is not list or len(row) != len(answer_row):
            raise ValueError(f"Task {task_id}: board column count does not match")
        for col_index, (value, expected) in enumerate(zip(row, answer_row)):
            if expected == "M" and value != "M":
                raise ValueError(f"Task {task_id}: expected M at ({row_index}, {col_index})")
            if expected != "M" and (type(value) is not int or value < 0):
                raise ValueError(f"Task {task_id}: expected a non-negative integer at ({row_index}, {col_index})")


def _validate_answer(answer: Any, answer_map: Sequence[Sequence[int | str]], task_id: int) -> None:
    if type(answer) is not dict or set(answer) != {"task_id", "board", "reasoning"}:
        raise ValueError(f"Task {task_id}: expected task_id, board, and reasoning")
    if type(answer["task_id"]) is not int or answer["task_id"] != task_id:
        raise ValueError(f"Task {task_id}: task_id is missing or out of order")
    if type(answer["reasoning"]) is not str or not answer["reasoning"].strip():
        raise ValueError(f"Task {task_id}: reasoning must be a non-empty string")
    _validate_board(answer["board"], answer_map, task_id)


def parse_response(
    response_text: str, answer_maps: Sequence[Sequence[Sequence[int | str]]]
) -> list[dict[str, Any] | None]:
    """Return None for each missing or malformed task answer."""
    payload = _load_payload(response_text)
    if payload is None:
        raise ValueError("response must contain a readable JSON object")
    if type(payload) is not dict or set(payload) != {"answers"}:
        raise ValueError("response must contain only an answers array")
    answers = payload["answers"]
    if type(answers) is not list:
        raise ValueError("answers must be an array")
    parsed: list[dict[str, Any] | None] = []
    for task_id, answer_map in enumerate(answer_maps, start=1):
        answer = answers[task_id - 1] if task_id <= len(answers) else None
        try:
            _validate_answer(answer, answer_map, task_id)
        except ValueError:
            parsed.append(None)
        else:
            parsed.append(answer)
    return parsed


def _load_payload(response_text: str) -> Any | None:
    """Decode JSON from direct, fenced, ACE-wrapped, or lightly prefixed output."""
    candidates = [response_text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE))
    start, end = response_text.find("{"), response_text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(response_text[start:end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "final_answer" in payload:
            answer = payload["final_answer"]
            if isinstance(answer, str):
                nested = _load_payload(answer)
                if nested is not None:
                    return nested
            elif isinstance(answer, dict):
                return answer
        return payload
    return None
