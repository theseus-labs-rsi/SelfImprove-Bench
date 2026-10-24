"""Play orchestrator for Sweepmine.

Responsibilities (to be filled in from the design discussion):
- Accept a core (board) and an agent.
- Drive the loop: agent action -> core transition -> observation history.
- Own per-episode trajectory text; never mutate shared/global state so that
  multiple workers can run independent episodes concurrently.
- Prompt construction is intentionally separated from this file.

Import safety: only imports the sibling ``core`` module (one-way dependency,
no circular import). Import style is package-relative.
"""

from __future__ import annotations

import json
import re
from typing import Any

from benchmark.contracts import (
    BackendResponse,
    AgentCallRecord,
    ExecutionStatus,
    FeedbackResult,
    StepFeedback,
    TaskEvaluation,
)

from .core import SweepmineCore, SweepmineEvent, SweepmineState
from .prompt import build_initial_prompt, build_turn_prompt

def merge_prompt(base: str, addition: str) -> str:
    """Append a new observation/feedback block to the running prompt."""
    return base + "\n" + addition


def render_board(view: list[list]) -> str:
    """Render one board view to text.

    Cells are space-joined per row and rows newline-separated, e.g.
    "0 0 0\n0 1 0\n? ? ?". ``view`` must be the model-visible mask
    (inspect(): None -> '?', marks -> 'M'), never the ground truth.
    """
    return "\n".join(
        " ".join("?" if cell is None else str(cell) for cell in row)
        for row in view
    )


def serialize_board(view: list[list]) -> list[list]:
    """Preserve board structure while making hidden-cell semantics explicit."""
    return [
        ["?" if cell is None else cell for cell in row]
        for row in view
    ]

def resolve_act(text: str, task: SweepmineCore) -> SweepmineEvent:
    """Parse the agent's single-line JSON act and apply it to the board.

    Returns the action outcome event, which resolve_response turns into text.
    """
    try:
        payload = json.loads(extract_action(text).strip())
        act = payload["act"]
        row = int(payload["row"])
        col = int(payload["col"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        # TODO(core): provide an explicit invalid_act() entry so unparsable
        # actions charge a step without the out-of-range click fallback.
        return task.click(-1, -1)

    if act == "click":
        return task.click(row, col)
    if act == "mark":
        return task.mark(row, col)
    if act == "unmark":
        return task.unmark(row, col)
    # Unknown action name: same abnormal handling as above.
    return task.click(-1, -1)


def resolve_response(code: SweepmineEvent) -> str:
    """Map an action outcome event to the feedback text shown to the model."""
    if code == SweepmineEvent.NORMAL_ACTION:
        return "Result: This action has been done. The task is still not finished."
    if code == SweepmineEvent.ABNORMAL_ACTION:
        return "Result: This action is illegal. The step is wasted"
    if code == SweepmineEvent.SUCCESS:
        return "Result: The game Succeeded!"
    if code == SweepmineEvent.RESTART:
        return "Result: You clicked the target. The game has restarted."
    # FAIL: step budget exhausted (not an illegal action)
    return "Result: The step budget is exhausted."
def play_feedback(task: SweepmineCore, agent: Any) -> FeedbackResult:
    """Run one complete Sweepmine task and retain structured feedback."""
    max_steps = task.max_steps
    task_snapshot = snapshot_task(task)
    trajectory: list[StepFeedback] = []
    static = build_initial_prompt()
    history: list[str] = []

    while task.status == SweepmineState.ONGOING:
        prompt = build_turn_prompt(static, history, render_board(task.inspect()))
        try:
            response_text, agent_call = _call_agent(agent, prompt)
        except Exception as exc:
            return build_error_feedback(task, task_snapshot, trajectory, exc)

        action = extract_action(response_text)
        board_before = task.inspect()
        event = resolve_act(action, task)
        feedback = resolve_response(event)
        board_after = task.inspect()
        environment_response = {
            "event": event.name.lower(),
            "message": feedback,
            "board": serialize_board(board_after),
            "remaining_steps": task.steps,
            "restart_count": task.restart_count,
        }
        trajectory.append(
            StepFeedback(
                step_index=len(trajectory),
                prompt=prompt,
                image_records=(),
                action=action,
                environment_response=environment_response,
                success=_event_success(event, board_before, board_after),
                score=None,
                feedback=feedback,
                agent_call=agent_call,
            )
        )
        if event == SweepmineEvent.RESTART:
            history.clear()
        elif task.status == SweepmineState.ONGOING:
            history.append(f"{action} -> {feedback}")

    success = task.status == SweepmineState.SUCCESS
    consumed = max_steps - task.steps
    # Omniscient optimum: mark each mine once, without exploratory actions.
    optimal_steps = task.mine_count
    if success:
        slack = max_steps - optimal_steps
        score = (max_steps - consumed) / slack if slack > 0 else 1.0
    else:
        score = 0.0
    final_feedback = (
        "The task succeeded within the step budget."
        if success
        else "The task ended because the step budget was exhausted."
    )
    return FeedbackResult(
        task_id=task.task_id,
        task=task_snapshot,
        trajectory=tuple(trajectory),
        evaluation=TaskEvaluation(
            execution_status=ExecutionStatus.COMPLETED,
            success=success,
            score=score,
            feedback=final_feedback,
            metrics={
                "steps_consumed": consumed,
                "optimal_steps": optimal_steps,
                "restart_count": task.restart_count,
                "max_steps": max_steps,
            },
        ),
    )


def play(task: SweepmineCore, agent: Any) -> dict[str, Any]:
    """Backward-compatible dictionary result for the original experiments."""
    result = play_feedback(task, agent)
    evaluation = result.evaluation
    raw_actions = [
        step.agent_call.response_text if step.agent_call else step.action
        for step in result.trajectory
    ]
    return {
        "success": evaluation.success,
        "score": evaluation.score,
        "trace": _render_legacy_trace(result),
        "steps_consumed": evaluation.metrics["steps_consumed"],
        "restart_count": evaluation.metrics["restart_count"],
        "raw_actions": raw_actions,
        "config": dict(result.task),
    }


def extract_action(response_text: str) -> str:
    """Unwrap one complete JSON code fence, without interpreting method envelopes."""
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
        response_text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        inner = fenced.group(1).strip()
        try:
            payload = json.loads(inner)
        except (ValueError, TypeError):
            return response_text
        if isinstance(payload, dict):
            return inner
    return response_text


def _call_agent(agent: Any, prompt: str) -> tuple[str, AgentCallRecord | None]:
    """Call the common act interface, with legacy play support."""
    if hasattr(agent, "act"):
        response = agent.act(prompt)
    elif hasattr(agent, "play"):
        response = agent.play(prompt)
    else:
        raise TypeError("Agent must provide act(prompt) or legacy play(prompt)")

    if isinstance(response, str):
        return response, None
    if not isinstance(response, BackendResponse):
        raise TypeError("Agent act() must return str or BackendResponse")
    return response.text, AgentCallRecord(
        response_text=response.text,
        reasoning=response.reasoning,
        usage=response.usage,
        provider=response.provider,
        model=response.model,
        status=response.status,
        finish_reason=response.finish_reason,
        metadata=response.metadata,
    )


def snapshot_task(task: SweepmineCore) -> dict[str, Any]:
    """Capture public construction data before the mutable task runs."""
    return {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "seed": task.seed,
        "rows": task.rows,
        "cols": task.cols,
        "mine_count": task.mine_count,
        "max_steps": task.max_steps,
    }


def _event_success(
    event: SweepmineEvent,
    board_before: list[list],
    board_after: list[list],
) -> bool:
    if event == SweepmineEvent.SUCCESS:
        return True
    return event == SweepmineEvent.NORMAL_ACTION and board_after != board_before


def build_error_feedback(
    task: SweepmineCore,
    task_snapshot: dict[str, Any],
    trajectory: list[StepFeedback],
    exc: Exception,
) -> FeedbackResult:
    error = f"{type(exc).__name__}: {exc}"
    return FeedbackResult(
        task_id=task.task_id,
        task=task_snapshot,
        trajectory=tuple(trajectory),
        evaluation=TaskEvaluation(
            execution_status=ExecutionStatus.ERROR,
            success=False,
            score=0.0,
            feedback=error,
            metrics={
                "steps_consumed": task.max_steps - task.steps,
                "restart_count": task.restart_count,
                "max_steps": task.max_steps,
            },
        ),
    )


def _render_legacy_trace(result: FeedbackResult) -> str:
    parts: list[str] = []
    for step in result.trajectory:
        parts.extend((step.prompt, step.action, step.feedback or ""))
    return "\n".join(parts)


__all__ = [
    "extract_action",
    "build_error_feedback",
    "merge_prompt",
    "play",
    "play_feedback",
    "render_board",
    "serialize_board",
    "resolve_act",
    "resolve_response",
    "snapshot_task",
]
