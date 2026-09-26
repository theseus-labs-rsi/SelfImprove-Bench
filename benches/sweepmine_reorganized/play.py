"""One-call evaluation and feedback for Sweepmine rule induction."""

import json
from typing import Any, Mapping

from benchmark.contracts import (
    AgentCallRecord, BackendResponse, ExecutionStatus, FeedbackResult, StepFeedback, TaskEvaluation,
)

from .response_parser import parse_response
from .scorer import score_answers, score_details
from .task_organizer import get_task


def _call_agent(agent: Any, prompt: str) -> tuple[str, AgentCallRecord | None]:
    """Make exactly one model call and retain its original response."""
    response = agent.act(prompt)
    if isinstance(response, str):
        return response, None
    if not isinstance(response, BackendResponse):
        raise TypeError("Agent act() must return str or BackendResponse")
    call = AgentCallRecord(
        response_text=response.text, reasoning=response.reasoning, usage=response.usage,
        provider=response.provider, model=response.model, status=response.status,
        finish_reason=response.finish_reason, metadata=response.metadata,
    )
    return response.text, call


def _answer_text(response_text: str) -> str:
    """Accept both direct JSON answers and an ACE final_answer envelope."""
    try:
        payload = json.loads(response_text)
    except (TypeError, json.JSONDecodeError):
        return response_text
    if not isinstance(payload, dict) or "final_answer" not in payload:
        return response_text
    answer = payload["final_answer"]
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def play_feedback(task: Mapping[str, Any], agent: Any) -> FeedbackResult:
    """Construct, ask once, parse every query, and return one batch score."""
    prompt, answer_maps, task_details = get_task(
        dif=int(task["difficulty"]), task_counts=int(task["task_counts"]),
        seed=int(task["seed"]), core_mines=int(task["core_mines"]),
        example_counts=int(task["example_counts"]),
    )
    task_id = str(task["task_id"])
    try:
        response_text, agent_call = _call_agent(agent, prompt)
    except Exception as exc:
        return FeedbackResult(
            task_id=task_id, task=dict(task), trajectory=(),
            evaluation=TaskEvaluation(
                execution_status=ExecutionStatus.ERROR, success=False, score=0.0,
                feedback=f"Agent call failed: {type(exc).__name__}: {exc}",
                metrics={"query_count": len(answer_maps), "score_percent": 0.0},
            ),
        )

    parse_error = None
    try:
        parsed = parse_response(_answer_text(response_text), answer_maps)
    except ValueError as exc:
        parsed = [None] * len(answer_maps)
        parse_error = str(exc)
    score_percent = score_answers(parsed, answer_maps)
    cell_details = score_details(parsed, answer_maps)
    query_results = []
    for index, (answer, expected, cell_detail) in enumerate(
        zip(parsed, answer_maps, cell_details), start=1
    ):
        predicted = answer["board"] if answer is not None else None
        mismatches = []
        if predicted is not None:
            mismatches = [
                [row, col]
                for row, (predicted_row, expected_row) in enumerate(zip(predicted, expected))
                for col, (predicted_value, expected_value) in enumerate(
                    zip(predicted_row, expected_row)
                )
                if predicted_value != expected_value
            ]
        query_results.append({
            "task_id": index,
            "correct": predicted == expected,
            "mismatch_count": len(mismatches) if predicted is not None else None,
            "mismatch_positions": mismatches if predicted is not None else None,
            **cell_detail,
            "reasoning": answer["reasoning"] if answer is not None else None,
        })
    correct_count = sum(item["correct"] for item in query_results)
    score = score_percent / 100.0
    feedback_text = f"Cell accuracy C: {score_percent:.2f}% over non-mine cells."
    if parse_error:
        feedback_text += f" Response format error: {parse_error}"
    trajectory = (StepFeedback(
        step_index=0, prompt=prompt, image_records=(), action=response_text,
        environment_response={
            "examples": task_details["examples"],
            "queries": task_details["queries"],
            "query_results": query_results,
            "score_percent": score_percent,
        },
        success=correct_count == len(answer_maps), score=score, feedback=feedback_text,
        agent_call=agent_call,
    ),)
    return FeedbackResult(
        task_id=task_id, task=dict(task), trajectory=trajectory,
        evaluation=TaskEvaluation(
            execution_status=ExecutionStatus.COMPLETED,
            success=correct_count == len(answer_maps), score=score,
            feedback=feedback_text,
            metrics={
                "query_count": len(answer_maps), "correct_count": correct_count,
                "score_percent": score_percent, "query_results": query_results,
                "examples": task_details["examples"], "queries": task_details["queries"],
            },
        ),
    )
