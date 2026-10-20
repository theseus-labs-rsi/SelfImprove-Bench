"""Offline lifecycle example, NOT a learning algorithm or a competitive agent.

Replace FixedAgent.act and MinimalAdapter.refine with your own method. No game,
runner, provider registry, or benchmark scoring code needs to be modified.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.contracts import BackendResponse, FeedbackResult, TokenUsage
from benchmark.interfaces import MethodContext


class FixedAgent:
    def __init__(self, action: Mapping[str, Any]) -> None:
        self.action = deepcopy(dict(action))

    def act(self, prompt: str, *, image_paths: Sequence[Path] = ()) -> BackendResponse:
        # The example makes no model call, so its reported model usage is zero.
        return BackendResponse(text=json.dumps(self.action), usage=TokenUsage(
            input_tokens=0, output_tokens=0, total_tokens=0, reasoning_tokens=0))


class MinimalAdapter:
    def __init__(self, context: MethodContext) -> None:
        self.context = context
        self.action = deepcopy(context.options.get("action", {"days": []}))
        self.rounds = 0

    def init(self) -> None:
        self.rounds = 0

    def load(self) -> FixedAgent:
        # Return a fresh execution session; never train here (holdout/replay use it).
        return FixedAgent(self.action)

    def refine(self, feedback: list[FeedbackResult], *, output_dir: Path) -> None:
        self.rounds += len(feedback)
        self.last_refinement_report = {
            "status": "completed",
            "token_usage_report": {
                "total_tokens": 0, "prompt_tokens": 0, "answer_tokens": 0,
                "reasoning_tokens": 0, "complete": True, "no_model_calls": True,
            },
        }
        # Real methods commit their improved state here, or return CandidateBatch.

    def export_state(self) -> dict[str, Any]:
        return {"rounds": self.rounds, "action": deepcopy(self.action)}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.rounds = int(state["rounds"])
        self.action = deepcopy(state["action"])

    def close(self) -> None:
        pass
