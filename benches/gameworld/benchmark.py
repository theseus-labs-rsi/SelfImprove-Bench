"""CSD-native adapter for GameWorld browser-game episodes."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

from benchmark.contracts import (
    AgentCallRecord,
    EvaluationOutcome,
    ExecutionStatus,
    FeedbackResult,
    ImageRecord,
    StepFeedback,
    TaskEvaluation,
    TokenUsage,
)


from integrations.gameworld.frameworks import SUPPORTED_FRAMEWORKS, client_import_spec


FRAMEWORK_CLIENTS = {
    framework: client_import_spec(framework)
    for framework in SUPPORTED_FRAMEWORKS
}


class GameWorldBenchmark:
    """Run one external GameWorld episode and emit CSD feedback contracts."""

    def __init__(self, *, output_root: str | Path | None = None) -> None:
        self.output_root = Path(
            output_root or os.environ.get("GAMEWORLD_RESULTS_ROOT", "benchmark_results/gameworld")
        ).expanduser().resolve()

    @staticmethod
    def _prepare_imports(gameworld_root: Path) -> Path:
        code_root = gameworld_root / "code"
        if not code_root.is_dir():
            raise FileNotFoundError(f"GameWorld code directory not found: {code_root}")
        integration_root = Path(__file__).resolve().parents[2] / "integrations" / "gameworld"
        for path in (integration_root, code_root):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
        return code_root

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _usage(value: Any) -> TokenUsage | None:
        if not isinstance(value, Mapping):
            return None
        fields = {
            name: value.get(name)
            for name in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        }
        if not any(isinstance(item, int) for item in fields.values()):
            return None
        return TokenUsage(**fields)

    @classmethod
    def _feedback(cls, task: Mapping[str, Any], run_root: Path) -> FeedbackResult:
        agent_root = run_root / "agent_0"
        summary = cls._read_json(agent_root / "evaluation" / "summary.json")
        trajectory: list[StepFeedback] = []
        interaction_path = agent_root / "interactions.jsonl"
        if interaction_path.is_file():
            for line in interaction_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, Mapping):
                    continue
                input_data = record.get("input") if isinstance(record.get("input"), Mapping) else {}
                output_data = record.get("output") if isinstance(record.get("output"), Mapping) else {}
                screenshot = input_data.get("screenshot")
                if screenshot:
                    image_path = Path(str(screenshot))
                    if not image_path.is_absolute():
                        image_path = agent_root / image_path
                    image_records = (ImageRecord(path=str(image_path.resolve())),)
                else:
                    image_records = ()
                action = output_data.get("executed_action")
                evaluation = record.get("task_evaluation")
                if not isinstance(evaluation, Mapping):
                    evaluation = {}
                response_text = output_data.get("raw_response") or output_data.get("response_text") or ""
                task_status = str(evaluation.get("task_status") or "").lower()
                progress = evaluation.get("progress")
                try:
                    step_score = float(progress) if progress is not None else None
                except (TypeError, ValueError):
                    step_score = None
                trajectory.append(
                    StepFeedback(
                        step_index=int(record.get("interaction_id") or len(trajectory)),
                        prompt=str(input_data.get("raw_message_sent") or input_data.get("prompt") or ""),
                        image_records=image_records,
                        action=json.dumps(action, ensure_ascii=False) if isinstance(action, Mapping) else str(action or ""),
                        environment_response=record.get("game_state") or {},
                        success=task_status == "success" if task_status else None,
                        score=step_score,
                        feedback=str(evaluation.get("summary") or "") or None,
                        agent_call=AgentCallRecord(
                            response_text=str(response_text),
                            reasoning=output_data.get("diagnostic_reasoning"),
                            usage=cls._usage(output_data.get("usage")),
                            provider="gameworld",
                            model=record.get("model_name"),
                            status=output_data.get("error") and "error" or "ok",
                            metadata={
                                "action_validity": output_data.get("action_validity"),
                                "parsed_action": output_data.get("parsed_action"),
                            },
                        ),
                    )
                )
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
        progress = summary.get("progress", metrics.get("progress", 0.0))
        try:
            score = float(progress or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        finalized = bool(summary.get("finalized"))
        task_status = str(summary.get("task_status") or "").lower()
        success = task_status == "success"
        feedback = str(summary.get("summary") or "")
        if not summary:
            feedback = "GameWorld did not produce evaluation/summary.json"
        evaluation_metrics = {
            **dict(metrics),
            "task_status": task_status or None,
            "stop_reason": summary.get("stop_reason"),
            "max_steps": summary.get("max_steps"),
            "game_completion_progress": summary.get("game_completion_progress"),
        }
        return FeedbackResult(
            task_id=str(task["task_id"]),
            task=dict(task),
            trajectory=tuple(trajectory),
            evaluation=TaskEvaluation(
                execution_status=ExecutionStatus.COMPLETED if finalized else ExecutionStatus.ERROR,
                success=success,
                score=score,
                feedback=feedback,
                metrics=evaluation_metrics,
            ),
        )

    def evaluate(self, task: Mapping[str, Any], agent: Any) -> EvaluationOutcome:
        framework = str(task.get("framework") or "").strip().lower()
        if framework not in FRAMEWORK_CLIENTS:
            raise ValueError(f"GameWorld task requires framework in {tuple(FRAMEWORK_CLIENTS)}")
        gameworld_root = Path(
            str(task.get("gameworld_root") or os.environ.get("GAMEWORLD_ROOT") or "")
        ).expanduser().resolve()
        if not gameworld_root.is_dir():
            raise FileNotFoundError(
                "Set GAMEWORLD_ROOT to the open-source GameWorld checkout before running gameworld tasks"
            )
        self._prepare_imports(gameworld_root)
        from main import main as gameworld_main
        from utils import build_agent_clients

        task_key = str(task["task_id"])
        run_root = self.output_root / task_key / f"attempt_{uuid.uuid4().hex[:8]}"
        run_root.mkdir(parents=True, exist_ok=False)
        client_module, client_class = FRAMEWORK_CLIENTS[framework]
        module = importlib.import_module(client_module)
        client_type = getattr(module, client_class)

        def client_factory(runtime_config: Any, agent_ids: list[str]) -> list[Any]:
            base_clients = build_agent_clients(runtime_config, agent_ids)
            return [
                client_type(
                    base_clients[index].config,
                    agent,
                    semantic_controls_specs=runtime_config.semantic_controls_specs[index],
                )
                for index in range(len(base_clients))
            ]

        model_spec = str(
            task.get("model_spec")
            or os.environ.get("GAMEWORLD_MODEL_ID")
            or ""
        ).strip()
        if not model_spec:
            raise RuntimeError(
                "GameWorld model is not configured; set game_options.gameworld.model_spec "
                "or GAMEWORLD_MODEL_ID"
            )
        import zlib

        port = 9300 + (zlib.crc32(task_key.encode("utf-8")) % 500)
        asyncio.run(
            gameworld_main(
                config_preset=f"{task['game_id']}+{task['game_task_id']}+{model_spec}",
                headless=True,
                port=port,
                log_root=str(run_root),
                rerun_policy="never",
                client_factory=client_factory,
                max_steps_override=int(task.get("max_steps") or 100),
            )
        )
        feedback = self._feedback(task, run_root)
        evaluation = feedback.evaluation
        total_tokens = sum(
            int(step.agent_call.usage.total_tokens or 0)
            for step in feedback.trajectory
            if step.agent_call
            and step.agent_call.usage
            and step.agent_call.usage.total_tokens is not None
        )
        status = "ok" if evaluation.execution_status == ExecutionStatus.COMPLETED else "error"
        record = {
            "status": status,
            "game": "gameworld",
            "game_id": task.get("game_id"),
            "game_task_id": task.get("game_task_id"),
            "framework": framework,
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "relative_score": evaluation.score,
            "raw_score": evaluation.metrics.get("score", evaluation.score),
            "success": evaluation.success,
            "error": evaluation.feedback if status != "ok" else None,
            "agent_output": None,
            "simulation": {"metrics": evaluation.metrics, "run_root": str(run_root)},
            "interactions": [step for step in feedback.trajectory],
            "llm_metrics": {"tokens": {"total_tokens": total_tokens}},
        }
        return EvaluationOutcome(record=record, feedback=feedback)


__all__ = ["FRAMEWORK_CLIENTS", "GameWorldBenchmark"]
