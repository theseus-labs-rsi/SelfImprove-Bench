"""CSD-native adapter for GameWorld browser-game episodes."""

from __future__ import annotations

import asyncio
import fcntl
from contextlib import nullcontext, contextmanager
import importlib
import json
import math
import os
import socket
import sys
import time
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
    to_jsonable,
)


from .frameworks import SUPPORTED_FRAMEWORKS, client_import_spec


FRAMEWORK_CLIENTS = {
    framework: client_import_spec(framework)
    for framework in SUPPORTED_FRAMEWORKS
}


@contextmanager
def _exclusive_episode_cell(cell: Path):
    # Acquire before selecting/creating an attempt. Locking only run_root
    # permits two starters to race into different paid attempts.
    cell.mkdir(parents=True, exist_ok=True)
    with (cell / '.episode.lock').open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Astray evaluation already running for {cell}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)



# task id -> id() of the first (parent) agent evaluated on it in this process.
_EPISODE_OWNERS: dict[str, int] = {}


def episode_cell_key(task_key: str, agent: Any) -> str:
    """The first agent evaluated on a task in this process is the round's parent:
    its episode cell is stable, so a restarted process resumes or reuses it.
    Candidate validations are new episodes of other agents and always get a
    fresh cell (a refinement redone after a restart yields new candidates)."""
    owner = _EPISODE_OWNERS.setdefault(task_key, id(agent))
    if owner == id(agent):
        return task_key
    return f"{task_key}__candidate_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def step_resume_enabled(task: Mapping[str, Any]) -> bool:
    """Step-level resume is an execution option of the launcher (RSI_STEP_RESUME=1),
    deliberately outside the hashed config/task manifest so it never changes the
    experiment's identity. The legacy per-task flag is still honored."""
    return task.get("resumable_steps") is True or os.environ.get("RSI_STEP_RESUME") == "1"


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
        for path in (code_root,):
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
        progress = summary.get("progress", metrics.get("progress"))
        try:
            score = float(progress)
        except (TypeError, ValueError):
            score = None
        finalized = bool(summary.get("finalized"))
        task_status = str(summary.get("task_status") or "").lower()
        valid = (
            finalized and task_status in {"success", "fail"}
            and not isinstance(progress, bool) and score is not None
            and math.isfinite(score) and 0.0 <= score <= 1.0
        )
        success = task_status == "success"
        feedback = str(summary.get("summary") or "")
        if not valid:
            score = None
            feedback = f"Invalid GameWorld evaluation: status={task_status!r}, finalized={finalized}; {feedback}"
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
                execution_status=ExecutionStatus.COMPLETED if valid else ExecutionStatus.ERROR,
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
        # GameWorld modules are cached in-process. A path change in sys.path
        # alone does NOT switch the training overlay to official holdout.
        existing_main = sys.modules.get("main")
        if existing_main is not None and Path(existing_main.__file__).absolute() != (gameworld_root / "code/main.py").absolute():
            raise RuntimeError("GameWorld root changed in this process; use a fresh process for official holdout")
        from main import main as gameworld_main
        from utils import build_agent_clients

        task_key = str(task["task_id"])
        resumable = step_resume_enabled(task)
        if resumable and task.get("game_id") not in ("03_astray", "18_minecraft-clone-glm"):
            raise RuntimeError("Step resume supports Astray and Minecraft only")
        cell = (self.output_root / framework / episode_cell_key(task_key, agent)
                if resumable else self.output_root / task_key)
        cell.mkdir(parents=True, exist_ok=True)
        with _exclusive_episode_cell(cell) if resumable else nullcontext():
            if resumable:
                attempts = sorted(cell.glob("attempt_*"))
                if len(attempts) > 1:
                    raise RuntimeError("Several resumable attempts in one episode cell; manual audit required")
                if attempts and not (attempts[0] / "astray_recovery/genesis.json").is_file():
                    # Died before its first checkpoint (no step was played): keep it
                    # aside as evidence and start this episode afresh.
                    attempts[0].rename(cell / f"abandoned_before_genesis_{attempts[0].name}")
                    attempts = []
                run_root = attempts[0] if attempts else cell / f"attempt_{uuid.uuid4().hex[:8]}"
                run_root.mkdir(parents=True, exist_ok=not not attempts)
            else:
                run_root = cell / f"attempt_{uuid.uuid4().hex[:8]}"
                run_root.mkdir(parents=True, exist_ok=False)
            client_module, client_class = FRAMEWORK_CLIENTS[framework]
            module = importlib.import_module(client_module)
            client_type = getattr(module, client_class)

            def client_factory(runtime_config: Any, agent_ids: list[str]) -> list[Any]:
                base_clients = build_agent_clients(runtime_config, agent_ids)
                effort = str(task.get("reasoning_effort") or "").strip()
                if effort and effort not in {"low", "medium", "high"}:
                    raise ValueError(f"Unsupported GameWorld reasoning_effort: {effort}")
                return [
                    client_type(
                        base_clients[index].config.with_overrides(reasoning_effort=effort)
                        if effort else base_clients[index].config,
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

            # The same task id is run concurrently by all three RSI providers.
            # Separate each framework's GameWorld ports to avoid cross-run collision.
            framework_offset = {"ace": 0, "hyperagents": 1000, "ahe": 2000}[framework]
            # Task ids repeat across games (e.g. train-gameworld-d1-0001), so key by game too.
            port_key = f"{task['game_id']}/{task_key}"
            port = 20000 + framework_offset + (zlib.crc32(port_key.encode("utf-8")) % 1000)
            if True:  # always probe (was: only for resumable Astray)
                # A just-closed local server may leave a short TCP TIME_WAIT, and
                # two games on one host can hash to the same port. Changing port
                # does not change the map or episode state.
                for index in range(8):
                    candidate = port + 4000 * index
                    try:
                        with socket.socket() as probe:
                            probe.bind(('127.0.0.1', candidate))
                        port = candidate
                        break
                    except OSError:
                        continue
                else:
                    raise RuntimeError('No isolated GameWorld port available for this episode')
            if resumable:
                from .astray_step_resume import AstrayTransaction, UnsafeResume, coordinator_resume
                transaction = AstrayTransaction(run_root=run_root, task=dict(task),
                                                gameworld_root=gameworld_root, framework=framework)
            finished = resumable and transaction.completed_summary(int(task.get('max_steps') or 100))
            if finished:
                # A fully played episode is reused as-is, never played again.
                transaction.adopt_completed_traces(agent)
            else:
                resumes_left = int(os.environ.get("RSI_EPISODE_RESUMES", "3")) if resumable else 0
                while True:
                    if resumable:
                        transaction.prepare_retry()
                    try:
                        with coordinator_resume(transaction) if resumable else nullcontext():
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
                        break
                    except Exception as exc:
                        chain, current = [], exc
                        while current is not None and len(chain) < 20:
                            chain.append(current)
                            current = current.__cause__ or current.__context__
                        if (resumes_left <= 0 or any(isinstance(item, UnsafeResume) for item in chain)
                                or (transaction.store.load() is None and not
                                    (transaction.root / "genesis.json").is_file())):
                            raise
                        resumes_left -= 1
                        print(f"[gameworld] episode interrupted ({type(exc).__name__}: {str(exc)[:200]}); "
                              f"resuming from the last committed step ({resumes_left} resumes left)", flush=True)
                        if framework == "ahe":
                            agent.reset_worker()  # restored from the checkpointed history next
                        time.sleep(float(os.environ.get("RSI_EPISODE_RESUME_DELAY", "30")))
            feedback = self._feedback(task, run_root)
            evaluation = feedback.evaluation
            # Unknown step usage is *not* zero: reporting a partial sum would make
            # AHE/HyperAgents look artificially cheap in Token/absolute-gain.
            step_usages = [
                step.agent_call.usage.total_tokens if step.agent_call and step.agent_call.usage else None
                for step in feedback.trajectory
            ]
            total_tokens = sum(step_usages) if step_usages and all(
                isinstance(value, int) and not isinstance(value, bool) for value in step_usages
            ) else None
            # A manually abandoned, potentially billed request does not appear in
            # interactions.jsonl. The observed successful-step sum is only a
            # lower bound, never the episode's complete Token cost.
            unknown_attempts = sorted(
                (run_root / "astray_recovery" / "abandoned_attempts").glob("*.json")
            ) if resumable else []
            observed_step_tokens = total_tokens
            if unknown_attempts:
                latest = transaction.store.load()
                if latest is None or latest["ledger"].get("billing_complete") is not False:
                    raise RuntimeError("Unreconciled billed attempt lacks an incomplete cost ledger")
                total_tokens = None
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
                "interactions": [to_jsonable(step) for step in feedback.trajectory],
                "llm_metrics": {"tokens": {
                    "total_tokens": total_tokens,
                    **({"known_observed_step_tokens": observed_step_tokens,
                        "unknown_may_have_billed_attempts": len(unknown_attempts)}
                       if unknown_attempts else {}),
                }},
            }
            return EvaluationOutcome(record=record, feedback=feedback)


__all__ = ["FRAMEWORK_CLIENTS", "GameWorldBenchmark"]
