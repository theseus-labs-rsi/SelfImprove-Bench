"""Overall Control Loop adaptor for the Hero's Journey benchmark."""

from __future__ import annotations

import copy
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from benchmark.contracts import (
    AgentCallRecord,
    BackendResponse,
    ExecutionStatus,
    FeedbackResult,
    FeedbackResults,
    StepFeedback,
    TaskEvaluation,
    to_jsonable,
)
from benchmark.token_ledger import aggregate_feedback_usage

from .task_factory import HeroJourneyDifficulty, load_difficulty_manifest


LOCAL_HEROS_JOURNEY = Path(__file__).resolve().parent / "HerosJourney"
if str(LOCAL_HEROS_JOURNEY) not in sys.path:
    sys.path.insert(0, str(LOCAL_HEROS_JOURNEY))

from herosjourney import get_task
from herosjourney.core.demo_generator import generate_mixed_demos
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.runner.adventure_episode import (
    Action,
    construct_demo_context,
    run_single_episode,
)

from .sequence_episode import SequenceEpisodeOutcome, run_sequence_episode


USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)
SCORE_METRIC = "reference_steps_over_executed_steps_v1"


@dataclass(frozen=True, slots=True)
class HeroJourneyTask:
    task_id: str
    difficulty: int
    seed: int
    episode_idx: int
    settings: HeroJourneyDifficulty
    action_limit: int
    logic_fingerprint: str | None
    source_tasks: tuple[Any, ...] = field(repr=False)
    generated_task: Any = field(repr=False)
    demo_context: str = field(repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "episode_idx": self.episode_idx,
            "task_type": self.settings.task_type,
            "logic_version": self.settings.logic_version,
            "action_budget_multiplier": self.settings.action_budget_multiplier,
            "action_limit": self.action_limit,
            "max_runs": self.settings.max_runs,
            "episode_mode": self.settings.episode_mode,
            "interaction_protocol": self.settings.interaction_protocol,
            "logic_fingerprint": self.logic_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class _ModelCall:
    prompt: str
    response: BackendResponse
    action_is_valid: bool


class HeroJourneyBenchmark:
    """Construct and execute independent Hero's Journey episodes."""

    def __init__(
        self,
        difficulties: Mapping[int, HeroJourneyDifficulty | Mapping[str, Any]] | None = None,
        seeds: Sequence[int] = (),
        *,
        max_workers: int = 1,
        verbose: bool = False,
    ) -> None:
        difficulty_source = load_difficulty_manifest() if difficulties is None else difficulties
        if not difficulty_source or not seeds:
            raise ValueError("difficulties and seeds must not be empty")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._difficulties = {
            int(level): self._normalize_difficulty(value)
            for level, value in difficulty_source.items()
        }
        self._seeds = tuple(int(seed) for seed in seeds)
        self._max_workers = max_workers
        self._verbose = verbose
        self._episode_artifacts: dict[str, dict[str, Any]] = {}

    def construct(
        self,
        difficulty: int,
        seed: int | None = None,
    ) -> list[HeroJourneyTask]:
        try:
            settings = self._difficulties[difficulty]
        except KeyError as exc:
            available = ", ".join(str(level) for level in sorted(self._difficulties))
            raise ValueError(
                f"Unknown Hero's Journey difficulty {difficulty}; available: {available}"
            ) from exc

        selected_seeds = self._seeds if seed is None else (int(seed),)
        tasks: list[HeroJourneyTask] = []
        for task_seed in selected_seeds:
            tasks.extend(self._construct_seed(difficulty, task_seed, settings))
        return tasks

    def construct_one(
        self, difficulty: int, *, seed: int, task_id: str, episode_idx: int
    ) -> HeroJourneyTask:
        tasks = self.construct(difficulty, seed)
        try:
            task = next(item for item in tasks if item.episode_idx == episode_idx)
        except StopIteration as exc:
            raise ValueError(f"episode_idx={episode_idx} is not configured") from exc
        return replace(task, task_id=task_id)

    def play(
        self,
        tasks: Sequence[HeroJourneyTask],
        agent: Any,
        *,
        reload_agent: Callable[[], Any] | None = None,
    ) -> FeedbackResults:
        task_list = list(tasks)
        self._validate_tasks(task_list)
        if not task_list:
            self._episode_artifacts = {}
            return []

        if reload_agent is not None and self._max_workers != 1:
            raise ValueError("Agent reload requires serial Hero episodes")
        if self._max_workers == 1:
            outcomes = []
            active_agent = agent
            for index, task in enumerate(task_list):
                outcome = self._play_safely(
                    task, active_agent, timeout_as_zero=reload_agent is not None
                )
                outcomes.append(outcome)
                if (reload_agent is not None and index + 1 < len(task_list)
                        and outcome[0].evaluation.execution_status == ExecutionStatus.TIMEOUT):
                    active_agent = reload_agent()
        else:
            worker_count = min(self._max_workers, len(task_list))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                outcomes = list(
                    executor.map(lambda task: self._play_safely(task, agent), task_list)
                )
        self._episode_artifacts = {
            feedback.task_id: artifact
            for feedback, artifact in outcomes
            if artifact is not None
        }
        return [feedback for feedback, _ in outcomes]

    def aggregate(self, feedback_results: FeedbackResults) -> dict[str, Any]:
        status_counts = {status.value: 0 for status in ExecutionStatus}
        completion_rates = []
        protocol_counts: dict[str, int] = {}
        for result in feedback_results:
            status_counts[result.evaluation.execution_status.value] += 1
            metrics = result.evaluation.metrics
            protocol = metrics.get("interaction_protocol", "legacy_unlabeled")
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
            completion_rate = metrics.get("completion_rate")
            if isinstance(completion_rate, (int, float)) and not isinstance(completion_rate, bool):
                completion_rates.append(float(completion_rate))
        usage_report = aggregate_feedback_usage(feedback_results)

        task_count = len(feedback_results)
        success_count = sum(result.evaluation.success for result in feedback_results)
        score_total = sum(result.evaluation.score for result in feedback_results)
        return {
            "task_count": task_count,
            "success_count": success_count,
            "success_rate": success_count / task_count if task_count else 0.0,
            "score_total": score_total,
            "score_mean": score_total / task_count if task_count else 0.0,
            "completion_rate_mean": (
                sum(completion_rates) / len(completion_rates) if completion_rates else None
            ),
            "completion_rate_count": len(completion_rates),
            "interaction_protocol_counts": protocol_counts,
            "execution_status_counts": status_counts,
            **usage_report,
        }

    def persist(
        self,
        feedback_results: FeedbackResults,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        artifact_dir = output_path / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for task_id, episode_data in self._episode_artifacts.items():
            self._write_json(artifact_dir / f"{self._safe_name(task_id)}.json", episode_data)

        feedback_path = output_path / "feedback_results.json"
        summary_path = output_path / "summary.json"
        self._write_json(feedback_path, feedback_results)
        self._write_json(summary_path, self.aggregate(feedback_results))
        return {
            "feedback_results": feedback_path,
            "summary": summary_path,
            "artifacts": artifact_dir,
        }

    def _construct_seed(
        self,
        difficulty: int,
        seed: int,
        settings: HeroJourneyDifficulty,
    ) -> list[HeroJourneyTask]:
        spec = get_task(settings.task_type)
        semantic_lexicon, nonce_lexicon = load_lexicons()
        with open(spec.rules, encoding="utf-8") as rule_file:
            rule = json.load(rule_file)
        elements = fill_elements(
            rule,
            semantic_lexicon,
            nonce_lexicon,
            seed=seed,
            split_spec=spec.split,
        )
        logic_info = elements.get("_variant_info", {}).get("seeded_logic", {})
        if settings.logic_version and logic_info.get("family") != settings.logic_version:
            raise ValueError(
                f"Hero logic version mismatch: expected {settings.logic_version!r}, "
                f"got {logic_info.get('family')!r}"
            )
        logic_fingerprint = logic_info.get("fingerprint")
        source_tasks = tuple(spec.gen_fn(elements, split="source", use_nonce=False))
        generated_tasks = tuple(spec.gen_fn(elements, split="gen", use_nonce=False))
        demos = generate_mixed_demos(list(source_tasks), distractor_tasks=[])
        demo_context = construct_demo_context(demos)

        tasks = []
        for episode_idx in settings.episode_indices:
            if not 0 <= episode_idx < len(generated_tasks):
                raise IndexError(
                    f"episode_idx={episode_idx} is outside the generated task set "
                    f"of size {len(generated_tasks)} for {settings.task_type!r}"
                )
            task_id = (
                f"hero-{self._safe_name(settings.task_type)}-d{difficulty}-"
                f"s{seed}-e{episode_idx}"
            )
            reference_length = len(generated_tasks[episode_idx].tree.get_solution())
            action_limit = (
                settings.max_runs if settings.max_runs is not None
                else reference_length * settings.action_budget_multiplier
            )
            tasks.append(
                HeroJourneyTask(
                    task_id=task_id,
                    difficulty=difficulty,
                    seed=seed,
                    episode_idx=episode_idx,
                    settings=settings,
                    action_limit=action_limit,
                    logic_fingerprint=logic_fingerprint,
                    source_tasks=source_tasks,
                    generated_task=generated_tasks[episode_idx],
                    demo_context=demo_context,
                )
            )
        return tasks

    def _play_safely(
        self,
        task: HeroJourneyTask,
        agent: Any,
        *,
        timeout_as_zero: bool = False,
    ) -> tuple[FeedbackResult, dict[str, Any] | None]:
        calls: list[_ModelCall] = []
        try:
            sequence_outcome: SequenceEpisodeOutcome | None = None
            if task.settings.interaction_protocol == "action_sequence_v1":
                sequence_outcome = run_sequence_episode(
                    episode_idx=task.episode_idx,
                    task=copy.deepcopy(task.generated_task),
                    source_tasks=copy.deepcopy(list(task.source_tasks)),
                    demo_context=task.demo_context,
                    action_budget_multiplier=task.settings.action_budget_multiplier,
                    action_limit=task.action_limit,
                    initial_currency=task.settings.initial_currency,
                    no_reasoning=task.settings.no_reasoning,
                    agent=agent,
                )
                episode = sequence_outcome.episode
                calls.append(_ModelCall(
                    sequence_outcome.prompt, sequence_outcome.response,
                    sequence_outcome.parse_error is None,
                ))
            else:
                episode = run_single_episode(
                    episode_idx=task.episode_idx,
                    task=copy.deepcopy(task.generated_task),
                    demo_context=task.demo_context,
                    max_runs=task.action_limit,
                    verbose=self._verbose,
                    truncate_window=task.settings.truncate_window,
                    initial_currency=task.settings.initial_currency,
                    num_tries=task.settings.action_budget_multiplier,
                    episode_mode=task.settings.episode_mode,
                    no_reasoning=task.settings.no_reasoning,
                    model_fn=self._make_model_fn(agent, calls),
                    converter_fn=self._identity_converter,
                    source_tasks=copy.deepcopy(list(task.source_tasks)),
                )
            feedback = self._build_feedback(task, episode, calls, sequence_outcome)
            artifact_data = self._safe_episode_data(episode.to_dict())
            artifact_data["interaction_protocol"] = task.settings.interaction_protocol
            artifact_data["action_limit"] = task.action_limit
            return feedback, artifact_data
        except TimeoutError as exc:
            status = ExecutionStatus.TIMEOUT if timeout_as_zero else ExecutionStatus.ERROR
            return self._error_feedback(task, calls, exc, status=status), None
        except Exception as exc:
            return self._error_feedback(task, calls, exc), None

    @staticmethod
    def _make_model_fn(agent: Any, calls: list[_ModelCall]):
        def model_fn(prompt: str, max_tokens: int = 512):
            del max_tokens
            response = agent.act(prompt)
            try:
                Action.model_validate_json(response.text)
                action_is_valid = True
            except Exception:
                action_is_valid = False
            calls.append(_ModelCall(prompt, response, action_is_valid))
            usage = response.usage
            token_counts = None if usage is None else {
                "prompt_tokens": usage.input_tokens or 0,
                "candidates_tokens": usage.output_tokens or 0,
                "total_tokens": usage.total_tokens or 0,
            }
            return response.text, response.reasoning, token_counts

        return model_fn

    def _build_feedback(self, task, episode, calls, sequence_outcome=None) -> FeedbackResult:
        score = float(episode.efficiency) if episode.success and episode.efficiency is not None else 0.0
        artifact = f"artifacts/{self._safe_name(task.task_id)}.json"
        metrics = {
            "task_type": episode.task_type,
            "seed": task.seed,
            "episode_idx": episode.episode_idx,
            "terminated": episode.terminated,
            "efficiency": episode.efficiency,
            "completion_rate": episode.completion_rate,
            "num_runs": episode.num_runs,
            "reference_length": episode.reference_length,
            "action_limit": task.action_limit,
            "interaction_protocol": task.settings.interaction_protocol,
            "score_metric": SCORE_METRIC,
            "artifact": artifact,
        }
        if sequence_outcome is not None:
            metrics["planned_action_count"] = sequence_outcome.planned_action_count
            metrics["parse_error"] = sequence_outcome.parse_error
        feedback = (sequence_outcome.parse_error if sequence_outcome else None) or episode.teaching_message or (
            "Hero's Journey episode succeeded."
            if episode.success
            else "Hero's Journey episode did not reach its goal."
        )
        return FeedbackResult(
            task_id=task.task_id,
            task=task.snapshot(),
            trajectory=(
                self._build_sequence_trajectory(calls[0], episode.action_obs_reasoning_history, sequence_outcome.parse_error)
                if sequence_outcome is not None else
                self._build_trajectory(calls, episode.action_obs_reasoning_history)
            ),
            evaluation=TaskEvaluation(
                execution_status=ExecutionStatus.COMPLETED,
                success=bool(episode.success),
                score=score,
                feedback=feedback,
                metrics=metrics,
            ),
        )

    def _error_feedback(
        self, task, calls, exc, *, status: ExecutionStatus = ExecutionStatus.ERROR
    ) -> FeedbackResult:
        message = f"{type(exc).__name__}: {exc}"
        return FeedbackResult(
            task_id=task.task_id,
            task=task.snapshot(),
            trajectory=self._build_trajectory(calls, []),
            evaluation=TaskEvaluation(
                execution_status=status,
                success=False,
                score=0.0,
                feedback=message,
                metrics={
                    "task_type": task.settings.task_type, "seed": task.seed,
                    "interaction_protocol": task.settings.interaction_protocol,
                    "score_metric": SCORE_METRIC,
                    "action_limit": task.action_limit,
                },
            ),
        )

    @classmethod
    def _build_trajectory(cls, calls, action_history) -> tuple[StepFeedback, ...]:
        accepted_actions = iter(action_history)
        steps = []
        for step_index, call in enumerate(calls):
            action_event = next(accepted_actions, None) if call.action_is_valid else None
            observation = action_event.get("observation") if action_event else None
            steps.append(
                StepFeedback(
                    step_index=step_index,
                    prompt=call.prompt,
                    image_records=(),
                    action=(action_event or {}).get("action", call.response.text),
                    environment_response=(
                        {"observation": observation}
                        if action_event else {"event": "invalid_model_response"}
                    ),
                    success=(action_event or {}).get("success", False),
                    score=None,
                    feedback=observation or "The response was not accepted as an action.",
                    agent_call=cls._agent_call(call.response),
                )
            )
        return tuple(steps)

    @classmethod
    def _build_sequence_trajectory(cls, call, action_history, parse_error) -> tuple[StepFeedback, ...]:
        """One agent call can yield many real environment steps."""
        if not action_history:
            return (StepFeedback(
                step_index=0, prompt=call.prompt, image_records=(),
                action=call.response.text,
                environment_response={"event": "invalid_action_sequence", "error": parse_error},
                success=False, score=None,
                feedback=parse_error or "No action was executed.",
                agent_call=cls._agent_call(call.response),
            ),)

        steps = []
        for index, event in enumerate(action_history):
            steps.append(StepFeedback(
                step_index=index,
                prompt=call.prompt if index == 0 else "",
                image_records=(), action=event["action"],
                environment_response={"observation": event["observation"]},
                success=event["success"], score=None,
                feedback=event["observation"],
                agent_call=cls._agent_call(call.response) if index == 0 else None,
            ))
        return tuple(steps)

    @staticmethod
    def _agent_call(response: BackendResponse) -> AgentCallRecord:
        return AgentCallRecord(
            response_text=response.text,
            reasoning=response.reasoning,
            usage=response.usage,
            provider=response.provider,
            model=response.model,
            status=response.status,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
        )

    @staticmethod
    def _identity_converter(response: str, schema: dict[str, Any]) -> str:
        del schema
        return response

    @staticmethod
    def _safe_episode_data(value: dict[str, Any]) -> dict[str, Any]:
        data = dict(value)
        data.pop("reference_solution", None)
        return data

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "task"

    @staticmethod
    def _normalize_difficulty(
        value: HeroJourneyDifficulty | Mapping[str, Any],
    ) -> HeroJourneyDifficulty:
        if isinstance(value, HeroJourneyDifficulty):
            return value
        return HeroJourneyDifficulty.from_mapping(value)

    @staticmethod
    def _validate_tasks(tasks: Sequence[HeroJourneyTask]) -> None:
        if any(not isinstance(task, HeroJourneyTask) for task in tasks):
            raise TypeError("Every Hero's Journey task must be a HeroJourneyTask")
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Hero's Journey task_id values must be unique")

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(to_jsonable(value), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = [
    "HeroJourneyBenchmark",
    "HeroJourneyDifficulty",
    "HeroJourneyTask",
]
