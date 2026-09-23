from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

LOCAL_HEROS_JOURNEY = Path(__file__).resolve().parent / "HerosJourney"
if str(LOCAL_HEROS_JOURNEY) not in sys.path:
    sys.path.insert(0, str(LOCAL_HEROS_JOURNEY))

from herosjourney import get_task
from herosjourney.core.demo_generator import generate_mixed_demos
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.eval import compute_ecsr
from herosjourney.runner.adventure_episode import (
    construct_demo_context,
    run_single_episode,
)


@dataclass(frozen=True)
class HeroJourneyPlayConfig:
    """Frozen internal-evaluation settings shared by all candidate agents."""

    case_id: str = "hero_journey"
    split: str = "train"
    task_type: str = "conditional"
    seed: int = 20
    episode_idx: int = 0
    min_model_tokens: int | None = None
    verbose: bool = False


def _build_task_set(config: HeroJourneyPlayConfig):
    """Create the complete source/gen task set for one deterministic seed."""
    spec = get_task(config.task_type)
    semantic_lexicon, nonce_lexicon = load_lexicons()
    with open(spec.rules, encoding="utf-8") as rule_file:
        rule = json.load(rule_file)

    elements = fill_elements(
        rule,
        semantic_lexicon,
        nonce_lexicon,
        seed=config.seed,
        split_spec=spec.split,
    )
    source_tasks = spec.gen_fn(elements, split="source", use_nonce=False)
    generation_tasks = spec.gen_fn(elements, split="gen", use_nonce=False)
    demos = generate_mixed_demos(source_tasks, distractor_tasks=[])
    return spec, source_tasks, generation_tasks, construct_demo_context(demos)


def _build_task_variant(config: HeroJourneyPlayConfig):
    """Select one indexed generated task while preserving the seed context."""
    spec, source_tasks, generation_tasks, demo_context = _build_task_set(config)
    if not 0 <= config.episode_idx < len(generation_tasks):
        raise IndexError(
            f"episode_idx={config.episode_idx} is outside the generated task set "
            f"of size {len(generation_tasks)} for {config.task_type!r}."
        )
    return spec, source_tasks, generation_tasks[config.episode_idx], demo_context


def _prediction_to_text(prediction: Any) -> str:
    if isinstance(prediction, str):
        return prediction
    return json.dumps(prediction, ensure_ascii=False)


def _make_model_fn(
    agent: Any, min_model_tokens: int | None
) -> Callable[..., tuple[Any, ...]]:
    """Adapt an RSI runtime agent to Hero's Journey's model callback."""

    def model_fn(prompt: str, max_tokens: int = 512) -> tuple[Any, ...]:
        output_tokens = (
            max_tokens
            if min_model_tokens is None
            else max(max_tokens, min_model_tokens)
        )
        response = agent.run(
            {
                "domain": "heros_journey",
                "instruction": prompt,
                "max_tokens": output_tokens,
            }
        )
        if not isinstance(response, dict) or "prediction" not in response:
            raise ValueError("agent.run() must return a mapping containing prediction.")
        usage = response.get("usage")
        token_counts = None
        if isinstance(usage, Mapping) and usage.get("available"):
            token_counts = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "candidates_tokens": int(usage.get("completion_tokens", 0)),
            }
        return _prediction_to_text(response["prediction"]), None, token_counts

    return model_fn


def _save_episode_artifact(agent: Any, episode_data: dict[str, Any], *, case_id: str) -> str:
    artifact_dir = os.path.abspath(agent.artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "case"
    filename = f"hero_journey_{safe_case_id}.json"
    artifact_path = os.path.join(artifact_dir, filename)
    with open(artifact_path, "w", encoding="utf-8") as artifact_file:
        json.dump(episode_data, artifact_file, ensure_ascii=False, indent=2)
    return f"artifacts/{filename}"


def create_hero_journey_play(
    config: HeroJourneyPlayConfig | None = None,
) -> Callable[[Any], dict[str, Any]]:
    """Build a reusable play(agent) callback over one frozen episode."""
    config = config or HeroJourneyPlayConfig()
    spec, source_tasks, task, demo_context = _build_task_variant(config)

    def play(agent: Any) -> dict[str, Any]:
        episode = run_single_episode(
            episode_idx=config.episode_idx,
            task=task,
            demo_context=demo_context,
            max_runs=None,
            verbose=config.verbose,
            truncate_window=None,
            converter_model="small",
            episode_mode="standard",
            model_fn=_make_model_fn(agent, config.min_model_tokens),
            source_tasks=source_tasks,
            num_tries=2,
        )
        episode_data = episode.to_dict()
        artifact_path = _save_episode_artifact(agent, episode_data, case_id=config.case_id)
        ecsr = float(compute_ecsr([episode], spec.max_tries))

        return {
            "case_id": config.case_id,
            "split": config.split,
            "valid": True,
            "selection_score": ecsr,
            "metrics": {
                "task_type": episode.task_type,
                "seed": config.seed,
                "success": episode.success,
                "efficiency": episode.efficiency,
                "ecsr": ecsr,
                "completion_rate": episode.completion_rate,
                "num_runs": episode.num_runs,
                "reference_length": episode.reference_length,
            },
            "feedback": {
                "terminated": episode.terminated,
                "completion_map": episode.completion_map,
                "teaching_message": episode.teaching_message,
            },
            "artifacts": {"episode": artifact_path},
            "cost": {
                "prompt_tokens": episode.prompt_tokens,
                "completion_tokens": episode.completion_tokens,
            },
            "err": None,
        }

    return play


def create_hero_journey_seed_play(
    config: HeroJourneyPlayConfig,
) -> Callable[[Any], dict[str, Any]]:
    """Evaluate every generated task for one seed, matching quickstart.py."""
    spec, source_tasks, generation_tasks, demo_context = _build_task_set(config)

    def play(agent: Any) -> dict[str, Any]:
        episodes = []
        case_results = []
        for episode_idx, task in enumerate(generation_tasks):
            episode = run_single_episode(
                episode_idx=episode_idx,
                task=task,
                demo_context=demo_context,
                max_runs=None,
                verbose=config.verbose,
                truncate_window=None,
                converter_model="small",
                episode_mode="standard",
                model_fn=_make_model_fn(agent, config.min_model_tokens),
                source_tasks=source_tasks,
                num_tries=2,
            )
            episodes.append(episode)
            case_id = f"{config.case_id}_episode_{episode_idx}"
            artifact = _save_episode_artifact(
                agent, episode.to_dict(), case_id=case_id
            )
            episode_ecsr = float(compute_ecsr([episode], spec.max_tries))
            case_results.append(
                _episode_result(config, episode, case_id, artifact, episode_ecsr)
            )

        seed_ecsr = float(compute_ecsr(episodes, spec.max_tries))
        return {
            "case_id": config.case_id,
            "split": config.split,
            "valid": all(item["valid"] for item in case_results),
            "selection_score": seed_ecsr,
            "metrics": {
                "task_type": config.task_type,
                "seed": config.seed,
                "episode_count": len(episodes),
                "success_rate": statistics.fmean(
                    float(episode.success) for episode in episodes
                ),
                "ecsr": seed_ecsr,
                "max_tries": spec.max_tries,
            },
            "cases": case_results,
            "feedback": {item["case_id"]: item["feedback"] for item in case_results},
            "artifacts": {
                item["case_id"]: item["artifacts"]["episode"]
                for item in case_results
            },
            "cost": {
                "prompt_tokens": sum(item["cost"]["prompt_tokens"] for item in case_results),
                "completion_tokens": sum(
                    item["cost"]["completion_tokens"] for item in case_results
                ),
            },
            "err": None,
        }

    return play


def _episode_result(config, episode, case_id, artifact_path, ecsr):
    """Convert one EpisodeResult into the shared leaf Result schema."""
    return {
        "case_id": case_id,
        "split": config.split,
        "valid": True,
        "selection_score": ecsr,
        "metrics": {
            "task_type": episode.task_type,
            "seed": config.seed,
            "episode_idx": episode.episode_idx,
            "success": episode.success,
            "efficiency": episode.efficiency,
            "ecsr": ecsr,
            "completion_rate": episode.completion_rate,
            "num_runs": episode.num_runs,
            "reference_length": episode.reference_length,
        },
        "feedback": {
            "terminated": episode.terminated,
            "completion_map": episode.completion_map,
            "teaching_message": episode.teaching_message,
        },
        "artifacts": {"episode": artifact_path},
        "cost": {
            "prompt_tokens": episode.prompt_tokens,
            "completion_tokens": episode.completion_tokens,
        },
        "err": None,
    }


def create_hero_journey_seed_suite(
    configs: Sequence[HeroJourneyPlayConfig],
    *,
    max_workers: int = 1,
    model_repeats: int = 1,
) -> Callable[[Any], dict[str, Any]]:
    """Aggregate quickstart-compatible seed evaluations into one RSI batch."""
    materialized = tuple(configs)
    if not materialized:
        raise ValueError("HeroJourney seed suite requires at least one seed.")
    if max_workers < 1:
        raise ValueError("HeroJourney seed suite max_workers must be at least 1.")
    if model_repeats < 1:
        raise ValueError("HeroJourney model_repeats must be at least 1.")
    case_ids = [config.case_id for config in materialized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("HeroJourney seed suite case_id values must be unique.")
    seed_plays = tuple(
        create_hero_journey_seed_play(
            replace(config, case_id=f"{config.case_id}_repeat_{repeat_idx}")
        )
        for config in materialized
        for repeat_idx in range(model_repeats)
    )

    def play(agent: Any) -> dict[str, Any]:
        seed_results = _run_plays(seed_plays, agent=agent, max_workers=max_workers)
        cases = [case for seed in seed_results for case in seed["cases"]]
        return {
            "valid": all(seed["valid"] for seed in seed_results),
            "selection_score": statistics.fmean(
                seed["selection_score"] for seed in seed_results
            ),
            "metrics": {
                "seed_count": len(materialized),
                "model_repeats": model_repeats,
                "episode_count": len(cases),
                "mean_seed_ecsr": statistics.fmean(
                    seed["selection_score"] for seed in seed_results
                ),
                "seed_results": [
                    {
                        "case_id": seed["case_id"],
                        "seed": seed["metrics"]["seed"],
                        "ecsr": seed["selection_score"],
                    }
                    for seed in seed_results
                ],
            },
            "cases": cases,
            "feedback": {
                case["case_id"]: case["feedback"] for case in cases
            },
            "artifacts": {
                case["case_id"]: case["artifacts"]["episode"] for case in cases
            },
            "cost": {
                "prompt_tokens": sum(case["cost"]["prompt_tokens"] for case in cases),
                "completion_tokens": sum(
                    case["cost"]["completion_tokens"] for case in cases
                ),
            },
            "err": None,
        }

    return play


def create_hero_journey_suite(
    configs: Sequence[HeroJourneyPlayConfig],
    *,
    max_workers: int = 1,
) -> Callable[[Any], dict[str, Any]]:
    """Build a play(agent) callback that evaluates several frozen cases."""

    materialized = tuple(configs)
    if not materialized:
        raise ValueError("HeroJourney suite requires at least one case.")
    if max_workers < 1:
        raise ValueError("HeroJourney suite max_workers must be at least 1.")
    case_ids = [config.case_id for config in materialized]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("HeroJourney suite case_id values must be unique.")
    case_plays = tuple(create_hero_journey_play(config) for config in materialized)

    def play(agent: Any) -> dict[str, Any]:
        case_results = _run_plays(case_plays, agent=agent, max_workers=max_workers)
        scores = [result["selection_score"] for result in case_results]
        costs = [result["cost"] for result in case_results]

        return {
            "valid": all(result["valid"] for result in case_results),
            "selection_score": statistics.fmean(scores),
            "metrics": {
                "case_count": len(case_results),
                "mean_ecsr": statistics.fmean(scores),
            },
            "cases": case_results,
            "feedback": {
                result["case_id"]: result["feedback"] for result in case_results
            },
            "artifacts": {
                result["case_id"]: result["artifacts"]["episode"]
                for result in case_results
            },
            "cost": {
                "prompt_tokens": sum(cost.get("prompt_tokens", 0) for cost in costs),
                "completion_tokens": sum(
                    cost.get("completion_tokens", 0) for cost in costs
                ),
            },
            "err": None,
        }

    return play


def _run_plays(
    plays: Sequence[Callable[[Any], dict[str, Any]]],
    *,
    agent: Any,
    max_workers: int,
) -> list[dict[str, Any]]:
    """Run independent cases in input order, isolating each concurrent Agent."""

    worker_count = min(max_workers, len(plays))
    if worker_count == 1:
        return [play(agent) for play in plays]

    fork = getattr(agent, "fork", None)
    if callable(fork):
        agents = [fork() for _ in plays]
    else:
        # Stateless runtime agents (e.g. a container exec per call) can share
        # one agent across concurrent plays without process isolation.
        agents = [agent for _ in plays]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda pair: pair[0](pair[1]), zip(plays, agents)))
