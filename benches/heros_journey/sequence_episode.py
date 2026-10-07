"""Execute a one-call Hero action sequence in the original environment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from benchmark.contracts import BackendResponse

from .sequence_prompt import build_sequence_prompt

from herosjourney.env.env import AdventureEnv
from herosjourney.eval.result import EpisodeResult
from herosjourney.runner.adventure_episode import Action


@dataclass(frozen=True, slots=True)
class SequenceEpisodeOutcome:
    episode: EpisodeResult
    prompt: str
    response: BackendResponse
    parse_error: str | None
    planned_action_count: int
    action_limit: int


def _parse_actions(text: str) -> list[Action]:
    """Validate the complete reply before any action changes the environment."""
    payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("actions"), list):
        raise ValueError("expected a JSON object with an actions array")
    if not payload["actions"]:
        raise ValueError("actions array must not be empty")
    for index, item in enumerate(payload["actions"]):
        if not isinstance(item, dict) or set(item) - {"action", "argument", "reasoning"}:
            raise ValueError(f"actions[{index}] has an invalid shape")
    return [Action.model_validate(item) for item in payload["actions"]]


def _prepare_environment(task: Any, source_tasks: list[Any], initial_currency: int):
    trees = [(item.tree, item.tree.root_id) for item in (task, *source_tasks)]
    env = AdventureEnv(
        trees=trees, initial_currency=initial_currency,
        initial_location="GameStart",
    )
    rule_seed = int(hashlib.md5(task.tree.root_id.encode()).hexdigest()[:8], 16) % (2**31)
    initial_observation = env.reset(
        tree_index=0, initial_currency=initial_currency,
        initial_location="GameStart", seed=rule_seed,
        rules_to_skip=task.rules_to_skip, task_label="Your task",
    )
    return env, initial_observation, task.tree.get_solution()


def _execute_actions(env: AdventureEnv, actions: list[Action], action_limit: int, prompt: str):
    """The environment validates each action; no model call occurs in this loop."""
    history: list[dict[str, Any]] = []
    action_history: list[str] = []
    trace = prompt
    completion_map: dict[str, float] = {}
    for index, action in enumerate(actions[:action_limit], start=1):
        full_action, observation, completion_map = env.step(action.action, action.argument)
        action_history.append(full_action)
        history.append({
            "step": index, "action": full_action,
            "observation": observation.message,
            "reasoning": action.reasoning,
            "success": observation.success,
        })
        trace += f"Action {index}: {full_action}\nObs {index}: {observation.message}\n>"
        if observation.done:
            break
    return action_history, history, completion_map, trace


def _episode_result(
    *, episode_idx: int, task: Any, reference_solution: list[str],
    action_budget_multiplier: int, response: BackendResponse,
    env: AdventureEnv, actions: list[str], history: list[dict[str, Any]],
    completion_map: dict[str, float], trace: str,
) -> EpisodeResult:
    num_runs = len(actions)
    success = env.done
    usage = response.usage
    return EpisodeResult(
        episode_idx=episode_idx, task_type=task.task_type, split=task.split,
        success=success, terminated=not success, num_runs=num_runs,
        reference_length=len(reference_solution), reference_solution=reference_solution,
        efficiency=len(reference_solution) / num_runs if success and num_runs else None,
        num_tries=action_budget_multiplier, full_trace=trace,
        action_history=actions, action_obs_reasoning_history=history,
        completion_map=completion_map,
        prompt_tokens=(usage.input_tokens or 0) if usage else 0,
        completion_tokens=(usage.output_tokens or 0) if usage else 0,
    )


def run_sequence_episode(
    *, episode_idx: int, task: Any, source_tasks: list[Any],
    demo_context: str, action_budget_multiplier: int,
    action_limit: int, initial_currency: int,
    no_reasoning: bool, agent: Any,
) -> SequenceEpisodeOutcome:
    """Ask once for a plan, then execute its validated actions in order."""
    env, initial_observation, reference_solution = _prepare_environment(
        task, source_tasks, initial_currency,
    )
    prompt = build_sequence_prompt(
        demo_context, initial_observation,
        action_limit=action_limit, no_reasoning=no_reasoning,
    )
    response = agent.act(prompt)
    try:
        actions = _parse_actions(response.text)
        parse_error = None
    except (ValueError, TypeError) as exc:
        actions, parse_error = [], f"{type(exc).__name__}: {exc}"

    action_history, history, completion_map, trace = _execute_actions(
        env, actions, action_limit, prompt,
    )
    if parse_error:
        trace += f"Invalid action sequence: {parse_error}\n"
    episode = _episode_result(
        episode_idx=episode_idx, task=task,
        reference_solution=reference_solution,
        action_budget_multiplier=action_budget_multiplier,
        response=response, env=env, actions=action_history,
        history=history, completion_map=completion_map, trace=trace,
    )
    return SequenceEpisodeOutcome(
        episode=episode, prompt=prompt, response=response,
        parse_error=parse_error, planned_action_count=len(actions),
        action_limit=action_limit,
    )
