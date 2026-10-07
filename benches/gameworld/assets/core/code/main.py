"""Prototype entry-point that ties together the multi-agent VLM framework."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

from catalog import build_runtime_config
from runtime import Agent, Coordinator, Evaluator, GameEnv
from tools.monitor.replay import register_auto_replay_hooks, trigger_auto_replays
from utils import (
    build_agent_clients,
    finalize_run_metadata,
    mark_run_running,
    prepare_run_artifacts,
    resolve_rerun_log_root,
    RERUN_POLICIES,
    setup_logging,
)

LOGGER = logging.getLogger(__name__)


def _load_resume_actions(resume_from: str | None, agents: list[Agent]):
    """Load executed actions in interaction order for deterministic replay."""
    if not resume_from:
        return [], {}
    agent_map = {agent.agent_id: agent for agent in agents}
    records = []
    for path in sorted(Path(resume_from).glob("agent_*/interactions.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                agent_id = record.get("agent_id")
                action = (record.get("output") or {}).get("executed_action")
                if agent_id in agent_map and isinstance(action, dict):
                    records.append((int(record.get("interaction_id") or 0), agent_id, action))
    records.sort(key=lambda item: (item[0], item[1]))
    counts = {agent.agent_id: 0 for agent in agents}
    actions = []
    for _, agent_id, action in records:
        actions.append((agent_map[agent_id], action))
        counts[agent_id] += 1
    return actions, counts


async def main(
    config_preset: str,
    headless: bool | None = None,
    port: int | None = None,
    log_root: str | None = None,
    ace_playbook: str | None = None,
    harness_playbook: str | None = None,
    rerun_policy: str = "auto",
    diagnosis_repeat_threshold: int = 0,
    diagnosis_recovery_steps: int = 1,
    diagnosis_cot: bool = False,
    mc_interaction_feedback: bool = False,
    mc_mine_guard: bool = False,
    resume_from: str | None = None,
    client_factory: Callable | None = None,
    max_steps_override: int | None = None,
    task_assets_root: str | Path | None = None,
):
    """Main entry point for the game worlds.

    Args:
        config_preset: Catalog preset spec (game_id+task_id+model1,model2,...).
    """
    runtime_config = build_runtime_config(config_preset, task_assets_root=task_assets_root)
    if max_steps_override is not None:
        if max_steps_override <= 0:
            raise ValueError("max_steps_override must be positive")
        runtime_config.max_steps = max_steps_override
    rerun_resolution = resolve_rerun_log_root(
        log_root,
        rerun_policy=rerun_policy,
        config_preset=config_preset,
    )
    if rerun_resolution.skip:
        LOGGER.info(
            "Skipping run because %s: %s",
            rerun_resolution.reason,
            rerun_resolution.log_root,
        )
        return
    if rerun_resolution.log_root != log_root:
        LOGGER.info(
            "Using rerun log root %s for previous run %s",
            rerun_resolution.log_root,
            rerun_resolution.metadata.get("source_run")
            if rerun_resolution.metadata
            else log_root,
        )
    log_root = rerun_resolution.log_root
    if ace_playbook:
        from ace.playbook import inject_playbook

        playbook_text = Path(ace_playbook).read_text(encoding="utf-8")
        runtime_config.system_prompts = [
            inject_playbook(prompt, playbook_text)
            for prompt in runtime_config.system_prompts
        ]
    if harness_playbook:
        from ace.playbook import inject_named_playbook

        playbook_text = Path(harness_playbook).read_text(encoding="utf-8")
        runtime_config.system_prompts = [
            inject_named_playbook(
                prompt,
                playbook_text,
                heading="Self-Improved Harness Playbook",
                tag="self_harness_playbook",
            )
            for prompt in runtime_config.system_prompts
        ]
    prepare_run_artifacts(
        runtime_config,
        config_preset=config_preset,
        port=port,
        log_root=log_root,
        rerun_metadata=rerun_resolution.metadata,
    )
    register_auto_replay_hooks(runtime_config.log_root)
    LOGGER.info(
        "Loading preset config: %s.\nAgent models: %s",
        config_preset,
        ", ".join(f"agent_{i}={mt}" for i, mt in enumerate(runtime_config.model_ids)),
    )

    agent_ids = [f"agent_{i}" for i in range(runtime_config.agent_count)]
    clients = (
        client_factory(runtime_config, agent_ids)
        if client_factory is not None
        else build_agent_clients(runtime_config, agent_ids)
    )
    if len(clients) != runtime_config.agent_count:
        raise ValueError(
            "client_factory returned an unexpected number of clients: "
            f"expected={runtime_config.agent_count}, actual={len(clients)}"
        )
    agents = [
        Agent(
            agent_id=agent_id,
            agent_type=clients[idx].config.model_type,
            client=clients[idx],
            controls=runtime_config.role_controls_maps[idx],
            semantic_controls_map=runtime_config.semantic_controls_maps[idx],
        )
        for idx, agent_id in enumerate(agent_ids)
    ]

    env = GameEnv(runtime_config, headless=headless, port=port)
    evaluator = Evaluator(runtime_config)
    coordinator = Coordinator(
        env=env,
        agents=agents,
        evaluator=evaluator,
        diagnosis_repeat_threshold=diagnosis_repeat_threshold,
        diagnosis_recovery_steps=diagnosis_recovery_steps,
        diagnosis_cot=diagnosis_cot,
        mc_interaction_feedback=mc_interaction_feedback,
        mc_mine_guard=mc_mine_guard,
    )

    final_return_code = 1
    final_status = "error"

    try:
        mark_run_running(runtime_config)
        if resume_from:
            await env.start()
            resume_actions, resume_counts = _load_resume_actions(resume_from, agents)
            replayed = await env.replay_actions(resume_actions)
            for agent in agents:
                agent.step_index = resume_counts.get(agent.agent_id, 0)
            LOGGER.info(
                "Resumed from %s: replayed %d actions; step counts=%s",
                resume_from, replayed, resume_counts,
            )
        await coordinator.run(start_env=not bool(resume_from))  # main loop
        final_return_code = 0
        final_status = "completed"

    except asyncio.CancelledError:
        LOGGER.info("Main loop cancelled")
        final_return_code = 130
        final_status = "error"
        trigger_auto_replays(runtime_config.log_root, reason="cancelled")
        raise
    except KeyboardInterrupt:
        LOGGER.info("Main loop interrupted")
        final_return_code = 130
        final_status = "error"
        trigger_auto_replays(runtime_config.log_root, reason="interrupted")
        raise
    except Exception:
        final_return_code = 1
        final_status = "error"
        raise
    finally:
        finalize_run_metadata(
            runtime_config,
            status=final_status,
            return_code=final_return_code,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM agents to play browser games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Catalog format: game_id+task_id+model1,model2
    python main.py --config 01_2048+01_01+gpt-5.2
    python main.py --config 10_doodle-jump+10_05+qwen3-vl-235b-a22b-cua
    """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PRESET",
        help="Catalog preset spec: game_id+task_id+model1,model2 (required)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Local game server port (default: internal default).",
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Exact output directory for this run.",
    )
    parser.add_argument(
        "--task-assets-root",
        default=None,
        help="Task YAML directory (defaults to bundled holdout assets).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Replay executed actions from an incomplete run before continuing.",
    )
    parser.add_argument(
        "--ace-playbook",
        default=None,
        help="Frozen ACE playbook (.md) injected into every agent system prompt.",
    )
    parser.add_argument(
        "--harness-playbook",
        default=None,
        help="Frozen generic harness playbook (.md) injected into every agent system prompt.",
    )
    parser.add_argument(
        "--rerun-policy",
        choices=RERUN_POLICIES,
        default="auto",
        help=(
            "How to handle an existing --log-root: auto creates <run>_rerunN for "
            "incomplete runs and skips active/finalized runs; never reuses the "
            "directory; always creates <run>_rerunN when it exists."
        ),
    )
    parser.add_argument(
        "--diagnosis-repeat-threshold",
        type=int,
        default=0,
        help=(
            "Inject a transient recovery advisory after this many identical actions "
            "without evaluator progress (0 disables it)."
        ),
    )
    parser.add_argument(
        "--diagnosis-recovery-steps",
        type=int,
        default=1,
        help="Number of consecutive recovery-advised actions after a loop trigger.",
    )
    parser.add_argument(
        "--diagnosis-cot",
        action="store_true",
        help=(
            "Require an Evidence/Cause/Recovery reasoning chain on runtime "
            "diagnosis steps before the tool call."
        ),
    )
    parser.add_argument(
        "--mc-interaction-feedback",
        action="store_true",
        help=(
            "For Minecraft tasks, inject previous structured interaction feedback "
            "(targeted block, break progress, score delta) into the next model step."
        ),
    )
    parser.add_argument(
        "--mc-mine-guard",
        action="store_true",
        help=(
            "For Minecraft tasks, replace mine_target with a recovery action when "
            "the latest structured state says the crosshair is not hitting a "
            "reachable block."
        ),
    )
    headless_group = parser.add_mutually_exclusive_group()
    headless_group.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode (default when no display is detected).",
    )
    headless_group.add_argument(
        "--headed",
        dest="headless",
        action="store_false",
        help="Force headed mode (requires an X server or Wayland display).",
    )
    parser.set_defaults(headless=None)
    args = parser.parse_args()
    setup_logging()

    try:
        asyncio.run(
            main(
                config_preset=args.config,
                headless=args.headless,
                port=args.port,
                log_root=args.log_root,
                task_assets_root=args.task_assets_root,
                resume_from=args.resume_from,
                ace_playbook=args.ace_playbook,
                harness_playbook=args.harness_playbook,
                rerun_policy=args.rerun_policy,
                diagnosis_repeat_threshold=args.diagnosis_repeat_threshold,
                diagnosis_recovery_steps=args.diagnosis_recovery_steps,
                diagnosis_cot=args.diagnosis_cot,
                mc_interaction_feedback=args.mc_interaction_feedback,
                mc_mine_guard=args.mc_mine_guard,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("Shutting down due to keyboard interrupt")
