"""Coordinator loop for agents, browser environment, and evaluation."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from agents.harness.semantic_controls import inspect_semantic_controls_output, map_semantic_controls_output
from tools.runtime_logger import RuntimeLogger

from .env import GameEnv
from .evaluator import Evaluator
from .game_agent import ActionPayload, Agent

LOGGER = logging.getLogger(__name__)

NO_GAME_STATE_SUMMARY = "(no game state)"
AGENT_LOOP_SLEEP_S = 0.05
MC_MIN_MINE_HOLD_SECONDS = 3.0
MC_MAX_RELIABLE_MINE_DISTANCE = 4.8
MC_STONE_BLOCKER_TYPES = {"grass", "dirt", "coal_ore", "iron_ore"}
MC_STONE_ORE_BLOCKER_TYPES = {"coal_ore", "iron_ore"}
MC_STONE_BLOCKER_NEAR_TARGET_DISTANCE = 2.25


class Coordinator:
    """Coordinates agents, environment, and evaluation in the main loop."""

    def __init__(
        self,
        env: GameEnv,
        agents: list[Agent],
        evaluator: Evaluator | None = None,
        diagnosis_repeat_threshold: int = 0,
        diagnosis_recovery_steps: int = 1,
        diagnosis_cot: bool = False,
        mc_interaction_feedback: bool = False,
        mc_mine_guard: bool = False,
    ):
        if not agents:
            raise ValueError("Coordinator requires at least one agent.")

        self.env = env
        self.agents = agents
        self.evaluator = evaluator
        self._stop_event = asyncio.Event()
        self.agent_loggers: dict[str, RuntimeLogger] = {}
        self._evaluation_agent_id = self.agents[0].agent_id if self.agents else None
        self._fatal_error: Exception | None = None
        self._diagnosis_repeat_threshold = max(0, int(diagnosis_repeat_threshold))
        self._diagnosis_recovery_steps = max(1, int(diagnosis_recovery_steps))
        self._diagnosis_cot = bool(diagnosis_cot)
        self._mc_interaction_feedback = bool(mc_interaction_feedback)
        self._mc_mine_guard = bool(mc_mine_guard)
        self._recent_action_ids: dict[str, list[str]] = {
            agent.agent_id: [] for agent in self.agents
        }
        self._diagnostic_recovery_ids: dict[str, list[str]] = {
            agent.agent_id: [] for agent in self.agents
        }
        self._diagnosis_pending: set[str] = set()
        self._active_recovery_remaining: dict[str, int] = {
            agent.agent_id: 0 for agent in self.agents
        }
        self._active_blocked_ids: dict[str, list[str]] = {
            agent.agent_id: [] for agent in self.agents
        }
        self._latest_state_by_agent: dict[str, dict[str, Any] | None] = {
            agent.agent_id: None for agent in self.agents
        }
        self._mc_guard_counts: dict[str, int] = {agent.agent_id: 0 for agent in self.agents}
        self._mc_mining_streaks: dict[str, dict[str, Any]] = {
            agent.agent_id: {"target": None, "count": 0}
            for agent in self.agents
        }
        self._init_agent_loggers()

    @staticmethod
    def _get_nested(data: dict[str, Any] | None, path: str) -> Any:
        current: Any = data
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _is_minecraft_task(self) -> bool:
        env = getattr(self, "env", None)
        config = getattr(env, "config", None)
        return getattr(config, "game_id", None) == "18_minecraft-clone-glm"

    def _target_resource_hint(self) -> str:
        env = getattr(self, "env", None)
        config = getattr(env, "config", None)
        evaluator_config = getattr(config, "evaluator_config", {})
        if not isinstance(evaluator_config, dict):
            evaluator_config = {}
        score_field = str(evaluator_config.get("score_field") or "")
        marker = "game_state.inventory.item_gains."
        if score_field.startswith(marker):
            return score_field.removeprefix(marker)
        if score_field == "metrics.collected_positive_total":
            return "any newly collectable block"
        return score_field or "the requested task resource"

    def _minecraft_target_matches_task(self, target_type: Any) -> bool:
        target_resource = self._target_resource_hint()
        if not target_type or target_resource == "any newly collectable block":
            return True
        if target_type == target_resource:
            return True
        # Stone tasks may require digging through cover or nearby ore first to
        # open a reachable shaft/face. These are legitimate setup blocks, but
        # only stone advances the target metric. Wood-log tasks should not
        # spend actions on grass/leaves because they do not advance the target
        # metric and are not useful blockers for trunks.
        if target_resource == "stone" and target_type in MC_STONE_BLOCKER_TYPES:
            return True
        return False

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _minecraft_stone_blocker_advice(
        *,
        target_resource: str,
        target_type: Any,
        target_distance: Any,
        target_scan: dict[str, Any] | None,
    ) -> str | None:
        if target_resource != "stone" or target_type not in MC_STONE_ORE_BLOCKER_TYPES:
            return None
        distance = Coordinator._safe_float(target_distance)
        if distance is None or distance > MC_MAX_RELIABLE_MINE_DISTANCE:
            return None
        nearest_stone_distance = None
        if isinstance(target_scan, dict):
            nearest_stone_distance = Coordinator._safe_float(target_scan.get("nearest_distance"))
        if nearest_stone_distance is None or nearest_stone_distance > MC_STONE_BLOCKER_NEAR_TARGET_DISTANCE:
            return None
        return (
            "- Immediate blocker recovery: the crosshair is on a reachable "
            f"`{target_type}` and scanned stone is very close "
            f"(nearest_distance={nearest_stone_distance:.2f}). Treat this ore as a "
            "temporary blocker and choose `mine_target` now; do not spend this step "
            "on more look/strafe adjustment."
        )

    @staticmethod
    def _minecraft_resource_scan_for_target(
        nearby_resources: Any,
        target_resource: str,
    ) -> dict[str, Any] | None:
        if not isinstance(nearby_resources, dict):
            return None
        if target_resource == "any newly collectable block":
            candidates = ["wood_log", "grass", "dirt", "stone", "leaves"]
            available = [
                item
                for item in (nearby_resources.get(candidate) for candidate in candidates)
                if isinstance(item, dict) and int(item.get("count") or 0) > 0
            ]
            if not available:
                return None
            return min(
                available,
                key=lambda item: float(item.get("nearest_distance") or float("inf")),
            )
        target_scan = nearby_resources.get(target_resource)
        return target_scan if isinstance(target_scan, dict) else None

    @staticmethod
    def _minecraft_local_direction_hint(
        resource_scan: dict[str, Any] | None,
        player: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        if not isinstance(resource_scan, dict):
            return None, None
        try:
            dx = float(resource_scan.get("nearest_dx"))
            dy = float(resource_scan.get("nearest_dy"))
            dz = float(resource_scan.get("nearest_dz"))
        except (TypeError, ValueError):
            return None, None

        try:
            yaw = float(player.get("yaw") or 0.0)
        except (TypeError, ValueError):
            yaw = 0.0

        # In this benchmark, yaw=0 moves forward toward negative z.  Positive x
        # is camera-right at yaw=0.  Convert world dx/dz into current facing
        # coordinates so the prompt can recommend registered actions.
        forward_x = math.sin(-yaw)
        forward_z = -math.cos(yaw)
        right_x = math.cos(yaw)
        right_z = math.sin(-yaw)
        forward = dx * forward_x + dz * forward_z
        right = dx * right_x + dz * right_z
        local_text = (
            f"relative_to_facing=(right={right:.2f}, forward={forward:.2f}, "
            f"up={dy:.2f})"
        )

        suggestions: list[str] = []
        if abs(right) >= 1.0 and abs(right) >= abs(forward) * 0.6:
            side = "right" if right > 0 else "left"
            suggestions.append(f"use move_{side} and/or look_{side}")
        if forward >= 1.0:
            suggestions.append("use move_forward")
        elif forward <= -1.0:
            suggestions.append("the resource is behind current facing; turn with look_left/look_right or step backward before mining")
        if dy >= 1.0:
            suggestions.append("use look_up to put the trunk/log face on the crosshair")
        elif dy <= -1.0:
            suggestions.append("use look_down to put the target block on the crosshair")
        if not suggestions:
            suggestions.append("fine-adjust camera with look_left/look_right/look_up/look_down until targeted_block_type matches")
        return local_text, "; ".join(suggestions)

    def _classify_minecraft_interaction_issue(
        self,
        *,
        target_exists: bool,
        target_type: Any,
        target_distance: Any,
        previous_action: Any,
        same_target_mines: int,
        score_delta: Any,
    ) -> str:
        if not target_exists:
            return "crosshair_misses_block"
        if not self._minecraft_target_matches_task(target_type):
            return "wrong_target_type"
        if isinstance(target_distance, (int, float)) and target_distance > MC_MAX_RELIABLE_MINE_DISTANCE:
            return "target_near_reach_limit"
        if previous_action == "mine_target" and same_target_mines >= 1 and score_delta == 0:
            return "mine_action_made_no_score_progress"
        return "ready_to_mine"

    @staticmethod
    def _minecraft_target_signature(state: dict[str, Any] | None) -> tuple[Any, ...] | None:
        targeted_block = Coordinator._get_nested(
            state,
            "game_state.environment.targeted_block",
        )
        if not isinstance(targeted_block, dict) or targeted_block.get("exists") is not True:
            return None
        return (
            targeted_block.get("x"),
            targeted_block.get("y"),
            targeted_block.get("z"),
            targeted_block.get("block_type"),
        )

    def _update_minecraft_mining_streak(
        self,
        agent: Agent,
        raw_action: ActionPayload,
        state: dict[str, Any] | None,
    ) -> None:
        if not self._is_minecraft_task() or not isinstance(raw_action, dict):
            return

        action_id = str(raw_action.get("tool_name") or "")
        score_current = agent.eval_metrics.get("score_current")
        target_signature = self._minecraft_target_signature(state)
        streak = self._mc_mining_streaks.setdefault(
            agent.agent_id,
            {"target": None, "count": 0, "score_current": None},
        )
        previous_score = streak.get("score_current")
        streak["score_current"] = score_current
        score_improved = (
            isinstance(score_current, (int, float))
            and isinstance(previous_score, (int, float))
            and score_current > previous_score
        )
        if action_id != "mine_target" or score_improved or target_signature is None:
            streak["target"] = target_signature
            streak["count"] = 0
            return

        if streak.get("target") == target_signature:
            streak["count"] = int(streak.get("count") or 0) + 1
        else:
            streak["target"] = target_signature
            streak["count"] = 1

    def _build_minecraft_feedback_advisory(self, agent: Agent) -> str | None:
        if not getattr(self, "_mc_interaction_feedback", False) or not self._is_minecraft_task():
            return None

        state = self._latest_state_by_agent.get(agent.agent_id)
        if not isinstance(state, dict):
            return (
                "Minecraft interaction feedback is enabled. On each step, mine only "
                "after the crosshair is actually targeting a reachable block."
            )

        targeted_block = self._get_nested(state, "game_state.environment.targeted_block")
        if not isinstance(targeted_block, dict):
            targeted_block = {}
        player = self._get_nested(state, "game_state.player")
        if not isinstance(player, dict):
            player = {}
        nearby_resources = self._get_nested(state, "game_state.environment.nearby_resources")

        target_exists = targeted_block.get("exists") is True
        target_type = targeted_block.get("block_type")
        target_distance = targeted_block.get("distance")
        break_progress = self._get_nested(state, "game_state.environment.break_progress")
        score_delta = agent.eval_metrics.get("score_delta")
        score_current = agent.eval_metrics.get("score_current")
        no_progress_steps = agent.eval_metrics.get("no_progress_steps")
        previous_action = (self._recent_action_ids.get(agent.agent_id) or [None])[-1]
        target_resource = self._target_resource_hint()
        target_scan = self._minecraft_resource_scan_for_target(nearby_resources, target_resource)
        target_local_direction, target_action_hint = self._minecraft_local_direction_hint(
            target_scan,
            player,
        )
        mining_streaks = getattr(self, "_mc_mining_streaks", {})
        mining_streak = mining_streaks.get(agent.agent_id) or {}
        same_target_mines = int(mining_streak.get("count") or 0)
        diagnosis = self._classify_minecraft_interaction_issue(
            target_exists=target_exists,
            target_type=target_type,
            target_distance=target_distance,
            previous_action=previous_action,
            same_target_mines=same_target_mines,
            score_delta=score_delta,
        )
        stone_blocker_advice = self._minecraft_stone_blocker_advice(
            target_resource=target_resource,
            target_type=target_type,
            target_distance=target_distance,
            target_scan=target_scan,
        )

        lines = [
            "# Minecraft Interaction Feedback",
            f"- Task target resource: {target_resource}.",
            f"- Previous action: {previous_action or 'none'}.",
            (
                "- Last observed crosshair target: "
                f"exists={target_exists}, type={target_type}, distance={target_distance}."
            ),
            f"- Last break_progress={break_progress}; score_current={score_current}; score_delta={score_delta}; no_progress_steps={no_progress_steps}.",
            f"- Consecutive no-score mines on the same targeted block: {same_target_mines}.",
            f"- Runtime diagnosis category: {diagnosis}.",
            (
                "- Rule: choose `mine_target` only if the current crosshair is hitting "
                "a reachable block face. If exists=false, first change camera/position."
            ),
            (
                f"- Action calibration: `mine_target` is executed as one long hold "
                f"(at least {MC_MIN_MINE_HOLD_SECONDS:.1f}s). Short holds can reset "
                "break progress on mouseup before any item drops."
            ),
        ]

        if isinstance(target_scan, dict):
            lines.append(
                f"- Nearest scanned {target_resource}: count={target_scan.get('count')}, "
                f"distance={target_scan.get('nearest_distance')}, "
                f"world=({target_scan.get('nearest_x')}, {target_scan.get('nearest_y')}, {target_scan.get('nearest_z')}), "
                f"relative_world=(dx={target_scan.get('nearest_dx')}, dy={target_scan.get('nearest_dy')}, dz={target_scan.get('nearest_dz')})."
            )
            if target_resource == "stone":
                lines.append(
                    "- Stone setup rule: grass, dirt, coal_ore, and iron_ore may be mined "
                    "only as temporary blockers to open a reachable face toward stone; "
                    "they do not count as stone progress."
                )
            if target_local_direction:
                lines.append(f"- Nearest scanned {target_resource} direction: {target_local_direction}.")
            if target_action_hint:
                lines.append(
                    f"- Navigation hint toward scanned {target_resource}: {target_action_hint}. "
                    "After moving/turning, mine when the crosshair targets the task resource "
                    "or an explicitly allowed setup blocker."
                )
        elif target_resource != "the requested task resource":
            lines.append(
                f"- Nearest scanned {target_resource}: not present in nearby_resources. "
                "Use visual navigation until the scan reports the task resource or the crosshair targets it."
            )

        if stone_blocker_advice:
            lines.append(stone_blocker_advice)
        elif previous_action == "mine_target" and not target_exists:
            lines.append(
                "- Diagnosis: the last mining action hit air (`targeted_block_exists=false`). "
                "Do not repeat `mine_target` from this same pose; use look_down/look_left/look_right "
                "or move_forward to make the crosshair raycast hit a block face."
            )
        elif target_exists and not self._minecraft_target_matches_task(target_type):
            lines.append(
                "- Diagnosis: the crosshair is on the wrong material for this task. "
                "Do not mine it unless it is an explicit setup block; change aim or "
                "position until the target type matches the requested resource."
            )
        elif (
            target_exists
            and isinstance(target_distance, (int, float))
            and target_distance > MC_MAX_RELIABLE_MINE_DISTANCE
        ):
            lines.append(
                "- Diagnosis: the target is near the reach boundary. Move closer or "
                "adjust aim before mining; otherwise mining can appear valid but fail."
            )
        elif target_exists and same_target_mines < 4:
            lines.append(
                "- Diagnosis: the crosshair is on a concrete block. `break_progress` may stay "
                "0 after mouseup even when mining is working. Prefer one calibrated "
                "long `mine_target` hold over repeated short clicks."
            )
        elif target_exists and same_target_mines >= 4:
            lines.append(
                "- Diagnosis: repeated mining on the same targeted block has not produced score. "
                "Now change distance or aim, then reacquire a block before mining."
            )

        return "\n".join(lines)

    def _select_minecraft_recovery_action(self, agent: Agent, *, reason: str) -> dict[str, Any] | None:
        available = set(agent.semantic_controls_map or {})
        count = self._mc_guard_counts.get(agent.agent_id, 0)
        self._mc_guard_counts[agent.agent_id] = count + 1

        state = self._latest_state_by_agent.get(agent.agent_id)
        player = self._get_nested(state, "game_state.player")
        player = player if isinstance(player, dict) else {}
        nearby_resources = self._get_nested(state, "game_state.environment.nearby_resources")
        target_resource = self._target_resource_hint()
        target_scan = self._minecraft_resource_scan_for_target(nearby_resources, target_resource)
        nearest_target_distance = None
        if isinstance(target_scan, dict):
            nearest_target_distance = self._safe_float(target_scan.get("nearest_distance"))
        pitch = self._safe_float(player.get("pitch"))
        if (
            "targeted_block_exists=false" in reason
            and target_resource == "stone"
            and nearest_target_distance is not None
            and nearest_target_distance <= MC_STONE_BLOCKER_NEAR_TARGET_DISTANCE
            and pitch is not None
            and pitch <= -1.3
            and "look_up" in available
        ):
            return {
                "tool_name": "look_up",
                "arguments": {
                    "reasoning": (
                        "Runtime MC guard detected nearby stone while the crosshair ray "
                        "misses at a near-vertical downward pitch. Look up one notch to "
                        "reacquire a side or floor block face before mining."
                    )
                },
                "runtime_override": {
                    "method": "mc_interaction_feedback_v3",
                    "original_tool_name": "mine_target",
                    "reason": reason,
                    "nearest_target_distance": nearest_target_distance,
                    "pitch": pitch,
                },
            }

        sequence = ["look_down", "move_forward", "look_left", "look_right", "move_right", "move_left"]
        for offset in range(len(sequence)):
            action_id = sequence[(count + offset) % len(sequence)]
            if action_id in available:
                return {
                    "tool_name": action_id,
                    "arguments": {
                        "reasoning": (
                            f"Runtime MC guard replaced ineffective mine_target: {reason}. "
                            "Recover by changing aim/position before mining."
                        )
                    },
                    "runtime_override": {
                        "method": "mc_interaction_feedback_v2",
                        "original_tool_name": "mine_target",
                        "reason": reason,
                    },
                }
        return None

    def _maybe_guard_minecraft_action(self, agent: Agent, raw_action: ActionPayload) -> ActionPayload:
        if not getattr(self, "_mc_mine_guard", False) or not self._is_minecraft_task():
            return raw_action
        if agent.agent_type != "generalist" or not isinstance(raw_action, dict):
            return raw_action
        if str(raw_action.get("tool_name") or "") != "mine_target":
            return raw_action

        state = self._latest_state_by_agent.get(agent.agent_id)
        targeted_block = self._get_nested(state, "game_state.environment.targeted_block")
        targeted_block = targeted_block if isinstance(targeted_block, dict) else {}
        target_exists = targeted_block.get("exists") is True
        target_type = targeted_block.get("block_type")
        target_distance = targeted_block.get("distance")
        if not target_exists:
            reason = "targeted_block_exists=false"
        elif not self._minecraft_target_matches_task(target_type):
            reason = (
                f"targeted_block_type={target_type} does not match task target "
                f"{self._target_resource_hint()}"
            )
        elif isinstance(target_distance, (int, float)) and target_distance > MC_MAX_RELIABLE_MINE_DISTANCE:
            reason = f"targeted_block_distance={target_distance:.2f} is near the reach boundary"
        else:
            return raw_action

        recovery_action = self._select_minecraft_recovery_action(agent, reason=reason)
        if recovery_action is None:
            return raw_action
        LOGGER.info(
            "MC interaction guard (%s): %s -> %s because %s",
            agent.agent_id,
            raw_action,
            recovery_action,
            reason,
        )
        return recovery_action

    def _calibrate_minecraft_resolved_action(
        self,
        agent: Agent,
        raw_action: ActionPayload,
        resolved_action: ActionPayload,
    ) -> ActionPayload:
        """Patch known MC action-primitive failures at execution time.

        This is deliberately below the prompt layer: if the model correctly
        chooses `mine_target`, a too-short browser hold should not make the
        task fail by resetting break progress on mouseup.
        """
        if not self._is_minecraft_task() or agent.agent_type != "generalist":
            return resolved_action
        if not isinstance(raw_action, dict) or str(raw_action.get("tool_name") or "") != "mine_target":
            return resolved_action
        if not isinstance(resolved_action, dict) or resolved_action.get("action") != "click_hold":
            return resolved_action

        calibrated = dict(resolved_action)
        old_duration = calibrated.get("duration")
        try:
            old_duration_value = float(old_duration)
        except (TypeError, ValueError):
            old_duration_value = 0.0
        if old_duration_value >= MC_MIN_MINE_HOLD_SECONDS:
            return resolved_action

        calibrated["duration"] = MC_MIN_MINE_HOLD_SECONDS
        calibrated["runtime_calibration"] = {
            "method": "mc_action_primitive_calibration_v1",
            "reason": "mine_target short holds can reset break_progress on mouseup before item gain",
            "old_duration": old_duration,
            "min_duration": MC_MIN_MINE_HOLD_SECONDS,
        }
        LOGGER.info(
            "MC action calibration (%s): mine_target duration %s -> %.1f",
            agent.agent_id,
            old_duration,
            MC_MIN_MINE_HOLD_SECONDS,
        )
        return calibrated

    def _set_recovery_advisory(
        self,
        agent: Agent,
        *,
        reason: str,
        blocked_actions: list[str],
    ) -> None:
        blocked_text = ", ".join(f"`{item}`" for item in blocked_actions)
        existing = getattr(agent.client, "_runtime_advisory", None)
        prefix = f"{existing.strip()}\n\n" if isinstance(existing, str) and existing.strip() else ""
        agent.client.set_runtime_advisory(
            f"{prefix}{reason} The task metric made no progress. For this recovery step, "
            f"do not select any of: {blocked_text}. Select a different registered "
            "action that changes viewpoint, position, target, selected control, or "
            "interaction type. The task and action interface are unchanged.",
            require_reasoning=self._diagnosis_cot,
        )
        self._diagnosis_pending.add(agent.agent_id)

    def _update_runtime_diagnosis(self, agent: Agent) -> None:
        """Inject a task-independent recovery warning only after a verified loop."""
        agent.client.set_runtime_advisory(None)
        mc_advisory = self._build_minecraft_feedback_advisory(agent)
        if mc_advisory:
            agent.client.set_runtime_advisory(mc_advisory)
        threshold = self._diagnosis_repeat_threshold
        recent = self._recent_action_ids.get(agent.agent_id, [])
        no_progress_steps = agent.eval_metrics.get("no_progress_steps", 0)
        if no_progress_steps == 0:
            self._diagnostic_recovery_ids.setdefault(agent.agent_id, []).clear()
            self._active_recovery_remaining[agent.agent_id] = 0
            self._active_blocked_ids.setdefault(agent.agent_id, []).clear()
        if threshold <= 0:
            return

        remaining = self._active_recovery_remaining.get(agent.agent_id, 0)
        if remaining > 0:
            blocked_actions = list(
                dict.fromkeys(
                    [
                        *self._active_blocked_ids.get(agent.agent_id, []),
                        *self._diagnostic_recovery_ids.get(agent.agent_id, []),
                    ]
                )
            )
            self._set_recovery_advisory(
                agent,
                reason=f"Recovery sequence active ({remaining} steps remaining).",
                blocked_actions=blocked_actions,
            )
            LOGGER.info(
                "Runtime diagnosis continuation (%s): remaining=%s blocked=%s",
                agent.agent_id,
                remaining,
                blocked_actions,
            )
            return
        if (
            len(recent) < threshold
            or len(set(recent[-threshold:])) != 1
            or not isinstance(no_progress_steps, int)
            or no_progress_steps < threshold
        ):
            return

        repeated_action = recent[-1]
        if repeated_action == "no_function_call":
            return
        prior_recoveries = self._diagnostic_recovery_ids.get(agent.agent_id, [])
        blocked_actions = list(dict.fromkeys([repeated_action, *prior_recoveries]))
        self._active_recovery_remaining[agent.agent_id] = self._diagnosis_recovery_steps
        self._active_blocked_ids[agent.agent_id] = [repeated_action]
        self._set_recovery_advisory(
            agent,
            reason=(
                f"The loop detector fired: the last {threshold} selected actions "
                f"were `{repeated_action}`. Previously attempted recovery actions "
                "without subsequent progress are also blocked."
            ),
            blocked_actions=blocked_actions,
        )
        LOGGER.info(
            "Runtime diagnosis (%s): repeated_action=%s count=%s no_progress_steps=%s",
            agent.agent_id,
            repeated_action,
            threshold,
            no_progress_steps,
        )

    def _record_action_choice(self, agent: Agent, raw_action: ActionPayload) -> None:
        action_id = "no_function_call"
        if isinstance(raw_action, dict):
            candidate = str(raw_action.get("tool_name") or "").strip()
            if candidate:
                action_id = candidate
        if agent.agent_id in self._diagnosis_pending:
            self._diagnosis_pending.discard(agent.agent_id)
            self._active_recovery_remaining[agent.agent_id] = max(
                0,
                self._active_recovery_remaining.get(agent.agent_id, 0) - 1,
            )
            if action_id != "no_function_call":
                recoveries = self._diagnostic_recovery_ids.setdefault(agent.agent_id, [])
                if action_id not in recoveries:
                    recoveries.append(action_id)
        recent = self._recent_action_ids.setdefault(agent.agent_id, [])
        recent.append(action_id)
        keep = max(1, self._diagnosis_repeat_threshold)
        del recent[:-keep]

    @staticmethod
    def _build_runtime_logger(agent: Agent, env: GameEnv) -> RuntimeLogger:
        client = agent.client
        return RuntimeLogger(
            log_dir=client.config.log_dir,
            session_id=client.config.log_session_id,
            game_name=env.config.game_id,
            model_name=client.config.model,
            agent_id=agent.agent_id,
            session_root=getattr(client.config, "log_root", None),
            memory_screenshot_mode=getattr(client.config, "memory_screenshot_mode", "path"),
        )

    def _init_agent_loggers(self) -> None:
        for agent in self.agents:
            self.agent_loggers[agent.agent_id] = self._build_runtime_logger(agent, self.env)

    def _log_model_interaction(
        self,
        agent: Agent,
        agent_logger: RuntimeLogger | None,
    ) -> dict[str, Any] | None:
        trace = agent.client.pop_logged_interaction()
        if agent_logger:
            agent_logger.log_interaction_from_trace(trace)
        return trace

    @staticmethod
    def _count_proposed_actions(action: ActionPayload) -> int:
        return len(Coordinator._collect_proposed_actions(action))

    @staticmethod
    def _collect_proposed_actions(action: ActionPayload) -> list[Any]:
        if isinstance(action, list):
            return list(action) or [None]

        if isinstance(action, dict):
            return [action]

        return [action]

    def _build_action_validity_record(
        self,
        agent: Agent,
        raw_action: ActionPayload,
        resolved_action: ActionPayload,
        trace: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trace = trace or {}
        executor = self.env._get_executor(agent)

        if agent.agent_type == "generalist":
            proposed_tool_call = trace.get("tool_call")
            semantic = inspect_semantic_controls_output(
                proposed_tool_call,
                agent.semantic_controls_map,
            )
            low_level = executor.inspect_action(
                resolved_action if isinstance(resolved_action, dict) else None
            )
            is_valid = bool(semantic.get("is_valid")) and bool(low_level.get("is_valid"))
            invalid_kind = None
            reason = "valid"
            if not semantic.get("is_valid"):
                invalid_kind = semantic.get("invalid_kind")
                reason = str(semantic.get("reason") or "invalid_semantic_action")
            elif not low_level.get("is_valid"):
                invalid_kind = low_level.get("invalid_kind")
                reason = str(low_level.get("reason") or "invalid_low_level_action")

            return {
                "agent_type": agent.agent_type,
                "is_valid": is_valid,
                "reason": reason,
                "invalid_kind": invalid_kind,
                "proposed_action_count": 1,
                "valid_action_count": 1 if is_valid else 0,
                "raw_action": raw_action,
                "raw_tool_call": proposed_tool_call,
                "resolved_action": resolved_action,
                "normalized_action": low_level.get("normalized_action"),
                "semantic_control_id": semantic.get("control_id"),
            }

        proposed_count = self._count_proposed_actions(raw_action)
        if trace.get("error"):
            return {
                "agent_type": agent.agent_type,
                "is_valid": False,
                "reason": "client_parse_error",
                "invalid_kind": "no_function_call",
                "proposed_action_count": proposed_count,
                "valid_action_count": 0,
                "raw_action": raw_action,
                "resolved_action": resolved_action,
                "normalized_action": None,
            }

        proposed_actions = self._collect_proposed_actions(raw_action)
        inspections = [executor.inspect_action(item) for item in proposed_actions]
        valid_count = sum(1 for item in inspections if item.get("is_valid"))
        first_invalid = next((item for item in inspections if not item.get("is_valid")), None)
        is_valid = valid_count == proposed_count
        return {
            "agent_type": agent.agent_type,
            "is_valid": is_valid,
            "reason": (
                "valid"
                if is_valid
                else str((first_invalid or {}).get("reason") or "invalid_action")
            ),
            "invalid_kind": None if is_valid else (first_invalid or {}).get("invalid_kind"),
            "proposed_action_count": proposed_count,
            "valid_action_count": valid_count,
            "raw_action": raw_action,
            "resolved_action": resolved_action,
            "normalized_action": (inspections[0].get("normalized_action") if inspections else None),
        }

    async def _capture_state(self) -> tuple[dict[str, Any] | None, str]:
        snapshot = await self.env.capture_state()
        if not snapshot:
            return None, NO_GAME_STATE_SUMMARY
        return snapshot.state, snapshot.summary

    @staticmethod
    def _extract_completion_progress(state: dict[str, Any] | None) -> object:
        if not isinstance(state, dict):
            return None
        game_state = state.get("game_state")
        if not isinstance(game_state, dict):
            return None
        return game_state.get("completion_progress")

    @staticmethod
    def _extract_game_status(state: dict[str, Any] | None) -> object:
        if not isinstance(state, dict):
            return None
        return state.get("status")

    def _build_evaluation_payload(
        self,
        agent: Agent,
        result: Any,
        game_status: object,
        progress: object,
        game_completion_progress: object,
    ) -> dict[str, Any]:
        return {
            "interaction_id": agent.step_index,
            "step": agent.step_index,
            "max_steps": self.env.config.max_steps,
            "task_status": result.status,
            "game_status": game_status,
            "summary": result.summary,
            "should_stop": result.should_stop,
            "should_reset": result.should_reset,
            "stop_reason": result.stop_reason,
            "finalized": result.finalized,
            "progress": progress,
            "game_completion_progress": game_completion_progress,
            "metrics": dict(agent.eval_metrics),
        }

    async def _evaluate_step(
        self,
        agent: Agent,
        state: dict[str, Any] | None,
        agent_logger: RuntimeLogger | None,
    ):
        if not self.evaluator:
            return None
        if self._evaluation_agent_id and agent.agent_id != self._evaluation_agent_id:
            return None

        result = await self.evaluator.evaluate(agent, state)
        if not result:
            return None
        if result.should_stop and not result.should_reset:
            result = await self.evaluator.summarize(agent, state)

        progress = result.metrics.get("progress") if isinstance(result.metrics, dict) else None
        game_completion_progress = self._extract_completion_progress(state)
        game_status = self._extract_game_status(state)
        evaluation_payload = self._build_evaluation_payload(
            agent,
            result,
            game_status,
            progress,
            game_completion_progress,
        )
        if agent_logger:
            agent_logger.log_task_evaluation(evaluation_payload)

        LOGGER.task(
            "Task eval (%s): step=%s/%s task_status=%s game_status=%s "
            "stop=%s reset=%s finalized=%s reason=%s progress=%s "
            "game_completion_progress=%s metrics=%s",
            agent.agent_id,
            agent.step_index,
            self.env.config.max_steps,
            result.status,
            game_status,
            result.should_stop,
            result.should_reset,
            result.finalized,
            result.stop_reason,
            progress,
            game_completion_progress,
            evaluation_payload["metrics"],
        )
        return result

    async def _handle_eval_controls(self, agent: Agent, result) -> None:
        if result and result.should_reset:
            reset_ok = await self.env.reset_game()
            if reset_ok:
                if self.evaluator:
                    agent.eval_metrics = self.evaluator.reset_metrics(agent.eval_metrics)
                LOGGER.task(
                    "Task eval: auto-reset after fail; continuing at step=%s",
                    agent.step_index,
                )
            else:
                LOGGER.task(
                    "Task eval: auto-reset failed; stopping run at step=%s",
                    agent.step_index,
                )
                self._stop_event.set()
        if result and result.should_stop:
            self._stop_event.set()

    async def _get_raw_action(self, agent: Agent) -> ActionPayload:
        self._update_runtime_diagnosis(agent)
        screenshot_path = await self.env.capture_screenshot(agent.agent_id)

        set_game_state = getattr(agent.client, "set_game_state", None)
        if callable(set_game_state):
            state = self._latest_state_by_agent.get(agent.agent_id)
            if state is None:
                state, _ = await self._capture_state()
                self._latest_state_by_agent[agent.agent_id] = state
            set_game_state(state)

        paused = False
        if self.env.pause_during_inference:
            await self.env.pause_game()
            paused = True

        try:
            return await asyncio.to_thread(agent.client.get_action, screenshot_path)
        finally:
            if paused:
                await self.env.resume_game()

    def _resolve_action(self, agent: Agent, raw_action: ActionPayload) -> ActionPayload:
        if agent.agent_type == "generalist" and agent.semantic_controls_map:
            action = map_semantic_controls_output(raw_action, agent.semantic_controls_map)
            action = self._calibrate_minecraft_resolved_action(agent, raw_action, action)
            LOGGER.model(
                "Agent %s raw action %s -> semantic_controls mapped action: %s",
                agent.agent_id,
                raw_action,
                action,
            )
            return action

        LOGGER.model("Agent %s raw action %s", agent.agent_id, raw_action)
        return raw_action

    async def _run_agent_step(self, agent: Agent, agent_logger: RuntimeLogger | None) -> None:
        raw_action = await self._get_raw_action(agent)
        raw_action = self._maybe_guard_minecraft_action(agent, raw_action)
        self._record_action_choice(agent, raw_action)
        trace = self._log_model_interaction(agent, agent_logger)
        action = self._resolve_action(agent, raw_action)
        action_validity = self._build_action_validity_record(agent, raw_action, action, trace)
        if agent_logger:
            agent_logger.log_action_validity(action_validity)
        await self.env.execute_action(agent, action)
        if agent_logger:
            agent_logger.log_executed_action(action)

        state, state_summary = await self._capture_state()
        self._latest_state_by_agent[agent.agent_id] = state
        if agent_logger:
            agent_logger.log_game_state(state)
        LOGGER.game("Game state: %s", state_summary)

        agent.step_index += 1
        result = await self._evaluate_step(agent, state, agent_logger)
        self._update_minecraft_mining_streak(agent, raw_action, state)
        await self._handle_eval_controls(agent, result)
        if agent_logger:
            agent_logger.finalize_step()

    async def _agent_loop(self, agent: Agent) -> None:
        agent_logger = self.agent_loggers.get(agent.agent_id)
        while not self._stop_event.is_set():
            try:
                await self._run_agent_step(agent, agent_logger)
            except Exception as exc:
                if agent_logger:
                    agent_logger.flush_pending_step()
                LOGGER.exception("Agent %s loop error: %s", agent.agent_id, exc)
                if self._fatal_error is None:
                    self._fatal_error = exc
                self._stop_event.set()
                break

            await asyncio.sleep(AGENT_LOOP_SLEEP_S)

    async def run(self, *, start_env: bool = True) -> None:
        tasks: list[asyncio.Task] = []
        try:
            if start_env:
                await self.env.start()
            tasks = [asyncio.create_task(self._agent_loop(agent)) for agent in self.agents]
            await self._stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for agent_logger in self.agent_loggers.values():
                agent_logger.flush_pending_step()
            await self.env.close_game()

        if self._fatal_error is not None:
            raise RuntimeError("Agent loop failed; run is incomplete and eligible for rerun") from self._fatal_error


__all__ = ["Coordinator"]
