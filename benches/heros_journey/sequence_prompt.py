"""Prompt for Hero's Journey's one-call action-sequence protocol."""

from __future__ import annotations


ACTION_SEQUENCE_PROTOCOL = "action_sequence_v1"


def build_sequence_prompt(
    demo_context: str,
    initial_observation: str,
    *,
    action_limit: int,
    no_reasoning: bool = False,
) -> str:
    """Ask for one complete plan; the environment supplies observations later."""
    if action_limit <= 0:
        raise ValueError("action_limit must be positive")

    action_shape = (
        '{"action": "<name>", "argument": "<value>"}'
        if no_reasoning else
        '{"action": "<name>", "argument": "<value>", "reasoning": "<brief reason>"}'
    )
    return f"""You are playing an adventure game. Study the demonstration episodes and the task below.

Available actions:
  go [location]       — travel to a location
  buy [object]        — purchase an item at the current location
  get [object]        — pick up a free item at the current location
  defeat [enemy]      — defeat an enemy at their location
  rescue [npc]        — rescue an NPC at their location
  perform [ritual]    — perform a ritual
  drink [potion]      — drink a potion
  check_inventory     — list your current items
  check_location      — show your current location

The rules shown for your task may be incomplete. Infer missing requirements from the demonstrations.
Submit your complete ordered action sequence in ONE response. You will not receive observations while planning.
Your action budget is {action_limit} steps. Submit at most {action_limit} actions; only the first {action_limit} will be executed.
The environment executes actions in order and records the real observation after each step. A failed action counts toward the budget, and later actions still execute in order. Execution stops when the goal is reached, the budget is exhausted, or your sequence ends.
Do not invent observations or include conditional branches in the action sequence.

Output ONLY a JSON object with this structure:
{{"actions": [{action_shape}, {action_shape}]}}

{demo_context}

{initial_observation}
"""
