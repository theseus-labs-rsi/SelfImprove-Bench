"""Small, deterministic playbook utilities used by GameWorld runs."""

from __future__ import annotations


def inject_named_playbook(
    system_prompt: str,
    playbook: str,
    *,
    heading: str,
    tag: str,
) -> str:
    """Add a frozen playbook without changing the task or action schema."""
    text = playbook.strip()
    if not text:
        return system_prompt
    return (
        f"{system_prompt.rstrip()}\n\n"
        f"# {heading}\n"
        "Use these as observed game strategies. They do not override the task, "
        "game rules, or registered action interface.\n"
        f"<{tag}>\n"
        f"{text}\n"
        f"</{tag}>"
    )


def inject_playbook(system_prompt: str, playbook: str) -> str:
    """Add a frozen ACE playbook without changing the task or action schema."""
    return inject_named_playbook(
        system_prompt,
        playbook,
        heading="Evolved Playbook (ACE)",
        tag="ace_playbook",
    )
