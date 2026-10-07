"""Fixed capability boundaries for evolvable HyperDSH candidates."""

SAFE_OPTIONAL_TOOLS = frozenset(
    {
        "tool-fs",
        "tool-fs-search",
        "tool-todo",
        "tool-goal",
        "tool-result-pruner",
        "tool-skill",
    }
)

ALL_GATED_TOOLS = (
    "tool-bash",
    "tool-pwsh",
    "tool-jobs",
    "tool-fs",
    "tool-fs-search",
    "tool-skill",
    "tool-subagent-control",
    "tool-subagent-list-agents",
    "tool-subagent",
    "tool-subagent-fork",
    "tool-workflow",
    "tool-result-pruner",
    "tool-todo",
    "tool-goal",
    "tool-ralph",
    "tool-web",
)

# These rows are owned by the runtime overlay or enforce the process boundary.
PROTECTED_PROFILE_IDS = frozenset(
    {
        "agent-default-model",
        "approval",
        "credentials",
        "fs-sandbox",
        "headless-runner",
        "headless-startup",
        "llm-deepseek",
        "llm-pi-ai",
        "permission",
        "sandbox",
        "sandbox-policy",
        "session-telemetry-otel",
        "settings",
        "system-prompt",
        "tools",
    }
)

__all__ = ["ALL_GATED_TOOLS", "PROTECTED_PROFILE_IDS", "SAFE_OPTIONAL_TOOLS"]
