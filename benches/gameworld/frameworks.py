"""Framework registry for the open-source GameWorld integration.

The built-in frameworks and external RSI adapters are independent from the model
used by the framework. Model and credential names are supplied by the caller through
environment variables; this module contains no provider key or model secret.
"""

from __future__ import annotations

import os
from typing import Any

from benchmark.methods import RSI_PROVIDERS


SUPPORTED_FRAMEWORKS = (*RSI_PROVIDERS, "claude")
CLIENT_MODULES = {name: "benches.gameworld.rsi_game_client" for name in RSI_PROVIDERS}
CLIENT_CLASSES = {name: "RSIGameClient" for name in RSI_PROVIDERS}
CLIENT_MODULES["claude"] = "agents.mm_agents.claude_code"
CLIENT_CLASSES["claude"] = "ClaudeCodeAgent"
ADAPTOR_IMPORTS = {
    "ace": "rsi_methods.ACE.ACEAdaptor.ACEAdaptor",
    "hyperagents": "rsi_methods.HyperAgents.HyperAgentsAdaptor.HyperAgentsAdaptor",
    "ahe": "rsi_methods.AHE.AHEAdaptor.AHEAdaptor",
}


def normalize_framework(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"Unsupported framework {value!r}; choose one of {SUPPORTED_FRAMEWORKS}"
        )
    return normalized


def client_import_spec(framework: str) -> tuple[str, str]:
    normalized = normalize_framework(framework)
    return CLIENT_MODULES[normalized], CLIENT_CLASSES[normalized]


def model_config_from_environment(
    *,
    model_env: str = "GAMEWORLD_RSI_MODEL",
    base_url_env: str = "OPENAI_BASE_URL",
    api_key_env: str = "OPENAI_API_KEY",
) -> dict[str, Any]:
    """Resolve model routing without embedding credentials in source code."""

    model = os.environ.get(model_env, "").strip()
    if not model:
        raise RuntimeError(f"Set {model_env} before starting a model-backed run")
    return {
        "model": model,
        "base_url_env": base_url_env,
        "api_key_env": api_key_env,
    }


__all__ = [
    "ADAPTOR_IMPORTS",
    "CLIENT_CLASSES",
    "CLIENT_MODULES",
    "SUPPORTED_FRAMEWORKS",
    "client_import_spec",
    "model_config_from_environment",
    "normalize_framework",
]
