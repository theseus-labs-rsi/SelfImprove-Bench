"""Method discovery and construction, independent of game implementations."""

from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from .interfaces import MethodContext


BUILTIN_METHODS = ("ace", "ahe", "hyperagents")
BUILTIN_ADAPTERS = {
    "ace": "rsi_methods.ACE.ACEAdaptor:ACEAdaptor",
    "ahe": "rsi_methods.AHE.AHEAdaptor:AHEAdaptor",
    "hyperagents": "rsi_methods.HyperAgents.HyperAgentsAdaptor:HyperAgentsAdaptor",
}
RSI_PROVIDERS = (*BUILTIN_METHODS, "rsi")
REQUIRED_HOOKS = ("init", "load", "refine", "export_state", "restore_state", "close")


class MethodRuntimePolicy:
    """Optional method-owned behavior; games and the runner use generic hooks."""

    reload_holdout_agent = False

    def answer(self, response: Any) -> str:
        return response.text

    def reasoning(self, response: Any) -> str | None:
        return response.reasoning

    def is_fatal_error(self, value: Any) -> bool:
        return False

    def normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return result

    def selection_error(self, error: Exception) -> str:
        return str(error)

    def scheduled_refinements(self, count: int, adaptor: Any) -> int:
        return count

    def interrupted_paths(self, adaptor: Any) -> list[Path]:
        return []


DEFAULT_RUNTIME_POLICY = MethodRuntimePolicy()


def runtime_policy(agent: Any) -> MethodRuntimePolicy:
    provider = agent.strip().lower() if isinstance(agent, str) else getattr(agent, "provider", None)
    spec = BUILTIN_ADAPTERS.get(provider) if provider != "rsi" else getattr(agent, "adapter", None)
    if not spec:
        return DEFAULT_RUNTIME_POLICY
    module, name = validate_adapter_spec(spec).split(":")
    factory = getattr(importlib.import_module(module), name)
    return getattr(factory, "runtime_policy", DEFAULT_RUNTIME_POLICY)


def validate_adapter_spec(spec: Any) -> str:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("agent.adapter must be an import path: package.module:AdapterClass")
    module, attribute = spec.split(":")
    if not all(part.isidentifier() for part in module.split(".")) or not attribute.isidentifier():
        raise ValueError("agent.adapter must be an import path: package.module:AdapterClass")
    return spec


def method_identity(agent: Any) -> str:
    """Keep method provenance separate from the common RSI transport."""
    def value(name: str) -> Any:
        return agent.get(name) if isinstance(agent, dict) else getattr(agent, name, None)

    provider = value("provider")
    if provider == "rsi":
        return next((name for name, spec in BUILTIN_ADAPTERS.items()
                     if spec == value("adapter")), "rsi")
    return provider


def create_adaptor(method: str, *, artifact_dir: Path, task_count: int,
                   agent: Any, resuming: bool = False, seed: int = 0) -> Any:
    """Load external code only when starting a run, never when parsing configs."""
    if method in BUILTIN_ADAPTERS:
        spec = BUILTIN_ADAPTERS[method]
    elif method == "rsi":
        spec = validate_adapter_spec(getattr(agent, "adapter", None))
    else:
        raise ValueError(f"Unsupported RSI method: {method}")
    module, name = spec.split(":")
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"Cannot load RSI adapter {spec}: {exc}") from exc
    if not callable(factory):
        raise TypeError(f"RSI adapter {spec} must be callable")
    adaptor = factory(MethodContext(
        artifact_dir=artifact_dir, task_count=task_count,
        options=deepcopy(getattr(agent, "adapter_options", {})),
        model=getattr(agent, "model", None),
        timeout=float(getattr(agent, "timeout", 300.0)), resuming=resuming,
        seed=seed,
    ))
    missing = [name for name in REQUIRED_HOOKS if not callable(getattr(adaptor, name, None))]
    if missing:
        raise TypeError(f"RSI adapter {spec} is missing hooks: {', '.join(missing)}")
    return adaptor
