"""Preflight checks for paired backend-comparison suites."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
from typing import Any

from agents.factory import get_config_for_model
from catalog import build_runtime_config, load_model

from .spec import SuiteSpec


MODEL_FIELDS = (
    "model",
    "reasoning_effort",
    "endpoint",
    "request_timeout",
    "enable_memory",
    "memory_rounds",
    "memory_format",
)
RUNTIME_FIELDS = (
    "game_id",
    "task_id",
    "task_prompt",
    "evaluator_id",
    "evaluator_config",
    "task_start_score_field",
    "task_target_score_field",
    "max_steps",
    "pause_during_inference",
    "continue_on_fail",
    "memory_rounds",
    "memory_format",
    "semantic_controls_specs",
    "system_prompts",
)


def _effective_model_config(model_id: str) -> Any:
    profile = load_model(model_id)
    config = get_config_for_model(profile.model_name)
    for key, value in profile.config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def _openai_credential_available(configs: dict[str, Any]) -> bool:
    fallback_key = os.environ.get("OPENAI_API_KEY")
    auth_path = Path("/root/.codex/auth.json")
    if not fallback_key and auth_path.is_file():
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        fallback_key = payload.get("OPENAI_API_KEY")
    return all(bool(getattr(config, "api_key", None) or fallback_key) for config in configs.values())


def _uses_backend(model_spec: str, backend: str) -> bool:
    """Return whether every game role uses the requested backend."""
    parts = [part.strip() for part in model_spec.split(",") if part.strip()]
    return bool(parts) and all(part == backend for part in parts)


def validate_comparison_contract(suite: SuiteSpec) -> None:
    """Reject a paired suite when controlled experiment settings differ."""
    contract = suite.config.get("comparison_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError("comparison_contract must be a mapping")

    baseline = str(contract.get("baseline_model") or "").strip()
    candidate = str(contract.get("candidate_model") or "").strip()
    if not baseline or not candidate or baseline == candidate:
        raise ValueError("comparison_contract requires distinct baseline_model and candidate_model")

    configs = {
        baseline: _effective_model_config(baseline),
        candidate: _effective_model_config(candidate),
    }
    mismatches = []
    if not _openai_credential_available(configs):
        mismatches.append("OPENAI_API_KEY is unavailable to the paired backends")
    codex_bin = str(getattr(configs[candidate], "codex_bin", "") or "")
    if codex_bin and shutil.which(codex_bin) is None:
        mismatches.append(f"candidate codex executable not found: {codex_bin!r}")
    for field in MODEL_FIELDS:
        values = [getattr(configs[model], field, None) for model in (baseline, candidate)]
        if values[0] != values[1]:
            mismatches.append(f"model.{field}: {baseline}={values[0]!r}, {candidate}={values[1]!r}")

    grouped: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for run in suite.runs:
        grouped[(str(run["game_id"]), str(run["task_id"]), int(run["repeat_index"]))].add(
            str(run["model_spec"])
        )

    for (game_id, task_id, repeat_index), model_specs in grouped.items():
        baseline_specs = [spec for spec in model_specs if _uses_backend(spec, baseline)]
        candidate_specs = [spec for spec in model_specs if _uses_backend(spec, candidate)]
        if len(baseline_specs) != 1 or len(candidate_specs) != 1:
            mismatches.append(
                f"{game_id}+{task_id} repeat={repeat_index} must include both comparison models"
            )
            continue
        baseline_runtime = build_runtime_config(f"{game_id}+{task_id}+{baseline_specs[0]}")
        candidate_runtime = build_runtime_config(f"{game_id}+{task_id}+{candidate_specs[0]}")
        for field in RUNTIME_FIELDS:
            left = getattr(baseline_runtime, field)
            right = getattr(candidate_runtime, field)
            if left != right:
                mismatches.append(f"{game_id}+{task_id}.{field} differs")

    if mismatches:
        raise ValueError("Comparison preflight failed:\n- " + "\n- ".join(mismatches))

    print(
        "Comparison preflight passed: "
        f"{baseline} vs {candidate}; paired cases={len(grouped)}; "
        f"controlled model fields={len(MODEL_FIELDS)}; runtime fields={len(RUNTIME_FIELDS)}"
    )


__all__ = ["validate_comparison_contract"]
