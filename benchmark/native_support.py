"""Transport and rollout services for the bundled native method controllers.

No game scoring policy lives here. Native controllers own proposal schedules and
admission; the runner supplies isolated, repeatable executions.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .contracts import to_jsonable
from .method_support import StructuredStateAdapter, _json_text, _parse_json_object, _usage_report
from .resume_state import atomic_json

_IMPORT_LOCK = threading.RLock()


def native_package(name: str, root: Path) -> None:
    """Load bundled packages with absolute imports, without changing sys.path."""
    with _IMPORT_LOCK:
        existing = sys.modules.get(name)
        if existing is not None:
            if Path(existing.__file__).resolve() != (root / "__init__.py").resolve():
                raise ImportError(f"{name} already loaded from a different installation")
            return
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(str(root))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise


def write_files(root: Path, files: Mapping[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = root / name
        if not name or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"artifact path escapes its root: {name}")
        if not isinstance(content, str):
            raise ValueError(f"artifact must contain text: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


class NativeAdapter(StructuredStateAdapter):
    """Shared *transport* only; subclasses implement the native algorithm."""

    def __init__(self, context):
        super().__init__(context)
        if int(self.options.get("actor_tool_steps", 12)) < 1:
            raise ValueError("actor_tool_steps must be positive")
        self._evaluate_method = None
        self._calls_lock = threading.RLock()
        self._calls: list[dict[str, Any]] = []
        self._work_dir: Path | None = None

    def bind_evaluator(self, evaluator) -> None:
        self._evaluate_method = evaluator

    def start_refinement(self, feedback, output_dir):
        if self._evaluate_method is None:
            raise RuntimeError(f"{self.method_name} requires the runner's bind_evaluator service")
        self._work_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._calls = []
        atomic_json(output_dir / "method_feedback.json", to_jsonable(feedback))
        self.last_refinement_report = {"method": self.method_name, "stages": self._calls}
        self._update_usage()

    def _update_usage(self):
        with self._calls_lock:
            fields = ("prompt_tokens", "answer_tokens", "reasoning_tokens", "total_tokens")
            usage = {}
            for field in fields:
                values = [c["usage"].get(field) for c in self._calls]
                usage[field] = sum(values) if all(isinstance(v, int) for v in values) else None
            usage["complete"] = usage["total_tokens"] is not None
            usage["no_model_calls"] = not self._calls
            self.last_refinement_report["token_usage_report"] = usage
            if self._work_dir:
                atomic_json(self._work_dir / "refinement_report.json", self.last_refinement_report)

    def ask_text(self, prompt: str, *, stage: str) -> str:
        try:
            response = self.dispatcher.distribute(
                "openai", prompt, model=self.optimizer_model, api_mode=self.api_mode,
                max_output_tokens=self.optimizer_max_output_tokens,
                reasoning_effort=self.optimizer_reasoning_effort,
                temperature=float(self.temperature) if self.temperature is not None else None,
            )
        except Exception:
            self._record_unknown_usage(stage)
            raise
        with self._calls_lock:
            self._calls.append({"stage": stage, "usage": _usage_report(response.usage)})
            if self._work_dir:
                atomic_json(self._work_dir / "model_calls" / f"{len(self._calls):04d}.json", {
                    "stage": stage, "prompt": prompt, "response": to_jsonable(response),
                })
            self._update_usage()
        return response.text

    def ask(self, prompt: str, *, stage: str) -> dict:
        return _parse_json_object(self.ask_text(prompt, stage=stage))

    def propose(self, state, evidence, *, stage, instruction=""):
        return self._validate_state(self.normalize_candidate(self.ask(
            self.refinement_instructions(evidence) + "\n" + instruction
            + "\nCurrent artifact:\n" + _json_text(state)
            + "\nExecution evidence:\n" + _json_text(evidence)
            + "\nReturn JSON only.", stage=stage), current=state))

    def rollout(self, state, *, label, split="train", index=0):
        agent = self._agent(deepcopy(state))
        try:
            result = self._evaluate_method(agent, label=label, split=split, index=index,
                                           output_dir=self._work_dir / "rollouts" / label)
        except Exception:
            self._record_unknown_usage(label)
            raise
        metrics = result["record"].get("llm_metrics", {}).get("tokens", {})
        result["mechanism_fingerprint"] = getattr(agent, "fingerprint", {})
        total = metrics.get("total_tokens")
        with self._calls_lock:
            self._calls.append({"stage": label, "kind": "rollout", "usage": {
                "total_tokens": total, "prompt_tokens": metrics.get("prompt_tokens"),
                "answer_tokens": metrics.get("answer_tokens"),
                "reasoning_tokens": metrics.get("reasoning_tokens"),
            }})
            self._update_usage()
        if result.get("provider_failure"):
            raise RuntimeError(f"native rollout provider failure: {result['record'].get('error')}")
        return result

    def _record_unknown_usage(self, stage):
        # A failed request may already have incurred tokens. Never report it as
        # zero usage or claim no_model_calls after a provider/runtime exception.
        with self._calls_lock:
            self._calls.append({"stage": stage, "failed": True, "usage": _usage_report(None)})
            self._update_usage()

    def finish_refinement(self, state, *, accepted, detail):
        report = {"method": self.method_name, "accepted": bool(accepted),
                  "version_before": self._version, **to_jsonable(detail)}
        if accepted:
            self._state = self._validate_state(deepcopy(state))
            self._version += 1
        report["version_after"] = self._version
        self._history.append(report)
        self.last_selection_report = report
        self.last_refinement_report["selection"] = report
        self._update_usage()
        self._persist_state()
        atomic_json(self._work_dir / "selection.json", report)


def valid_score(result):
    record = result["record"]
    score = record.get("relative_score")
    if record.get("status") != "ok" or not isinstance(score, (float, int)):
        raise RuntimeError(f"native method rollout failed: {record.get('error') or record.get('status')}")
    return float(score)


__all__ = ["NativeAdapter", "native_package", "write_files", "valid_score"]
