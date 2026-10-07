"""Benchmark-facing DeepSeek Harness headless execution bridge."""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable
from uuid import uuid4

from benchmark.contracts import BackendResponse, TokenUsage
from .constants import ALL_GATED_TOOLS, SAFE_OPTIONAL_TOOLS
from .versioning import CandidateSnapshot, materialize_snapshot, validate_harness

@dataclass(frozen=True, slots=True)
class DSHRuntimeConfig:
    command: str
    profile: str
    timeout_seconds: float
    telemetry_disabled: bool
    provider: str
    model: str
    reasoning_effort: str | None
    max_output_tokens: int
    api: str
    base_url: str
    api_key_env: str
    api_key: str | None = None
    base_url_env: str = "OPENAI_BASE_URL"
    max_signal_retries: int = 1


class HyperDSHAgent:
    """Run one exact versioned harness through an isolated DSH home."""

    def __init__(
        self,
        snapshot: CandidateSnapshot,
        *,
        runtime_root: str | Path,
        artifact_dir: str | Path,
        config: DSHRuntimeConfig,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.snapshot = snapshot
        self.runtime_root = Path(runtime_root).resolve()
        self.artifact_dir = Path(artifact_dir).resolve()
        self.config = config
        self.runner = runner
        self._call_index = 0
        if (
            isinstance(self.config.max_signal_retries, bool)
            or not isinstance(self.config.max_signal_retries, int)
            or not 0 <= self.config.max_signal_retries <= 3
        ):
            raise ValueError("HyperDSH max_signal_retries must be an integer from 0 to 3")
        self._base_url = self._configured_base_url()
        self.candidate_dir = materialize_snapshot(
            snapshot, self.runtime_root / "candidate"
        )
        self.harness_dir = self.candidate_dir / "harness"
        self.tools = validate_harness(self.harness_dir)
        self.dsh_home = self.runtime_root / "dsh_home"
        self.workspace = self.runtime_root / "task_workspace"
        self._prepare_runtime_home()

    def act(self, prompt: str) -> BackendResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("HyperDSH prompt must be non-empty text")
        executable = shutil.which(self.config.command)
        if executable is None:
            raise RuntimeError(
                f"DeepSeek Harness command {self.config.command!r} was not found on PATH"
            )
        api_key = self.config.api_key or os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                "HyperDSH requires task_provider.api_key or credential "
                f"environment variable {self.config.api_key_env}"
            )

        self._call_index += 1
        call_id = f"call_{self._call_index:04d}_{uuid4().hex[:8]}"
        call_dir = self.artifact_dir / call_id
        call_dir.mkdir(parents=True, exist_ok=False)
        overlay = call_dir / "runtime.patch.yml"
        overlay.write_text(self._runtime_overlay(prompt), encoding="utf-8")
        environment = os.environ.copy()
        # DSH's provider schema references an environment-variable name. A
        # directly configured key is injected only into the child process and
        # is never serialized into settings or invocation artifacts.
        environment[self.config.api_key_env] = api_key
        environment["DSH_HOME"] = str(self.dsh_home)
        environment["DSH_PERMISSION_MODE"] = "read-only"
        if self.config.telemetry_disabled:
            environment["DSH_TELEMETRY_DISABLED"] = "1"
        command = [
            executable,
            "--profile",
            self.config.profile,
            "--patch",
            str(overlay),
            call_id,
        ]
        process_attempts: list[dict[str, object]] = []
        completed: subprocess.CompletedProcess[str] | None = None
        stdout = ""
        stderr = ""
        for process_attempt in range(1, self.config.max_signal_retries + 2):
            try:
                completed = self.runner(
                    command,
                    cwd=self.workspace,
                    env=environment,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                    # Detach DSH from the benchmark's SSH/terminal session so
                    # a controlling-terminal hangup is not propagated.
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"HyperDSH task timed out after {self.config.timeout_seconds:g}s"
                ) from exc
            except OSError as exc:
                raise RuntimeError(f"HyperDSH task process failed: {exc}") from exc

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            signal_name = _termination_signal(completed.returncode)
            will_retry = (
                signal_name is not None
                and process_attempt <= self.config.max_signal_retries
            )
            attempt_record: dict[str, object] = {
                "process_attempt": process_attempt,
                "returncode": completed.returncode,
                "signal": signal_name,
                "will_retry": will_retry,
            }
            process_attempts.append(attempt_record)
            attempt_dir = call_dir / f"process_attempt_{process_attempt:02d}"
            attempt_dir.mkdir()
            (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
            (attempt_dir / "invocation.json").write_text(
                json.dumps(attempt_record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if not will_retry:
                break

        if completed is None:
            raise AssertionError("HyperDSH process loop produced no result")
        (call_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (call_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        (call_dir / "invocation.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate_id": str(self.snapshot.candidate_id),
                    "profile": self.config.profile,
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "returncode": completed.returncode,
                    "signal": _termination_signal(completed.returncode),
                    "process_attempt_count": len(process_attempts),
                    "max_signal_retries": self.config.max_signal_retries,
                    "process_attempts": process_attempts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            detail = (stderr or stdout).strip()
            signal_name = _termination_signal(completed.returncode)
            if signal_name is not None:
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"DeepSeek Harness terminated by signal {signal_name} "
                    f"(return code {completed.returncode}) after "
                    f"{len(process_attempts)} process attempts{suffix}"
                )
            raise RuntimeError(
                f"DeepSeek Harness exited with code {completed.returncode}: {detail}"
            )
        answer = stdout.strip()
        if not answer:
            raise RuntimeError("DeepSeek Harness returned an empty answer")
        return BackendResponse(
            text=answer,
            reasoning=_reasoning(stderr),
            usage=_estimated_usage(prompt, answer),
            provider=self.config.provider,
            model=self.config.model,
            status="completed",
            finish_reason="stop",
            metadata={
                "hyperdsh": {
                    "candidate_id": str(self.snapshot.candidate_id),
                        "call_id": call_id,
                        "artifact_dir": str(call_dir),
                        "process_attempt_count": len(process_attempts),
                    }
            },
        )

    def close(self) -> None:
        """Headless calls leave no child process; artifacts remain reproducible."""

    def _prepare_runtime_home(self) -> None:
        profile_target = self.dsh_home / "profiles" / self.config.profile
        profile_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.harness_dir / "profile", profile_target)
        skills = self.harness_dir / "skills"
        if skills.is_dir():
            shutil.copytree(skills, self.dsh_home / "skills")
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.dsh_home / "settings.yaml").write_text(
            self._settings_yaml(), encoding="utf-8"
        )

    def _settings_yaml(self) -> str:
        provider = self.config.provider
        return (
            "llm-pi-ai:\n"
            "  providers:\n"
            f"    {provider}:\n"
            f"      apiKeyEnv: {json.dumps(self.config.api_key_env)}\n"
            f"      api: {json.dumps(self.config.api)}\n"
            f"      baseURL: {json.dumps(self._base_url)}\n"
            "      models:\n"
            f"        - id: {json.dumps(self.config.model)}\n"
            f"          maxTokens: {self.config.max_output_tokens}\n"
            "agent-default-model:\n"
            f"  provider: {json.dumps(provider)}\n"
            f"  model: {json.dumps(self.config.model)}\n"
        )

    def _configured_base_url(self) -> str:
        base_url = self.config.base_url.strip()
        if not base_url:
            base_url = os.environ.get(self.config.base_url_env, "").strip()
        if not base_url:
            raise RuntimeError(
                "HyperDSH requires task_provider.base_url or environment "
                f"variable {self.config.base_url_env}"
            )
        return base_url

    def _runtime_overlay(self, task: str) -> str:
        prompt = self._composed_prompt()
        enabled = set(self.tools["enabled"])
        unsupported = enabled - SAFE_OPTIONAL_TOOLS
        if unsupported:
            raise ValueError(
                f"HyperDSH tools.json enables unsupported tools: {sorted(unsupported)}"
            )
        lines = [
            "- id: sandbox-policy",
            "  config:",
            "    mode: read-only",
            "- id: approval",
            "  config:",
            "    policy: never",
            "- id: permission",
            "  config:",
            "    presets:",
            "      hyperdsh-read-only:",
            "        sandbox: read-only",
            "        approval: never",
            "        name: HyperDSH read-only",
            "        description: Read-only execution without interactive approval.",
            "    defaultPreset: hyperdsh-read-only",
            "- id: system-prompt",
            "  config:",
            f"    personaPrefix: {json.dumps(prompt, ensure_ascii=False)}",
            "    personaSuffix: Your working directory is {{cwd}}.",
            "- id: agent-default-model",
            "  config:",
            f"    provider: {json.dumps(self.config.provider)}",
            f"    model: {json.dumps(self.config.model)}",
        ]
        if self.config.reasoning_effort:
            lines.append(
                f"    reasoningEffort: {json.dumps(self.config.reasoning_effort)}"
            )
        for tool_id in ALL_GATED_TOOLS:
            lines.extend(
                [f"- id: {tool_id}", f"  disabled: {'false' if tool_id in enabled else 'true'}"]
            )
        lines.extend(
            [
                "- id: headless-runner",
                "  config:",
                f"    task: {json.dumps(task, ensure_ascii=False)}",
            ]
        )
        return "\n".join(lines) + "\n"

    def _composed_prompt(self) -> str:
        sections = [(self.harness_dir / "prompt.md").read_text(encoding="utf-8").strip()]
        skill_root = self.harness_dir / "skills"
        for skill in sorted(skill_root.rglob("*.md")) if skill_root.is_dir() else ():
            content = skill.read_text(encoding="utf-8").strip()
            if content:
                sections.append(f"Skill: {skill.relative_to(skill_root).as_posix()}\n{content}")
        prompt = "\n\n".join(sections)
        if len(prompt.encode("utf-8")) > 65536:
            raise ValueError("HyperDSH composed prompt and skills exceed 65536 bytes")
        return prompt


def _estimated_usage(prompt: str, answer: str) -> TokenUsage:
    def estimate(text: str) -> int:
        cjk = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
        non_cjk = re.sub(r"[\u3400-\u9fff\uf900-\ufaff\s]", "", text)
        return cjk + ceil(len(non_cjk) / 4)

    input_tokens = estimate(prompt)
    output_tokens = estimate(answer)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _reasoning(stderr: str) -> str | None:
    content = stderr.strip()
    return content or None


def _termination_signal(returncode: int) -> str | None:
    if returncode >= 0:
        return None
    signal_number = -returncode
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"SIGNAL_{signal_number}"


__all__ = ["DSHRuntimeConfig", "HyperDSHAgent", "SAFE_OPTIONAL_TOOLS"]
