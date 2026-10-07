"""Codex Exec agent implementation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .base.base_client import BaseClientConfig
from .base.generalist_agent import GeneralistAgent
from ..harness.memory import memory_entries_to_image_paths, memory_entries_to_text


LOGGER = logging.getLogger(__name__)


@dataclass
class CodexExecConfig(BaseClientConfig):
    """Configuration for Codex Exec."""

    model: str = "gpt-5.6-sol"
    model_type: str = "generalist"
    api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    codex_bin: str = "codex"
    sandbox: str = "read-only"
    reasoning_effort: str = "medium"
    timeout_retries: int = 2
    retry_backoff_sec: float = 2.0


class CodexExecAgent(GeneralistAgent):
    """Codex CLI backend that returns one semantic action per step."""

    def __init__(self, config: BaseClientConfig, **shared_tools):
        super().__init__(config, **shared_tools)
        self._model_name = config.model or "gpt-5.6-sol"
        self._codex_bin = getattr(config, "codex_bin", "codex")
        self._sandbox = getattr(config, "sandbox", "read-only")
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            auth_path = Path("/root/.codex/auth.json")
            if auth_path.is_file():
                try:
                    auth_payload = json.loads(auth_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    auth_payload = {}
                api_key = auth_payload.get("OPENAI_API_KEY")
        self._api_key = self._resolve_api_key(api_key, env_vars=("OPENAI_API_KEY",))

    def _build_prompt_text(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[object],
    ) -> str:
        memory_text = memory_entries_to_text(memory_entries) or ""
        sections: list[str] = []
        if system_prompt and system_prompt.strip():
            sections.append(system_prompt.strip())
        if memory_text:
            sections.append("# Memory\n" + memory_text.strip())
        if user_prompt and user_prompt.strip():
            sections.append(user_prompt.strip())
        sections.append(
            "Do not run shell commands. Return exactly one JSON object with "
            "`tool_name` and `arguments.reasoning`."
        )
        return "\n\n".join(sections).strip()

    @staticmethod
    def _build_schema(action_names: list[str]) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "enum": action_names,
                },
                "arguments": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                        }
                    },
                    "required": ["reasoning"],
                    "additionalProperties": False,
                },
            },
            "required": ["tool_name", "arguments"],
            "additionalProperties": False,
        }

    def build_request_payload(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[object],
        tools: list[dict[str, object]],
        screenshot_path: Path,
    ) -> dict[str, object]:
        del tools
        prompt = self._build_prompt_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            memory_entries=memory_entries,
        )
        image_paths = [
            str(path)
            for path in [
                *memory_entries_to_image_paths(memory_entries),
                screenshot_path,
            ]
            if path and Path(path).exists()
        ]
        schema_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="codex_action_schema_",
            delete=False,
        )
        json.dump(self._build_schema(sorted(self._action_tool_names)), schema_file, indent=2)
        schema_file.flush()
        schema_file.close()
        return {
            "prompt": prompt,
            "images": image_paths,
            "schema_path": schema_file.name,
        }

    def send_request(self, request_payload: dict[str, object]) -> object:
        prompt = str(request_payload.get("prompt") or "")
        image_paths = [str(path) for path in request_payload.get("images") or [] if str(path).strip()]
        schema_path = str(request_payload.get("schema_path") or "")
        output_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="codex_last_message_",
            delete=False,
        )
        output_file.close()

        codex_home = tempfile.mkdtemp(prefix="codex_home_")
        try:
            env = dict(os.environ)
            env["OPENAI_API_KEY"] = self._api_key
            auth_path = Path(codex_home) / "auth.json"
            auth_path.write_text(
                json.dumps({"OPENAI_API_KEY": self._api_key}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            endpoint = getattr(self.config, "endpoint", None) or "https://free-router.opendatalab.com/v1"
            config_path = Path(codex_home) / "config.toml"
            reasoning_effort = str(getattr(self.config, "reasoning_effort", "medium")).strip().lower()
            if reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
                raise ValueError(f"Unsupported Codex reasoning effort: {reasoning_effort!r}")
            config_path.write_text(
                "model_provider = \"gameworld\"\n"
                f"model = \"{self._model_name}\"\n"
                f"model_reasoning_effort = \"{reasoning_effort}\"\n\n"
                "[model_providers.gameworld]\n"
                "name = \"gameworld\"\n"
                f"base_url = \"{endpoint.rstrip('/') }\"\n"
                "wire_api = \"responses\"\n"
                "requires_openai_auth = true\n",
                encoding="utf-8",
            )
            env["CODEX_HOME"] = codex_home

            cmd = [
                self._codex_bin,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                self._sandbox,
                "--model",
                self._model_name,
                "--output-schema",
                schema_path,
                "--output-last-message",
                output_file.name,
            ]
            for image_path in image_paths:
                cmd.extend(["-i", image_path])
            cmd.append("-")
            timeout_retries = max(0, int(getattr(self.config, "timeout_retries", 2)))
            retry_backoff_sec = max(0.0, float(getattr(self.config, "retry_backoff_sec", 2.0)))
            for attempt in range(timeout_retries + 1):
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(Path.cwd()),
                        env=env,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        timeout=self.config.request_timeout,
                        check=False,
                    )
                    break
                except subprocess.TimeoutExpired:
                    if attempt >= timeout_retries:
                        raise
                    LOGGER.warning(
                        "Codex exec timed out after %.1fs; retrying request (%d/%d)",
                        self.config.request_timeout,
                        attempt + 1,
                        timeout_retries,
                    )
                    Path(output_file.name).write_text("", encoding="utf-8")
                    if retry_backoff_sec:
                        time.sleep(retry_backoff_sec)
            last_message = Path(output_file.name).read_text(encoding="utf-8", errors="replace")
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "last_message": last_message,
            }
        finally:
            Path(schema_path).unlink(missing_ok=True)
            Path(output_file.name).unlink(missing_ok=True)
            shutil.rmtree(codex_home, ignore_errors=True)

    def _parse_action_payload(self, text: str) -> dict[str, object] | None:
        if not text:
            return None
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            start = text.find("{", idx)
            if start < 0:
                break
            try:
                payload, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                idx = start + 1
                continue
            if isinstance(payload, dict):
                tool_name = payload.get("tool_name")
                arguments = payload.get("arguments")
                if isinstance(tool_name, str) and tool_name.strip():
                    return {
                        "tool_name": tool_name.strip(),
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    }
            idx = start + max(1, end)
        return None

    def extract_tool_call(self, response: object) -> dict[str, object] | None:
        if not isinstance(response, dict):
            return None
        last_message = response.get("last_message")
        if isinstance(last_message, str):
            parsed = self._parse_action_payload(last_message)
            if parsed:
                return parsed
        stdout = response.get("stdout")
        if isinstance(stdout, str):
            parsed = self._parse_action_payload(stdout)
            if parsed:
                return parsed
        stderr = response.get("stderr")
        if isinstance(stderr, str):
            parsed = self._parse_action_payload(stderr)
            if parsed:
                return parsed
        return None

    def extract_reasoning(self, response: object) -> str | None:
        if not isinstance(response, dict):
            return None
        last_message = response.get("last_message")
        if not isinstance(last_message, str):
            return None
        parsed = self._parse_action_payload(last_message)
        if not parsed:
            return None
        arguments = parsed.get("arguments")
        if isinstance(arguments, dict):
            reasoning = arguments.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
        return None

    def extract_error(self, response: object) -> str | None:
        if not isinstance(response, dict):
            return None
        returncode = response.get("returncode")
        stderr = str(response.get("stderr") or "").strip()
        if returncode not in (0, None):
            return f"Codex exec failed with returncode={returncode}: {stderr or 'no stderr'}"
        return None


__all__ = ["CodexExecAgent", "CodexExecConfig"]
