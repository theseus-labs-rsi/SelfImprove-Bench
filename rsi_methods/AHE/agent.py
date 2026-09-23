"""AHE coding agent with one NexAU runtime per benchmark agent object.

The copied harness is fixed when this object is built.  A dedicated process
keeps candidate-specific ``tools`` imports separate, while retaining one
NexAU Agent across calls to ``act`` on this benchmark task.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from benchmark.contracts import BackendResponse


class AHEAgent:
    """Execute one harness snapshot and retain traces for ``refine()``.

    The child process starts with this object.  It loads ``AgentConfig`` and
    constructs the NexAU Agent once, so repeated ``act`` calls use the same
    harness and sandbox.  A later ``build()`` creates a new object for a new
    candidate or selected generation.
    """

    def __init__(
        self,
        workspace: Path,
        artifact_dir: Path,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 300,
    ) -> None:
        self.workspace = workspace.resolve()
        self.config_path = self.workspace / "code_agent.yaml"
        self.artifact_dir = artifact_dir.resolve()
        self.sandbox_dir = self.artifact_dir / "sandbox"
        self.upstream_root = Path(__file__).resolve().parent / "agentic-harness-engineering"
        self.model = model or os.environ.get("LLM_MODEL")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.trace_paths: list[Path] = []
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self.sandbox_dir.mkdir()
        parent, child = multiprocessing.Pipe()
        self._connection = parent
        self._process = multiprocessing.Process(
            target=AHEAgent._worker,
            args=(child, self.workspace, self.config_path, self.sandbox_dir,
                  self.upstream_root, self.artifact_dir / "agent.log", self.model,
                  self.reasoning_effort),
            daemon=True,
        )
        try:
            self._process.start()
            child.close()
            ready = self._receive(max(60.0, timeout_seconds))
            if not ready.get("ready"):
                raise RuntimeError(f"AHE Agent initialization failed: {ready.get('error')}; see {self.artifact_dir / 'agent.log'}")
        except BaseException:
            child.close()
            self.close()
            raise

    def act(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] | None = None,
    ) -> BackendResponse:
        """Run one turn; the newest trace includes this Agent's prior turns.

        Upstream ``start.py`` only demonstrates text input.  Image paths are
        rejected until NexAU's multimodal message shape is established.
        """

        if not isinstance(prompt, str):
            raise TypeError("AHEAgent.act() requires a text prompt")
        if image_paths:
            raise NotImplementedError("AHE image input needs a verified NexAU mapping")
        call_dir = self.artifact_dir / uuid4().hex
        call_dir.mkdir()
        (call_dir / "request.json").write_text(
            json.dumps({"prompt": prompt}, ensure_ascii=False), encoding="utf-8",
        )
        self._connection.send({"op": "act", "prompt": prompt, "call_dir": str(call_dir)})
        result = self._receive(self.timeout_seconds)
        if not result.get("ok"):
            raise RuntimeError(f"AHE act failed: {result.get('error')}; see {self.artifact_dir / 'agent.log'}")
        trace_path = Path(result["trace_path"])
        self.trace_paths.append(trace_path)
        return BackendResponse(
            text=result["text"], provider="ahe", model=self.model,
            status="completed", metadata={"ahe_trace_path": str(trace_path)},
        )

    def _receive(self, timeout: float) -> dict:
        """Bound both startup and an action; discard a worker that stops replying."""

        if not self._connection.poll(timeout):
            self.close()
            raise TimeoutError(f"AHE Agent did not respond within {timeout} seconds")
        try:
            return self._connection.recv()
        except EOFError as exc:
            self.close()
            raise RuntimeError(f"AHE Agent process exited; see {self.artifact_dir / 'agent.log'}") from exc

    def close(self) -> None:
        """Release the candidate process after the benchmark finishes with it."""

        process = getattr(self, "_process", None)
        connection = getattr(self, "_connection", None)
        if process is not None and process.is_alive():
            try:
                connection.send({"op": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        if connection is not None:
            connection.close()

    @staticmethod
    def _save_trace(agent: object, call_dir: Path) -> Path:
        """Write AHE's cleaned InMemoryTracer format after each action."""

        from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
        from trace_converter import extract_trace_data_from_inmemory_dump

        for tracer in agent.config.tracers:
            if isinstance(tracer, InMemoryTracer):
                cleaned = extract_trace_data_from_inmemory_dump(
                    tracer.dump_traces(), include_system_prompt_message=True,
                    include_user_message=True, capture_errors=True, jsonable_output=True,
                )
                path = call_dir / "nexau_in_memory_tracer.cleaned.json"
                path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
                return path
        raise RuntimeError("AHE coding agent has no InMemoryTracer")

    @staticmethod
    def _worker(
        connection: Connection,
        workspace: Path,
        config_path: Path,
        sandbox_dir: Path,
        upstream_root: Path,
        log_path: Path,
        model: str | None,
        reasoning_effort: str | None,
    ) -> None:
        """Initialize NexAU once, then serve turns from this candidate only."""

        with log_path.open("a", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    sys.path.insert(0, str(upstream_root))
                    # Candidate-local tools must win over any upstream imports;
                    # upstream remains available for trace_converter.py.
                    sys.path.insert(0, str(workspace))
                    from dotenv import load_dotenv
                    from nexau import Agent, AgentConfig
                    from nexau.archs.sandbox import LocalSandboxConfig

                    load_dotenv(upstream_root / ".env", override=False)
                    if model:
                        os.environ["LLM_MODEL"] = model
                    config = AgentConfig.from_yaml(config_path=config_path)
                    configure_reasoning_effort(config, reasoning_effort)
                    # The copied AHE YAML sets temperature=0.7 and top_p=0.95,
                    # but this benchmark model rejects both parameters. NexAU
                    # omits either request field when its value is None.
                    config.llm_config.temperature = None
                    config.llm_config.top_p = None
                    # AgentConfig is already constructed: NexAU expects a typed
                    # sandbox object here and accesses its .work_dir directly.
                    # LocalSandboxConfig has no persist_sandbox option; the
                    # process retains the sandbox for this Agent's lifetime.
                    config.sandbox_config = LocalSandboxConfig(work_dir=str(sandbox_dir))
                    agent = Agent(config=config)
                    connection.send({"ready": True})
                    AHEAgent._serve_actions(connection, agent, sandbox_dir, log)
                except BaseException as exc:
                    traceback.print_exc(file=log)
                    try:
                        connection.send({"ready": False, "error": str(exc)})
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                finally:
                    connection.close()

    @staticmethod
    def _serve_actions(connection: Connection, agent: object, sandbox_dir: Path, log: object) -> None:
        """Keep call-specific prompt and trace handling outside initialization."""

        while True:
            request = connection.recv()
            if request.get("op") == "close":
                return
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                raw = agent.run(
                    message=request["prompt"],
                    context={"date": now, "username": os.environ.get("USER") or os.environ.get("USERNAME"),
                             "working_directory": str(sandbox_dir),
                             "env_content": {"date": now, "working_directory": str(sandbox_dir)}},
                )
                answer = raw[0] if isinstance(raw, tuple) else raw
                trace_path = AHEAgent._save_trace(agent, Path(request["call_dir"]))
                connection.send({"ok": True, "text": str(answer or ""), "trace_path": str(trace_path)})
            except BaseException as exc:
                traceback.print_exc(file=log)
                connection.send({"ok": False, "error": str(exc)})


def configure_reasoning_effort(config: object, effort: str | None) -> None:
    """Apply reasoning effort only when the installed NexAU exposes the field.

    AHE is vendored against an external NexAU runtime.  Silently dropping the
    requested setting would make an experiment's model configuration false,
    so unsupported runtimes fail during initialization instead.
    """

    normalized = str(effort or "").strip()
    if not normalized:
        return
    llm_config = getattr(config, "llm_config", None)
    if llm_config is None:
        raise RuntimeError("NexAU AgentConfig does not expose llm_config; cannot set reasoning_effort")
    if isinstance(llm_config, dict):
        if "reasoning_effort" not in llm_config:
            raise RuntimeError("Installed NexAU LLM config does not support reasoning_effort")
        llm_config["reasoning_effort"] = normalized
        return
    fields = getattr(llm_config, "model_fields", None)
    if fields is None:
        fields = getattr(type(llm_config), "model_fields", None)
    if fields is not None and "reasoning_effort" not in fields:
        raise RuntimeError("Installed NexAU LLM config does not support reasoning_effort")
    if not hasattr(llm_config, "reasoning_effort"):
        raise RuntimeError("Installed NexAU LLM config does not support reasoning_effort")
    setattr(llm_config, "reasoning_effort", normalized)
