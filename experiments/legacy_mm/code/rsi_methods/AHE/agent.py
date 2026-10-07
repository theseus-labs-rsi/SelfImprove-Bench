"""AHE coding agent with one NexAU runtime per benchmark agent object.

The copied harness is fixed when this object is built.  A dedicated process
keeps candidate-specific ``tools`` imports separate, while retaining one
NexAU Agent across calls to ``act`` on this benchmark task.
"""

from __future__ import annotations

import base64
import json
import multiprocessing
import mimetypes
import os
import shutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

from benchmark.contracts import BackendResponse, TokenUsage
from benchmark.token_ledger import nexau_usage_snapshot


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
        agent_limits: Mapping[str, int] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.config_path = self.workspace / "code_agent.yaml"
        self.artifact_dir = artifact_dir.resolve()
        self.sandbox_dir = self.artifact_dir / "sandbox"
        self.upstream_root = Path(__file__).resolve().parent / "agentic-harness-engineering"
        self.model = model or os.environ.get("LLM_MODEL")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.agent_limits = dict(agent_limits or {})
        self.trace_paths: list[Path] = []
        self._last_usage_snapshot: dict | None = None
        self._attempt_index = 0
        self._unreconciled_failed_attempts = False
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self.sandbox_dir.mkdir()
        self._start_worker()

    def _start_worker(self) -> None:
        parent, child = multiprocessing.Pipe()
        self._connection = parent
        self._process = multiprocessing.Process(
            target=AHEAgent._worker,
            args=(child, self.workspace, self.config_path, self.sandbox_dir,
                  self.upstream_root, self.artifact_dir / "agent.log", self.model,
                  self.reasoning_effort, self.agent_limits),
            daemon=True,
        )
        try:
            self._process.start()
            child.close()
            ready = self._receive(max(60.0, self.timeout_seconds))
            if not ready.get("ready"):
                raise RuntimeError(f"AHE Agent initialization failed: {ready.get('error')}; see {self.artifact_dir / 'agent.log'}")
        except BaseException:
            child.close()
            self.close()
            raise

    def reset_worker(self) -> None:
        """Replace a stuck/failed worker with a fresh empty one (same harness).

        Used before an episode resumes from its last checkpoint: the caller then
        restores the checkpointed NexAU history via ``restore_session``. Costs of
        the abandoned worker stay recorded in token_attempts.jsonl.
        """
        self.close()
        self.trace_paths = []
        self._last_usage_snapshot = None
        self._attempt_index = 0
        self._unreconciled_failed_attempts = False
        self._start_worker()

    def snapshot_session(self) -> dict:
        """Read the child NexAU conversation without any inference call.

        The enclosing step transaction must persist this *with* its browser
        snapshot and external ledger before declaring the step committed.
        """
        self._connection.send({"op": "snapshot_session"})
        result = self._receive(self.timeout_seconds)
        if not result.get("ok") or not isinstance(result.get("history"), list):
            raise RuntimeError(f"AHE session snapshot failed: {result.get('error')}")
        return {"schema": 1, "history": result["history"],
                "attempt_index": self._attempt_index,
                "last_usage_snapshot": self._last_usage_snapshot,
                "unreconciled_failed_attempts": self._unreconciled_failed_attempts,
                # Per-step traces of this episode so far: AHE evolution reads them all.
                "trace_paths": [str(path) for path in self.trace_paths]}

    def restore_session(self, snapshot: dict) -> None:
        """Restore a fresh worker BEFORE its first model call; never erase fees.

        Cumulative NexAU traces belong to the previous process, so usage starts
        with a new trace; callers MUST add this segment to the persisted ledger.
        """
        if self._attempt_index or self.trace_paths:
            raise RuntimeError("AHE restore must precede all new model calls")
        if (not isinstance(snapshot, dict) or snapshot.get("schema") != 1
                or not isinstance(snapshot.get("history"), list)
                or not isinstance(snapshot.get("attempt_index"), int)
                or snapshot["attempt_index"] < 0
                or snapshot.get("unreconciled_failed_attempts")):
            raise RuntimeError("Invalid or financially unreconciled AHE session")
        if validate_nexau_history(snapshot["history"]) != snapshot["history"]:
            raise RuntimeError("AHE history cannot be losslessly reconstructed")
        self._connection.send({"op": "restore_session", "history": snapshot["history"]})
        result = self._receive(self.timeout_seconds)
        if not result.get("ok") or result.get("history") != snapshot["history"]:
            raise RuntimeError(f"AHE NexAU history restore failed: {result.get('error')}")
        self._attempt_index = snapshot["attempt_index"]
        self._last_usage_snapshot = None  # new child starts a new trace ID
        self.adopt_trace_paths(snapshot.get("trace_paths") or [])

    def adopt_trace_paths(self, paths: Sequence[str | Path]) -> None:
        """Carry an interrupted episode's earlier step traces into this agent."""
        if self.trace_paths:
            raise RuntimeError("AHE trace history can only be adopted before new calls")
        resolved = [Path(path) for path in paths]
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise RuntimeError(f"AHE step traces missing on disk: {missing[:3]}")
        self.trace_paths = resolved

    def act(
        self,
        prompt: str,
        image_paths: Sequence[str | Path] | None = None,
    ) -> BackendResponse:
        """Run one turn; the newest trace includes this Agent's prior turns.

        Pass screenshots as Responses API input_image items, not stringified
        multimodal content; the agent's upstream harness remains unmodified.
        """

        if not isinstance(prompt, str):
            raise TypeError("AHEAgent.act() requires a text prompt")
        paths = [str(Path(p).resolve(strict=True)) for p in (image_paths or [])]
        for path in paths:
            if not Path(path).is_file():
                raise ValueError(f"AHE screenshot is not a file: {path}")
        call_dir = self.artifact_dir / uuid4().hex
        call_dir.mkdir()
        # GameWorld deletes its browser screenshot temp dir when the episode ends,
        # before refinement archives this request; keep a copy with the call.
        archived = []
        for index, path in enumerate(paths):
            target = call_dir / f"input_{index}{Path(path).suffix.lower()}"
            shutil.copy2(path, target)
            archived.append(str(target.resolve()))
        (call_dir / "request.json").write_text(
            json.dumps({"prompt": prompt, "image_paths": archived, "source_image_paths": paths},
                       ensure_ascii=False), encoding="utf-8",
        )
        # The model and the archived request refer to the same durable bytes.
        self._connection.send({"op": "act", "prompt": prompt, "image_paths": archived, "call_dir": str(call_dir)})
        try:
            result = self._receive(self.timeout_seconds)
        except BaseException as exc:
            self._unreconciled_failed_attempts = True
            self._record_attempt(call_dir, "transport_failure", None, str(exc))
            raise
        snapshot_path = call_dir / "nexau_usage_snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else None
        if not result.get("ok"):
            self._unreconciled_failed_attempts = True
            self._record_attempt(call_dir, "failed_usage_unknown", snapshot, str(result.get("error")))
            error = f"AHE act failed: {result.get('error')}; see {self.artifact_dir / 'agent.log'}"
            if result.get("error_type") in {"APIConnectionError", "APITimeoutError", "ProxyError", "ConnectError"}:
                raise ConnectionError(error)
            if result.get("status_code") in {408, 429, 500, 502, 503, 504}:
                raise ConnectionError(error)
            raise RuntimeError(error)
        trace_path = Path(result["trace_path"])
        self.trace_paths.append(trace_path)
        previous = self._last_usage_snapshot
        fields = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "total_tokens")
        calls = snapshot.get("calls") if isinstance(snapshot, dict) else None
        new_calls = []
        if isinstance(calls, list) and (previous is None or isinstance(previous.get("calls"), list)):
            # NexAU traces are cumulative per session; bill only the LLM spans
            # added since the last valid snapshot.  One span without usage then
            # makes only this step unknown, not every later step.
            previous_ids = [x.get("span_id") for x in (previous or {}).get("calls") or []]
            valid = (snapshot.get("trace_id") is not None
                     and (previous is None or (previous.get("trace_id") == snapshot["trace_id"]
                          and [x.get("span_id") for x in calls[:len(previous_ids)]] == previous_ids)))
            new_calls = calls[len(previous_ids):] if valid else []
            usage_values = {field: sum(x[field] for x in new_calls)
                            if new_calls and all(isinstance(x.get(field), int) for x in new_calls) else None
                            for field in fields}
        else:
            # Snapshot without per-span detail: cumulative totals only.
            valid = (snapshot is not None and snapshot.get("total_tokens") is not None
                     and snapshot.get("trace_id") is not None
                     and (previous is None or (previous.get("trace_id") == snapshot["trace_id"]
                          and previous.get("total_tokens") is not None
                          and snapshot["total_tokens"] >= previous["total_tokens"])))
            usage_values = {field: (snapshot[field] - (previous[field] if previous else 0)
                                    if valid and isinstance(snapshot.get(field), int)
                                    and (previous is None or isinstance(previous.get(field), int)) else None)
                            for field in fields}
        delta = usage_values["total_tokens"]
        complete = valid and delta is not None and not self._unreconciled_failed_attempts
        self._unreconciled_failed_attempts = False
        # A broken/reset span stream cannot be silently treated as a fresh zero.
        if valid:
            self._last_usage_snapshot = snapshot
        self._record_attempt(call_dir, "completed" if complete else "usage_unknown", snapshot, None,
                             billed_tokens=delta,
                             missing_usage_spans=[x.get("span_id") for x in new_calls
                                                  if not isinstance(x.get("total_tokens"), int)])
        return BackendResponse(
            text=result["text"], provider="ahe", model=self.model,
            status="completed", usage=TokenUsage(**usage_values) if complete else None,
            metadata={"ahe_trace_path": str(trace_path), "ahe_usage_snapshot": str(snapshot_path),
                      "usage_complete": complete},
        )

    def _record_attempt(self, call_dir: Path, status: str, snapshot: dict | None,
                        error: str | None, *, billed_tokens: int | None = None,
                        missing_usage_spans: list | None = None) -> None:
        self._attempt_index += 1
        record = {"attempt": self._attempt_index, "call_dir": str(call_dir),
                  "status": status, "trace_id": snapshot.get("trace_id") if snapshot else None,
                  "cumulative_tokens": snapshot.get("total_tokens") if snapshot else None,
                  "billed_tokens": billed_tokens, "error": error}
        if missing_usage_spans:
            record["missing_usage_spans"] = missing_usage_spans
        with (self.artifact_dir / "token_attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

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
                raw_traces = tracer.dump_traces()
                snapshot = nexau_usage_snapshot(raw_traces)
                (call_dir / "nexau_usage_snapshot.json").write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                cleaned = extract_trace_data_from_inmemory_dump(
                    raw_traces, include_system_prompt_message=True,
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
        agent_limits: dict[str, int],
    ) -> None:
        """Initialize NexAU once, then serve turns from this candidate only."""

        with log_path.open("a", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    sys.path.insert(0, str(upstream_root))
                    # Candidate-local tools must win over any upstream imports;
                    # upstream remains available for trace_converter.py.
                    sys.path.insert(0, str(workspace))
                    # ``fork`` inherits the parent's imported ``tools`` package.
                    # After evolve() (or evaluation of an earlier candidate),
                    # that package can point at a *different* AHE workspace.
                    # NexAU resolves YAML bindings through importlib and would
                    # otherwise load the stale package rather than this snapshot.
                    # This process serves just one Agent, so purge only its
                    # private import table; never mutate the benchmark parent.
                    for name in list(sys.modules):
                        if name == "tools" or name.startswith("tools."):
                            del sys.modules[name]
                    from dotenv import load_dotenv
                    from nexau import Agent, AgentConfig
                    from nexau.archs.sandbox import LocalSandboxConfig

                    load_dotenv(upstream_root / ".env", override=False)
                    if model:
                        os.environ["LLM_MODEL"] = model
                    config = AgentConfig.from_yaml(config_path=config_path)
                    # Enforce limits even if an evolved candidate edited its YAML.
                    if agent_limits:
                        config.max_context_tokens = agent_limits["max_context_tokens"]
                        config.max_iterations = agent_limits["max_iterations"]
                        config.llm_config.max_tokens = agent_limits["max_tokens"]
                    configure_reasoning_effort(config, reasoning_effort)
                    # The copied AHE YAML sets temperature=0.7 and top_p=0.95,
                    # but this benchmark model rejects both parameters. NexAU
                    # omits either request field when its value is None.
                    config.llm_config.temperature = None
                    config.llm_config.top_p = None
                    # Without a timeout the OpenAI client waits up to 600 s per
                    # request, so one stalled gateway request exhausts the actor's
                    # whole step budget; fail it early and let NexAU retry. Worst
                    # case: 5 NexAU attempts x 2 client tries x 55 s + 15 s of
                    # backoff = 565 s, inside the 600 s actor step timeout.
                    config.llm_config.timeout = float(os.environ.get("RSI_AHE_REQUEST_TIMEOUT", 55))
                    config.llm_config.max_retries = 1
                    # The HTTP timeout bounds each read, so a stream that trickles
                    # events can outlive it; a non-streaming request (as the ACE and
                    # HyperAgents actors use) is bounded as a whole.
                    config.llm_config.stream = False
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
                if request.get("op") == "snapshot_session":
                    connection.send({"ok": True, "history": dump_nexau_history(agent)})
                    continue
                if request.get("op") == "restore_session":
                    history = request.get("history")
                    if not isinstance(history, list) or agent.history:
                        raise RuntimeError("NexAU worker is not a fresh empty session")
                    from nexau.core.messages import Message
                    validated = [Message.model_validate(item) for item in history]
                    agent.history.replace_all(validated)
                    agent.history.flush()
                    restored = dump_nexau_history(agent)
                    if restored != history:
                        raise RuntimeError("NexAU session history changed during restore")
                    connection.send({"ok": True, "history": restored})
                    continue
                if request.get("op") != "act":
                    raise RuntimeError("Unknown NexAU worker operation")
                # The gateway rejects more than 50 images per request. XBR
                # resets between tasks, but visual GameWorld turns still
                # accumulate full prompts within one task. Bound that history
                # while retaining the eight latest complete multimodal turns;
                # each original step trace remains on disk for evolution/audit.
                if request.get("image_paths"):
                    removed = compact_prior_game_turns(agent)
                    if removed:
                        (Path(request["call_dir"]) / "history_compaction.json").write_text(
                            json.dumps({"removed_messages": removed, "keep_recent_turns": 8}),
                            encoding="utf-8",
                        )
                compact_prior_screenshots(agent)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                raw = agent.run(
                    message=make_nexau_message(request["prompt"], request.get("image_paths") or []),
                    context={"date": now, "username": os.environ.get("USER") or os.environ.get("USERNAME"),
                             "working_directory": str(sandbox_dir),
                             "env_content": {"date": now, "working_directory": str(sandbox_dir)}},
                )
                answer = raw[0] if isinstance(raw, tuple) else raw
                trace_path = AHEAgent._save_trace(agent, Path(request["call_dir"]))
                connection.send({"ok": True, "text": str(answer or ""), "trace_path": str(trace_path)})
            except BaseException as exc:
                traceback.print_exc(file=log)
                try:
                    AHEAgent._save_trace(agent, Path(request["call_dir"]))
                except BaseException:
                    traceback.print_exc(file=log)
                connection.send({
                    "ok": False, "error": str(exc),
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                })



def dump_nexau_history(agent: object) -> list[dict]:
    """JSON-round-trippable transcript; includes the multimodal response_items."""
    return [message.model_dump(mode="json") for message in agent.history]


def validate_nexau_history(history: list[dict]) -> list[dict]:
    """Check that a persisted transcript round trips before paid inference."""
    from nexau.core.messages import Message
    return [Message.model_validate(message).model_dump(mode="json") for message in history]


def compact_prior_game_turns(agent: object, *, keep_recent: int = 8) -> int:
    """Drop old *complete* GameWorld turns, never an in-flight tool exchange.

    Per-step traces and the GameWorld action log remain immutable on disk. The
    next user prompt already includes GameWorld's recent action history. This
    is needed even with XBR's 100k context limit: limiting the *maximum* does
    not prevent re-billing a growing prompt every step of a visual game.
    """
    if keep_recent < 1:
        raise ValueError("keep_recent must be positive")
    history = list(agent.history)
    starts = []
    for index, message in enumerate(history):
        role = getattr(message, "role", None)
        if str(getattr(role, "value", role)).lower() != "user":
            continue
        items = getattr(message, "metadata", {}).get("response_items", [])
        if isinstance(items, list) and any(
            isinstance(item, dict) and isinstance(item.get("content"), list)
            and any(isinstance(part, dict) and part.get("type") == "input_text"
                    and "# Current Observation" in part.get("text", "")
                    for part in item["content"])
            for item in items
        ):
            starts.append(index)
    if len(starts) <= keep_recent:
        return 0
    start, cutoff = starts[0], starts[-keep_recent]
    removed = cutoff - start
    agent.history.replace_all(history[:start] + history[cutoff:])
    agent.history.flush()
    return removed


def compact_prior_screenshots(agent: object, *, keep_recent: int = 8) -> int:
    """Bound NexAU Responses input images without losing past text/actions.

    Only game screenshots inserted via ``make_nexau_message`` have this
    response_items shape. Replace older image items with the already present
    input_text; current observation is added *after* this call. Do not mutate
    the original messages in-place, because HistoryList persists replacements.
    """
    if keep_recent < 0:
        raise ValueError("keep_recent must be nonnegative")
    history = list(agent.history)
    image_positions: list[int] = []
    for index, message in enumerate(history):
        items = message.metadata.get("response_items", [])
        if isinstance(items, list) and any(
            isinstance(item, dict) and isinstance(item.get("content"), list)
            and any(isinstance(part, dict) and part.get("type") == "input_image"
                    for part in item["content"])
            for item in items
        ):
            image_positions.append(index)
    to_strip = image_positions[:-keep_recent] if keep_recent else image_positions
    if not to_strip:
        return 0
    for index in to_strip:
        old = history[index]
        cleaned = old.model_copy(deep=True)
        cleaned.metadata["response_items"] = [
            {**item, "content": [
                part for part in item["content"]
                if not isinstance(part, dict) or part.get("type") != "input_image"
            ]} if isinstance(item, dict) and isinstance(item.get("content"), list) else item
            for item in old.metadata["response_items"]
        ]
        history[index] = cleaned
    agent.history.replace_all(history)
    agent.history.flush()
    return len(to_strip)


def make_nexau_message(prompt: str, image_paths: Sequence[str | Path]) -> str | list:
    """Use NexAU 0.3.9's Responses passthrough for actual image input.

    Plain ImageBlocks are converted by NexAU's Responses adapter to
    ``str(content)``; response_items is its supported raw-message path.
    Keep the normal string path when there is no screenshot (upstream behavior).
    """
    if not image_paths:
        return prompt
    from nexau.core.messages import Message

    parts = [{"type": "input_text", "text": prompt}]
    for image_path in image_paths:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError(f"Unsupported AHE screenshot type: {path.suffix}")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append({"type": "input_image", "image_url": f"data:{mime};base64,{data}"})
    message = Message.user(prompt)
    message.metadata["response_items"] = [
        {"type": "message", "role": "user", "content": parts}
    ]
    return [message]


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
    # NexAU v0.3.9 stores OpenAI Responses API-specific options in
    # LLMConfig.extra_params. Its to_openai_params() forwards these options
    # to client.responses.create(), which accepts reasoning={"effort": ...}.
    # Check the serialized request rather than setting an unused attribute.
    if getattr(llm_config, "api_type", None) == "openai_responses":
        set_param = getattr(llm_config, "set_param", None)
        to_openai_params = getattr(llm_config, "to_openai_params", None)
        if callable(set_param) and callable(to_openai_params):
            reasoning = {"effort": normalized}
            set_param("reasoning", reasoning)
            if to_openai_params().get("reasoning") == reasoning:
                return
    # The upstream Evolve Agent uses Chat Completions rather than Responses.
    # NexAU forwards extra_params to that endpoint unchanged.
    if getattr(llm_config, "api_type", None) == "openai_chat_completion":
        set_param = getattr(llm_config, "set_param", None)
        to_openai_params = getattr(llm_config, "to_openai_params", None)
        if callable(set_param) and callable(to_openai_params):
            set_param("reasoning_effort", normalized)
            if to_openai_params().get("reasoning_effort") == normalized:
                return
    fields = getattr(llm_config, "model_fields", None)
    if fields is None:
        fields = getattr(type(llm_config), "model_fields", None)
    if fields is not None and "reasoning_effort" not in fields:
        raise RuntimeError("Installed NexAU LLM config does not support reasoning_effort")
    if not hasattr(llm_config, "reasoning_effort"):
        raise RuntimeError("Installed NexAU LLM config does not support reasoning_effort")
    setattr(llm_config, "reasoning_effort", normalized)
