"""File-based skills and bounded tool use for API-backed game agents."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .backend import OpenAIBackend
from .contracts import TokenUsage, to_jsonable
from .native_support import write_files


class ArtifactAgent:
    def __init__(self, owner, files, instructions, *, always_on, root, actor_config=None, _workspace=None):
        self.owner = owner
        self.files = dict(files)
        self.instructions = instructions
        self.always_on = tuple(always_on)
        # Each acting snapshot gets its own filesystem. Helpers can write local
        # working data without contaminating another task/trial of this harness.
        root.mkdir(parents=True, exist_ok=True)
        self.root = (_workspace if _workspace is not None else
                     write_files(Path(tempfile.mkdtemp(prefix="run-", dir=root)), self.files))
        self.actor_config = dict(actor_config or {})

    def act(self, prompt, *, image_paths=()):
        return self._run(prompt, image_paths=image_paths, depth=0)

    def _run(self, prompt, *, image_paths=(), depth=0):
        system = self.instructions + "\n\n" + "\n\n".join(
            f"# {name}\n{self.files[name]}" for name in self.always_on if name in self.files)
        system += "\nAvailable artifact files:\n" + "\n".join(sorted(self.files))
        system += "\nRead relevant skills/references on demand. Return the requested game action as final text."
        user = OpenAIBackend._build_chat_messages(prompt, tuple(Path(p) for p in image_paths))[0]["content"]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        specs = [
            ("read_file", "Read a skill, reference or helper source.", {"path": {"type": "string"}}, ["path"]),
            ("run_script", "Execute a Python helper from scripts/ with string arguments.",
             {"path": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}}}, ["path"]),
        ]
        if depth == 0:
            specs.append(("delegate", "Ask a subagent in its own context.",
                          {"path": {"type": "string"}, "task": {"type": "string"}}, ["path", "task"]))
        tools = [{"type": "function", "function": {"name": name, "description": desc,
                  "parameters": {"type": "object", "properties": props, "required": required}}}
                 for name, desc, props, required in specs]
        responses = []
        limit = int(self.owner.options.get("actor_tool_steps", 12))
        for _ in range(limit):
            response = self.owner.dispatcher.distribute("openai", "", messages=messages,
                tools=tools, model=self.owner.actor_model, api_mode=self.owner.api_mode,
                max_output_tokens=self.actor_config.get("max_output_tokens", self.owner.actor_max_output_tokens),
                reasoning_effort=self.actor_config.get("reasoning_effort", self.owner.actor_reasoning_effort),
                temperature=self.actor_config.get("temperature", self.owner.temperature))
            responses.append(response)
            if not response.tool_calls:
                fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")
                def total(field):
                    values = [getattr(r.usage, field, None) for r in responses]
                    return sum(values) if all(isinstance(v, int) for v in values) else None
                return replace(response, usage=TokenUsage(**{f: total(f) for f in fields}),
                    metadata={**dict(response.metadata or {}), "rsi_method": self.owner.method_name,
                              "artifact_calls": [to_jsonable(r) for r in responses]})
            messages.append({"role": "assistant", "content": response.text or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name,
                 "arguments": tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments)}}
                for tc in response.tool_calls]})
            for tc in response.tool_calls:
                try:
                    args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else dict(tc.arguments)
                    name = args["path"]
                    if name not in self.files:
                        raise ValueError("unknown artifact file")
                    if tc.name == "read_file":
                        result = self.files[name]
                    elif tc.name == "run_script":
                        if not (name.startswith("scripts/") or "/scripts/" in name) or not name.endswith(".py"):
                            raise ValueError("only Python artifact helpers are executable")
                        completed = subprocess.run([sys.executable, str(self.root / name), *args.get("args", [])],
                            cwd=self.root, capture_output=True, text=True,
                            timeout=min(30, self.owner._method_context.timeout))
                        result = json.dumps({"exit_code": completed.returncode,
                                             "stdout": completed.stdout, "stderr": completed.stderr})
                    elif tc.name == "delegate" and depth == 0 and name.startswith("sub_agents/"):
                        child = ArtifactAgent(self.owner, self.files, self.files[name],
                                              always_on=(), root=self.root, actor_config=self.actor_config,
                                              _workspace=self.root)
                        answer = child._run(args["task"], depth=1)
                        responses.append(answer)
                        result = answer.text
                    else:
                        raise ValueError("unsupported artifact tool")
                except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
                    result = f"Tool error: {exc}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        raise RuntimeError("artifact agent exceeded actor_tool_steps without a final action")


__all__ = ["ArtifactAgent"]
