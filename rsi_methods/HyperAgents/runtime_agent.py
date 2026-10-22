import base64
import binascii
import io
import json
import os
import tarfile
import uuid
from collections.abc import Sequence

from benchmark.contracts import BackendResponse, TokenUsage
from .network_preflight import container_model_environment


_RESULT_MARKER = "__RUNTIME_AGENT_RESULT__"

_RUN_TASK_AGENT = r"""
import base64
import importlib.util
import json
import sys

agent_path, encoded_model_config, history_path, encoded_inputs = sys.argv[1:]
inputs = json.loads(base64.b64decode(encoded_inputs).decode("utf-8"))

# Launcher transport bounds (execution option): default the SDK client's
# timeout/retries without touching the candidate repository's backend copy.
import os
if os.environ.get("RSI_OPENAI_REQUEST_TIMEOUT") or os.environ.get("RSI_OPENAI_MAX_RETRIES"):
    import openai
    _original_init = openai.OpenAI.__init__
    def _bounded_init(self, *args, **kwargs):
        if os.environ.get("RSI_OPENAI_REQUEST_TIMEOUT"):
            kwargs.setdefault("timeout", float(os.environ["RSI_OPENAI_REQUEST_TIMEOUT"]))
        if os.environ.get("RSI_OPENAI_MAX_RETRIES"):
            kwargs.setdefault("max_retries", int(os.environ["RSI_OPENAI_MAX_RETRIES"]))
        _original_init(self, *args, **kwargs)
    openai.OpenAI.__init__ = _bounded_init
model_config = json.loads(base64.b64decode(encoded_model_config).decode("utf-8"))

spec = importlib.util.spec_from_file_location("runtime_task_agent", agent_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load TaskAgent from {agent_path}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TaskAgentClass = module.TaskAgent
task_agent = TaskAgentClass(
    chat_history_file=history_path,
    **model_config,
)
prediction, history = task_agent.act(
    inputs["instruction"],
    image_paths=inputs.get("image_paths"),
    max_tokens=inputs.get("max_tokens", 16384),
)

payload = json.dumps(
    {"prediction": prediction, "history": history},
    ensure_ascii=False,
    default=str,
).encode("utf-8")
print("__RUNTIME_AGENT_RESULT__" + base64.b64encode(payload).decode("ascii"))
"""


class RuntimeAgent:
    """Runtime proxy for a TaskAgent whose code lives inside a container."""

    supports_parallel_calls = True

    def __init__(
        self,
        container,
        agent_path,
        repo_path,
        model,
        reasoning_effort,
        backend,
        api_mode,
        api_key_env,
        base_url_env,
        generation_id,
        artifact_dir,
        max_output_tokens=16384,
        llm_environment=None,
    ):
        self.container = container
        self.agent_path = agent_path
        self.repo_path = repo_path
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.backend = backend
        self.api_mode = api_mode
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.generation_id = generation_id
        self.artifact_dir = os.path.abspath(artifact_dir)
        self.max_output_tokens = max_output_tokens
        self.llm_environment = dict(llm_environment or {})

    def run(self, inputs):
        """Load and instantiate TaskAgent by path, then run one structured input."""
        encoded_inputs = base64.b64encode(
            json.dumps(inputs, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        model_config = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "backend": self.backend,
            "api_mode": self.api_mode,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
        }
        encoded_model_config = base64.b64encode(
            json.dumps(model_config).encode("utf-8")
        ).decode("ascii")
        history_path = os.path.join(
            "/tmp", f"task_agent_history_{self.generation_id}_{uuid.uuid4().hex}.md"
        )
        command = [
            "python",
            "-c",
            _RUN_TASK_AGENT,
            self.agent_path,
            encoded_model_config,
            history_path,
            encoded_inputs,
        ]
        environment = container_model_environment(self.llm_environment)
        for name in ("RSI_OPENAI_REQUEST_TIMEOUT", "RSI_OPENAI_MAX_RETRIES"):
            if os.environ.get(name):
                environment[name] = os.environ[name]
        exec_result = self.container.exec_run(
            cmd=command,
            workdir=self.repo_path,
            environment=environment,
        )
        output = exec_result.output.decode("utf-8", errors="replace")
        if exec_result.exit_code != 0:
            raise RuntimeError(f"TaskAgent failed with exit code {exec_result.exit_code}: {output}")

        marker_position = output.rfind(_RESULT_MARKER)
        if marker_position < 0:
            raise RuntimeError(f"TaskAgent returned no structured result: {output}")
        result_tail = output[marker_position + len(_RESULT_MARKER):]
        encoded_result = result_tail.partition("\n")[0].strip()
        try:
            payload = base64.b64decode(encoded_result, validate=True)
            result = json.loads(payload.decode("utf-8"))
            history = result.get("history")
            usage: dict = {}
            assistant_message: dict = {}
            if isinstance(history, list):
                for message in reversed(history):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        assistant_message = message
                        candidate = message.get("usage")
                        if isinstance(candidate, dict):
                            usage = candidate
                        break
            result["usage"] = (
                {"available": True, **usage} if usage
                else {"available": False}
            )
            result["reasoning"] = self._first_text(
                assistant_message,
                "reasoning",
                "reasoning_content",
                "thinking",
            )
            result["status"] = assistant_message.get("status", "completed")
            result["finish_reason"] = assistant_message.get("finish_reason")
            return result
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "TaskAgent returned a malformed structured result frame."
            ) from exc

    def _stage_images(
        self, image_paths: Sequence[str | os.PathLike[str]],
    ) -> list[str]:
        """Copy screenshots into the candidate container via Docker archives."""
        staged: list[str] = []
        destination = "/tmp/gameworld_images"
        self.container.exec_run(["mkdir", "-p", destination])
        for source in image_paths:
            path = os.path.abspath(os.fspath(source))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"GameWorld screenshot not found: {path}")
            name = f"{uuid.uuid4().hex}_{os.path.basename(path)}"
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as tar:
                info = tar.gettarinfo(path, arcname=name)
                info.uid = 0
                info.gid = 0
                with open(path, "rb") as source_file:
                    tar.addfile(info, source_file)
            archive.seek(0)
            if not self.container.put_archive(destination, archive):
                raise RuntimeError(f"Failed to stage screenshot into container: {path}")
            staged.append(f"{destination}/{name}")
        return staged

    def act(
        self,
        prompt: str,
        image_paths: Sequence[str | os.PathLike[str]] | None = None,
    ) -> BackendResponse:
        """Run one stateless benchmark turn through the candidate TaskAgent."""
        staged_images = self._stage_images(image_paths) if image_paths else []
        result = self.run(
            {
                "instruction": prompt,
                "image_paths": staged_images,
                "max_tokens": self.max_output_tokens,
            }
        )
        prediction = result.get("prediction")
        if not isinstance(prediction, str) or not prediction.strip():
            raise RuntimeError("TaskAgent returned an empty prediction.")
        return BackendResponse(
            text=prediction,
            reasoning=result.get("reasoning"),
            usage=self._token_usage(result.get("usage")),
            provider=self.backend,
            model=self.model,
            status=result.get("status", "completed"),
            finish_reason=result.get("finish_reason"),
        )

    @staticmethod
    def _first_text(source: dict, *keys: str) -> str | None:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @classmethod
    def _token_usage(cls, raw_usage) -> TokenUsage | None:
        if not isinstance(raw_usage, dict) or not raw_usage.get("available"):
            return None
        input_tokens = cls._first_int(
            raw_usage, "input_tokens", "prompt_tokens", "prompt_num_tokens"
        )
        output_tokens = cls._first_int(
            raw_usage, "output_tokens", "completion_tokens", "response_num_tokens"
        )
        total_tokens = cls._first_int(
            raw_usage, "total_tokens", "total_num_tokens"
        )
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cls._first_int(
                raw_usage, "cached_input_tokens", "cached_prompt_num_tokens"
            ),
            reasoning_tokens=cls._first_int(
                raw_usage, "reasoning_tokens", "reasoning_num_tokens"
            ),
        )

    @staticmethod
    def _first_int(source: dict, *keys: str) -> int | None:
        for key in keys:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None
