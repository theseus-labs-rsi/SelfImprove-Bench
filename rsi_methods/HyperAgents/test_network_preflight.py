"""Network startup checks never call the model API or expose credentials."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import lifecycle
from .HyperAgentsAdaptor import HyperAgentsAdaptor
from .network_preflight import _CHECK_NETWORK, check_model_network, container_model_environment
from .runtime_policy import HyperAgentsRuntimePolicy


class NetworkPreflightTests(unittest.TestCase):
    def probe(self, environment, *, names=None, error=None):
        output = io.StringIO()
        code = 0
        with patch.dict(os.environ, environment, clear=True), \
                patch("sys.argv", ["check", json.dumps(names or ["OPENAI_BASE_URL"])]), \
                patch("socket.create_connection", side_effect=error) as connect, \
                redirect_stdout(output):
            try:
                exec(compile(_CHECK_NETWORK, "<network-preflight>", "exec"), {})
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue(), connect

    def test_direct_endpoint(self):
        code, output, connect = self.probe({"OPENAI_BASE_URL": "https://model.example/v1"})
        self.assertEqual(code, 0)
        connect.assert_called_once_with(("model.example", 443), timeout=5)
        self.assertIn("authentication and model availability not tested", output)

    def test_default_endpoint(self):
        code, _, connect = self.probe({})
        self.assertEqual(code, 0)
        connect.assert_called_once_with(("api.openai.com", 443), timeout=5)

    def test_dead_loopback_proxy_is_actionable_and_redacts_secrets(self):
        code, output, connect = self.probe({
            "OPENAI_BASE_URL": "https://model.example/v1?token=endpoint-secret",
            "OPENAI_API_KEY": "key-secret",
            "HTTPS_PROXY": "http://user:proxy-secret@127.0.0.1:17890",
        }, error=ConnectionRefusedError())
        self.assertEqual(code, 1)
        connect.assert_called_once_with(("127.0.0.1", 17890), timeout=5)
        self.assertIn("HYPERAGENTS_DOCKER_NETWORK=host", output)
        for secret in ("endpoint-secret", "key-secret", "proxy-secret"):
            self.assertNotIn(secret, output)

    def test_no_proxy_bypasses_proxy(self):
        code, _, connect = self.probe({
            "OPENAI_BASE_URL": "https://model.example:8443/v1",
            "HTTPS_PROXY": "http://127.0.0.1:17890",
            "NO_PROXY": "model.example",
        })
        self.assertEqual(code, 0)
        connect.assert_called_once_with(("model.example", 8443), timeout=5)

    def test_lowercase_proxy_takes_precedence_and_all_proxy_is_supported(self):
        for overrides in ({"https_proxy": "http://lower.example:9000"},
                          {"ALL_PROXY": "socks5://lower.example:9000"}):
            with self.subTest(overrides=overrides):
                env = {"OPENAI_BASE_URL": "https://model.example/v1", **overrides}
                if "https_proxy" in overrides:
                    env["HTTPS_PROXY"] = "http://upper.example:8000"
                code, _, connect = self.probe(env)
                self.assertEqual(code, 0)
                connect.assert_called_once_with(("lower.example", 9000), timeout=5)

    def test_distinct_meta_and_task_routes_are_both_checked(self):
        code, _, connect = self.probe({
            "OPENAI_BASE_URL": "https://task.example/v1",
            "DEEPSEEK_BASE_URL": "http://meta.example:8080/v1",
        }, names=["OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"])
        self.assertEqual(code, 0)
        self.assertEqual([call.args[0] for call in connect.call_args_list],
                         [("task.example", 443), ("meta.example", 8080)])

    def test_duplicate_routes_only_checked_once(self):
        code, _, connect = self.probe({}, names=["OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"])
        self.assertEqual(code, 0)
        self.assertEqual(connect.call_count, 1)

    def test_bad_url_fails_without_printing_it(self):
        code, output, connect = self.probe({
            "OPENAI_BASE_URL": "https://model.example:secret/v1",
        })
        self.assertEqual(code, 1)
        self.assertNotIn("secret", output)
        connect.assert_not_called()

    def test_wrapper_uses_runtime_environment(self):
        container = Mock()
        container.exec_run.return_value = SimpleNamespace(exit_code=1, output=b"route unavailable")
        runtime = SimpleNamespace(task_model={}, meta_model={},
                                  llm_environment={"HTTPS_PROXY": "http://proxy:80"})
        with self.assertRaisesRegex(RuntimeError, "HyperAgents network preflight failed"):
            check_model_network(container, runtime)
        self.assertEqual(container.exec_run.call_args.kwargs["environment"],
                         container_model_environment(runtime.llm_environment))

    def test_preflight_failure_removes_container_before_candidate_reset(self):
        runtime = SimpleNamespace(docker_client=Mock())
        snapshot = SimpleNamespace(workspace="/candidate")
        container = Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(lifecycle, "setup_logger"), \
                patch.object(lifecycle, "build_container", return_value=container), \
                patch.object(lifecycle, "check_model_network", side_effect=RuntimeError("preflight")), \
                patch.object(lifecycle, "reconstruct_agent") as reconstruct:
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                lifecycle.instantiate_agent(snapshot, runtime=runtime, run_id="test",
                                            artifact_dir=Path(directory) / "artifacts")
        container.remove.assert_called_once_with(force=True)
        reconstruct.assert_not_called()

    def test_fatal_policy_stops_for_every_hyperagents_game_config(self):
        from benchmark.config import load_config
        from benchmark.runner import _must_stop_after_error

        root = Path(__file__).resolve().parents[2]
        configs = sorted((root / "configs/main").glob("*_hyperagents.json"))
        self.assertEqual(len(configs), 6)
        for path in configs:
            config = replace(load_config(path), continue_on_error=True)
            for message in ("HyperAgents network preflight failed: unavailable",
                            "Docker image build failed: permission denied"):
                with self.subTest(config=path.name, error=message):
                    self.assertTrue(_must_stop_after_error(config, {
                        "status": "error", "error": message,
                    }))
        self.assertIsInstance(HyperAgentsAdaptor.runtime_policy, HyperAgentsRuntimePolicy)
        self.assertFalse(HyperAgentsAdaptor.runtime_policy.is_fatal_error("invalid game action"))

    def test_all_proxy_is_forwarded(self):
        with patch.dict(os.environ, {"ALL_PROXY": "http://proxy:80"}, clear=True):
            self.assertEqual(HyperAgentsAdaptor._collect_llm_environment(set()),
                             container_model_environment({"ALL_PROXY": "http://proxy:80"}))

    def test_image_proxy_variables_are_always_overridden(self):
        environment = container_model_environment({"HTTPS_PROXY": "http://host-proxy:80"})
        self.assertEqual(environment["https_proxy"], "http://host-proxy:80")
        self.assertEqual(environment["http_proxy"], "")
        self.assertEqual(environment["HTTP_PROXY"], "")
        self.assertEqual(environment["all_proxy"], "")
        self.assertEqual(environment["NO_PROXY"], "")

    def test_empty_host_lowercase_proxy_disables_uppercase(self):
        environment = container_model_environment({
            "HTTPS_PROXY": "http://upper.example:80", "https_proxy": "",
        })
        code, _, connect = self.probe({
            **environment, "OPENAI_BASE_URL": "https://model.example/v1",
        })
        self.assertEqual(code, 0)
        connect.assert_called_once_with(("model.example", 443), timeout=5)


if __name__ == "__main__":
    unittest.main()
