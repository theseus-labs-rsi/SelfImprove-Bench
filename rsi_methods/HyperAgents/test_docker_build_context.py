"""Offline checks for candidate image build contexts."""

import importlib.util
from pathlib import Path
import os
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parent
# Use the package name expected by docker_utils' relative imports.
SPEC = importlib.util.spec_from_file_location(
    "rsi_methods.HyperAgents.hyperagents.utils.docker_utils",
    ROOT / "hyperagents/utils/docker_utils.py",
)
docker_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docker_utils)


class DockerBuildContextTests(unittest.TestCase):
    def test_cache_is_never_opened_and_existing_ignore_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ["Dockerfile", "Dockerfile.runtime", "task_agent.py",
                     "secret.txt", "__pycache__/task.pyc",
                     "nested/__pycache__/task.pyc", "nested/task.pyc",
                     "nested/task.pyo", "nested/task.py"]
            for name in files:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("test\n")
            (root / ".dockerignore").write_text("# private data\n\n secret.txt \n!**/*.pyc\n")
            client = Mock()
            captured = []

            def build(**kwargs):
                self.assertTrue(kwargs["custom_context"])
                self.assertNotIn("path", kwargs)
                with tarfile.open(fileobj=kwargs["fileobj"]) as archive:
                    captured.append(set(archive.getnames()))
                return Mock(), []

            client.images.build.side_effect = build
            original_open = tarfile.bltn_open

            def guarded_open(name, *args, **kwargs):
                path = Path(name)
                if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                    raise PermissionError("simulated container-owned cache")
                return original_open(name, *args, **kwargs)

            with patch.object(tarfile, "bltn_open", side_effect=guarded_open):
                for dockerfile in ("Dockerfile", "Dockerfile.runtime"):
                    docker_utils._build_image(client, root, dockerfile=dockerfile, tag="test")
            for names in captured:
                self.assertTrue({"Dockerfile", "Dockerfile.runtime", "task_agent.py",
                                 "nested/task.py"}.issubset(names))
                self.assertNotIn("secret.txt", names)
                self.assertFalse(any("__pycache__" in name or name.endswith((".pyc", ".pyo"))
                                     for name in names))
            self.assertTrue(client.images.build.call_args.kwargs["fileobj"].closed)

    def test_both_runtime_and_candidate_builds_use_filtered_context(self):
        client = Mock()
        client.images.list.return_value = []
        client.containers.get.side_effect = docker_utils.docker.errors.NotFound("missing")
        with patch.object(docker_utils, "_build_image", return_value=(Mock(), [])) as build:
            docker_utils.build_container(client, runtime_image_name="runtime", verbose=False)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(build.call_args_list[0].kwargs["dockerfile"], "Dockerfile.runtime")
        self.assertEqual(build.call_args_list[1].kwargs["tag"], "app")

    def test_host_network_requires_explicit_opt_in(self):
        for environment, expected in (({}, None), ({"HYPERAGENTS_DOCKER_NETWORK": "host"}, "host")):
            with self.subTest(environment=environment):
                client = Mock()
                client.images.list.return_value = []
                client.containers.get.side_effect = docker_utils.docker.errors.NotFound("missing")
                with patch.dict(os.environ, environment, clear=True), \
                        patch.object(docker_utils, "_build_image", return_value=(Mock(), [])):
                    docker_utils.build_container(client, verbose=False)
                self.assertEqual(client.containers.run.call_args.kwargs.get("network_mode"), expected)

    def test_container_startup_preserves_failure_reason(self):
        client = Mock()
        client.images.list.return_value = []
        client.containers.get.side_effect = docker_utils.docker.errors.NotFound("missing")
        client.containers.run.side_effect = RuntimeError("network unavailable")
        with patch.object(docker_utils, "_build_image", return_value=(Mock(), [])):
            with self.assertRaisesRegex(RuntimeError, "Docker container creation failed: network unavailable"):
                docker_utils.build_container(client, verbose=False)


if __name__ == "__main__":
    unittest.main()
