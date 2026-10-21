"""Real namespace checks: no API calls or NexAU installation required."""

import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

from rsi_methods.AHE.sandbox import reject_symlinks, sandbox_command, training_history_paths


@pytest.fixture
def bwrap_available():
    if sys.platform != "linux" or shutil.which("bwrap") is None:
        pytest.skip("Linux bubblewrap is required for namespace integration tests")
    probe = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-user", "/bin/true"],
                           capture_output=True, text=True)
    if probe.returncode:
        pytest.skip(f"User namespaces unavailable: {probe.stderr}")


def test_no_unsafe_fallback_without_bwrap(tmp_path):
    with patch("rsi_methods.AHE.sandbox.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="No unsandboxed fallback"):
            sandbox_command(["true"], cwd=tmp_path)


def test_real_filesystem_allowlist_and_subprocess_inheritance(tmp_path, bwrap_available):
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    workspace.mkdir()
    output.mkdir()
    hidden = tmp_path / "hidden-score.json"
    hidden.write_text("secret")
    (workspace / "prompt.txt").write_text("visible")
    (workspace / "escape").symlink_to(hidden)
    script = """
import pathlib, subprocess, sys
workspace, output, hidden = map(pathlib.Path, sys.argv[1:])
assert (workspace / 'prompt.txt').read_text() == 'visible'
assert not hidden.exists()
assert not (workspace / 'escape').exists()
try:
    (workspace / 'prompt.txt').write_text('tampered')
except OSError:
    pass
else:
    raise AssertionError('Read-only harness became writable')
assert subprocess.run(['/bin/cat', str(hidden)], capture_output=True).returncode != 0
(output / 'trace.txt').write_text('written')
from benchmark.contracts import BackendResponse
from rsi_methods.AHE.refine import _worker
try:
    import benches
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Game implementation was exposed to the actor')
print('isolated')
"""
    completed = subprocess.run(sandbox_command(
        [sys.executable, "-c", script, str(workspace), str(output), str(hidden)],
        readable=[workspace], writable=[output], cwd=output,
    ), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "isolated"
    assert hidden.read_text() == "secret"
    assert (workspace / "prompt.txt").read_text() == "visible"
    assert (output / "trace.txt").read_text() == "written"


def test_evolver_can_edit_only_explicit_host_outputs(tmp_path, bwrap_available):
    workspace = tmp_path / "candidate/workspace"
    workspace.mkdir(parents=True)
    evidence = tmp_path / "training-evidence"
    evidence.mkdir()
    (evidence / "score.json").write_text("0.2")
    manifest = workspace.parent / "change_manifest.json"
    manifest.touch()
    script = """
from pathlib import Path
import sys
workspace, evidence, manifest = map(Path, sys.argv[1:])
(workspace / 'actor.py').write_text('updated')
manifest.write_text('{}')
assert (evidence / 'score.json').read_text() == '0.2'
try:
    (evidence / 'score.json').write_text('1.0')
except OSError:
    pass
else:
    raise AssertionError('Evidence became writable')
"""
    completed = subprocess.run(sandbox_command(
        [sys.executable, "-c", script, str(workspace), str(evidence), str(manifest)],
        readable=[evidence], writable=[workspace, manifest], cwd=workspace,
    ), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert (workspace / "actor.py").read_text() == "updated"
    assert manifest.read_text() == "{}"
    assert (evidence / "score.json").read_text() == "0.2"


def test_external_package_path_is_visible_without_its_parent(
    tmp_path, bwrap_available, monkeypatch,
):
    packages = tmp_path / "external/lib/site-packages"
    packages.mkdir(parents=True)
    (packages / "ahe_runtime_probe.py").write_text("VALUE = 42\n")
    hidden = packages.parent / "holdout.json"
    hidden.write_text("secret")
    arbitrary = tmp_path / "host-source"
    arbitrary.mkdir()
    (arbitrary / "host_only_module.py").write_text("VALUE = 99\n")
    monkeypatch.syspath_prepend(str(packages))
    monkeypatch.syspath_prepend(str(arbitrary))
    script = """
import sys
from pathlib import Path
import ahe_runtime_probe
assert ahe_runtime_probe.VALUE == 42
assert not Path(sys.argv[1]).exists()
try:
    import host_only_module
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Arbitrary host import path was exposed')
"""
    completed = subprocess.run(sandbox_command(
        [sys.executable, "-c", script, str(hidden)], cwd=packages,
    ), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_rejects_harness_symlinks_before_host_copy(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "escape").symlink_to("/etc/passwd")
    with pytest.raises(RuntimeError, match="rejects symlinks"):
        reject_symlinks(workspace)


def test_history_cannot_bind_an_external_file(tmp_path):
    (tmp_path / "evolution_history.json").write_text('[{"change_evaluation":"../outside.json"}]')
    with pytest.raises(RuntimeError, match="escapes"):
        training_history_paths(tmp_path)


@pytest.mark.parametrize("linked_file", ["evolution_history.md", "change_manifest.json"])
def test_history_rejects_symlinked_evidence(tmp_path, linked_file):
    run = tmp_path / "run"
    run.mkdir()
    hidden = tmp_path / "holdout.json"
    hidden.write_text("secret")
    evidence = run / "candidates/iteration_001"
    evidence.mkdir(parents=True)
    (evidence / "change_evaluation.json").write_text("{}")
    (run / "evolution_history.json").write_text(
        '[{"change_evaluation":"candidates/iteration_001/change_evaluation.json"}]',
    )
    if linked_file == "change_manifest.json":
        variant = evidence / "variant_0"
        variant.mkdir()
        (variant / linked_file).symlink_to(hidden)
    else:
        (run / linked_file).symlink_to(hidden)
    with pytest.raises(RuntimeError, match="escapes"):
        training_history_paths(run)


def test_launcher_pipe_lifecycle_with_model_free_worker(tmp_path, bwrap_available):
    """Exercise real bwrap/FD passing, substituting only the NexAU worker body."""
    from rsi_methods.AHE.agent import AHEAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "code_agent.yaml").write_text("name: offline\n")
    hidden = tmp_path / "holdout.json"
    hidden.write_text("secret")
    script = """
import sys
from pathlib import Path
from multiprocessing.connection import Connection
connection = Connection(int(sys.argv[1]))
assert not Path(sys.argv[2]).exists()
connection.send({'ready': True})
assert connection.recv() == {'op': 'close'}
connection.close()
"""

    def synthetic_worker(command, **kwargs):
        fd = command[command.index("--worker-fd") + 1]
        return sandbox_command([sys.executable, "-c", script, fd, str(hidden)], **kwargs)

    with patch("rsi_methods.AHE.agent.sandbox_command", side_effect=synthetic_worker):
        agent = AHEAgent(workspace, tmp_path / "artifacts", timeout_seconds=5)
        try:
            assert agent._process.is_alive()
        finally:
            agent.close()
        assert not agent._process.is_alive()
        assert agent._process.exitcode == 0


def test_actor_rejects_symlinked_workspace_root(tmp_path):
    from rsi_methods.AHE.agent import AHEAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(RuntimeError, match="rejects symlinks"):
        AHEAgent(alias, tmp_path / "artifacts")
