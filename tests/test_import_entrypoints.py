"""No-API regressions for package entry points and explicit import bridges."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from benchmark.imports import load_vendor_package, prepend_legacy_paths


ROOT = Path(__file__).resolve().parents[1]


def run_python(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("module", [
    "benchmark", "analyze", "benchmark.hci",
    "benches.farm.batch_generate_tasks", "benches.combat.batch_generate_tasks",
])
def test_supported_module_entrypoints_have_help(module):
    result = run_python("-m", module, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("game", ["farm", "combat"])
def test_batch_generator_module_writes_one_reproducible_task(game, tmp_path):
    directories = [tmp_path / "first", tmp_path / "second"]
    for directory in directories:
        result = run_python(
            "-m", f"benches.{game}.batch_generate_tasks",
            "--min-difficulty", "1", "--max-difficulty", "1",
            "--per-difficulty", "1", "--seed", "901", "--output-dir", str(directory),
        )
        assert result.returncode == 0, result.stderr
    first, second = [list(directory.glob("*.json")) for directory in directories]
    assert len(first) == len(second) == 1
    task = json.loads(first[0].read_text())
    assert task == json.loads(second[0].read_text())
    assert task["difficulty"] == 1
    assert task["task_id"] == f"{game}-d1-01"
    assert task["reference_profit" if game == "farm" else "reference_score"] > 0


@pytest.fixture
def package_name():
    name = "_rsi_bundled_import_probe"
    previous = {key: value for key, value in sys.modules.items()
                if key == name or key.startswith(name + ".")}
    yield name
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    sys.modules.update(previous)


def test_vendor_package_retains_absolute_imports_without_exposing_siblings(tmp_path, package_name):
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text(f"from {package_name}.child import VALUE\n")
    (package / "child.py").write_text("VALUE = 42\n")
    before = list(sys.path)
    module = load_vendor_package(package_name, package)
    assert module.VALUE == 42
    assert list(module.__path__) == [str(package.resolve())]
    assert load_vendor_package(package_name, package) is module
    assert sys.path == before


def test_vendor_package_does_not_mix_with_an_existing_other_source(tmp_path, package_name, monkeypatch):
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 42\n")
    foreign = ModuleType(package_name)
    foreign.__file__ = str(tmp_path / "other" / "__init__.py")
    monkeypatch.setitem(sys.modules, package_name, foreign)
    with pytest.raises(ImportError, match="already loaded from"):
        load_vendor_package(package_name, package)
    assert sys.modules[package_name] is foreign


def test_failed_vendor_import_cleans_partial_modules_before_retry(tmp_path, package_name):
    package = tmp_path / package_name
    package.mkdir()
    initializer = package / "__init__.py"
    initializer.write_text(f"from {package_name}.child import VALUE\nraise RuntimeError('broken init')\n")
    (package / "child.py").write_text("VALUE = 42\n")
    with pytest.raises(RuntimeError, match="broken init"):
        load_vendor_package(package_name, package)
    assert not any(key == package_name or key.startswith(package_name + ".") for key in sys.modules)
    initializer.write_text(f"from {package_name}.child import VALUE\n")
    assert load_vendor_package(package_name, package).VALUE == 42


def test_legacy_path_bridge_preserves_precedence_and_is_idempotent(tmp_path, monkeypatch):
    candidate, upstream = tmp_path / "candidate", tmp_path / "upstream"
    candidate.mkdir()
    upstream.mkdir()
    monkeypatch.setattr(sys, "path", [str(upstream), "retained-entry", str(candidate), str(candidate)])
    prepend_legacy_paths(candidate, upstream)
    assert sys.path == [str(candidate), str(upstream), "retained-entry"]
    prepend_legacy_paths(candidate, upstream)
    assert sys.path == [str(candidate), str(upstream), "retained-entry"]


def test_missing_legacy_source_leaves_search_path_unchanged(tmp_path):
    before = list(sys.path)
    with pytest.raises(FileNotFoundError, match="Import source directory not found"):
        prepend_legacy_paths(tmp_path, tmp_path / "missing")
    assert sys.path == before


def test_isolated_worker_bridge_loads_candidate_tools_before_upstream(tmp_path):
    candidate, upstream = tmp_path / "candidate", tmp_path / "upstream"
    for directory, value in [(candidate, "candidate"), (upstream, "upstream")]:
        tools = directory / "tools"
        tools.mkdir(parents=True)
        (tools / "__init__.py").write_text(f"SOURCE = {value!r}\n")
    before = list(sys.path)
    result = run_python(
        "-c", "from pathlib import Path; import sys; "
        "from benchmark.imports import prepend_legacy_paths; "
        "prepend_legacy_paths(Path(sys.argv[1]), Path(sys.argv[2])); "
        "import tools; print(tools.SOURCE)", str(candidate), str(upstream),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "candidate"
    assert sys.path == before  # Worker paths never leak into the parent process.


def test_hero_runtime_loads_standalone_without_mutating_search_path():
    result = run_python(
        "-c", "import sys; before=list(sys.path); "
        "import benches.heros_journey.sequence_episode; "
        "import benches.heros_journey.benchmark; "
        "import benches.heros_journey.hero_journey_play; "
        "import herosjourney; "
        "assert herosjourney.get_task('override'); "
        "assert sys.path == before; print(herosjourney.__file__)",
    )
    assert result.returncode == 0, result.stderr
    assert "HerosJourney/herosjourney/__init__.py" in result.stdout


def test_hyperagents_host_import_does_not_expose_vendor_flat_modules():
    result = run_python(
        "-c", "import sys; before=list(sys.path); "
        "import rsi_methods.HyperAgents; "
        "from rsi_methods.HyperAgents.hyperagents.utils.constants import REPO_NAME; "
        "assert REPO_NAME; assert sys.path == before; "
        "assert 'agent' not in sys.modules; assert 'utils' not in sys.modules",
    )
    assert result.returncode == 0, result.stderr
