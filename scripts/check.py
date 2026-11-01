"""Preflight, focused static checks and the first-party model-free test suite.

Never installs dependencies, downloads resources, starts Docker, or calls models.
Use docs/engineering.md for explicit preparation and OS-level network isolation.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from .test_assets import default_cache_dir, verified_tokenizer

ROOT = Path(__file__).resolve().parents[1]


VENDORED = (
    "rsi_methods/ACE/ace", "rsi_methods/AHE/agentic-harness-engineering",
    "rsi_methods/HyperAgents/hyperagents", "benches/heros_journey/HerosJourney",
    "benches/gameworld/assets", "__pycache__",
)


def test_scope(root: Path = ROOT) -> list[str]:
    return ["tests", "benches/farm/test_farm_planner.py", "benches/combat/test_combat_planner.py",
            "benches/gameworld/test_network_retry.py", *[
                str(path.relative_to(root)) for path in sorted((root / "rsi_methods/HyperAgents").glob("test_*.py"))]]


def lint_scope(root: Path = ROOT) -> list[str]:
    files = []
    for directory in ("benchmark", "analyze", "examples", "tests", "scripts", "benches", "rsi_methods"):
        for path in (root / directory).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if not any(relative == prefix or relative.startswith(prefix + "/") for prefix in VENDORED):
                files.append(relative)
    return sorted(files)


def offline_environment(cache: Path) -> dict[str, str]:
    # Do not pass model credentials, proxy configuration or user pytest plugins.
    keep = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "TMP", "TEMP",
            "VIRTUAL_ENV", "LD_LIBRARY_PATH", "SYSTEMROOT"}
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", RSI_OFFLINE_CHECK="1",
               TIKTOKEN_CACHE_DIR=str(cache.resolve()),
               PYTHONPATH=os.pathsep.join((str(ROOT / "scripts/offline_guard"), str(ROOT))))
    return env


def pinned_dependencies(root: Path = ROOT) -> dict[str, str]:
    pins = {"nexau": "0.3.9"}
    for filename in ("requirements.txt", "requirements-dev.txt"):
        for line in (root / filename).read_text().splitlines():
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line.strip())
            if match:
                pins[match[1]] = match[2]
    return pins


def preflight(cache: Path, *, require_namespaces: bool = False) -> dict:
    failures = []
    if sys.version_info[:2] != (3, 13):
        failures.append("Use Python 3.13 (reference container: 3.13.15).")
    if sys.platform != "linux":
        failures.append("The reference check environment is Linux; use the documented container on other hosts.")
    installed = {}
    for name, expected in pinned_dependencies().items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        installed[name] = {"expected": expected, "installed": actual}
        if actual != expected:
            failures.append(f"{name}: expected {expected}, found {actual}; install requirements-dev.txt.")
    if not shutil.which("git"):
        failures.append("git is required by local reconstruction tests.")
    if not verified_tokenizer(cache):
        failures.append(f"Missing/corrupt tokenizer cache at {cache}. Run python -m scripts.test_assets with "
                        "--cache-dir set to this path and --source-dir CACHE or --download explicitly.")
    namespace = {"available": False, "reason": "Linux bubblewrap is not installed"}
    if sys.platform == "linux" and shutil.which("bwrap"):
        try:
            probe = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-user", "/bin/true"],
                                   capture_output=True, text=True, timeout=10)
            namespace = {"available": probe.returncode == 0, "reason": probe.stderr.strip()[:2000]}
        except (OSError, subprocess.TimeoutExpired) as error:
            namespace["reason"] = str(error)
    if require_namespaces and not namespace["available"]:
        failures.append("--require-namespaces requested, but the bubblewrap namespace probe failed.")
    return {"python": sys.version, "platform": platform.platform(), "dependencies": installed,
            "tokenizer_cache": str(cache.resolve()), "tokenizer_verified": verified_tokenizer(cache),
            "namespace": namespace, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/checks")
    parser.add_argument("--tokenizer-cache", type=Path, default=default_cache_dir())
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help="Run preflight and static checks only")
    parser.add_argument("--require-namespaces", action="store_true", help="Fail instead of allowing conditional namespace skips")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"preflight": preflight(args.tokenizer_cache, require_namespaces=args.require_namespaces),
              "test_scope": test_scope(), "checks": []}
    def save():
        (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    if report["preflight"]["failures"]:
        for message in report["preflight"]["failures"]:
            print(message, file=sys.stderr)
        print(f"Preflight failed; details: {output / 'checks.json'}", file=sys.stderr)
        return 2
    if not report["preflight"]["namespace"]["available"]:
        print("Namespace integration tests may skip: " + report["preflight"]["namespace"]["reason"], flush=True)
    if args.preflight_only:
        print(f"Preflight passed: {output / 'checks.json'}")
        return 0
    env = offline_environment(args.tokenizer_cache)
    commands = [
        ("pip-check", [sys.executable, "-m", "pip", "check"]),
        ("pip-freeze", [sys.executable, "-m", "pip", "freeze"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "--no-cache", *lint_scope()]),
        ("score-contract-types", [sys.executable, "-m", "mypy", "--config-file", "mypy.ini", "--no-incremental"]),
    ]
    if not args.skip_tests:
        commands.append(("pytest", [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-ra",
                                   f"--junitxml={output / 'pytest.xml'}", *test_scope()]))
    for name, command in commands:
        print(f"Running {name}...", flush=True)
        start = time.monotonic()
        with (output / f"{name}.log").open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "command": command, "returncode": completed.returncode,
                                 "seconds": time.monotonic() - start, "log": f"{name}.log"})
        save()
        if completed.returncode:
            print(f"{name} failed; see {output / (name + '.log')}", file=sys.stderr)
            return 1
    print(f"Checks passed. Read pytest.log / pytest.xml for conditional skips. Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
