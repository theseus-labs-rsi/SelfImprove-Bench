"""Create, reconstruct, and persist versioned HyperDSH harness snapshots."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import PROTECTED_PROFILE_IDS, SAFE_OPTIONAL_TOOLS


PACKAGE_ROOT = Path(__file__).resolve().parent
HYPERAGENTS_SEED = PACKAGE_ROOT.parent / "HyperAgents" / "hyperagents"
HARNESS_SEED = PACKAGE_ROOT / "seed_harness"
OPTIMIZER_SEED = PACKAGE_ROOT / "optimizer_seed"

_RUNTIME_DOCKERFILE = """FROM python:3.12-bookworm

WORKDIR /hyperagents

RUN pip install --no-cache-dir \\
    requests==2.32.4 \\
    python-dotenv==1.2.3 \\
    backoff==2.2.1 \\
    openai==3.2.0 \\
    GitPython==3.1.44

CMD ["tail", "-f", "/dev/null"]
"""
_CANDIDATE_DOCKERFILE = """FROM hyperagents-runtime:py312-v2

WORKDIR /hyperagents
COPY . .

CMD ["tail", "-f", "/dev/null"]
"""


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_id: str | int
    run_dir: Path
    workspace: Path
    root_commit: str
    parent_id: str | int | None = None
    lineage_patches: tuple[Path, ...] = ()


def with_lineage(
    initial: CandidateSnapshot,
    *,
    candidate_id: str | int,
    parent_id: str | int | None,
    lineage_patches: Iterable[str | Path],
) -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id=candidate_id,
        run_dir=initial.run_dir,
        workspace=initial.workspace,
        root_commit=initial.root_commit,
        parent_id=parent_id,
        lineage_patches=tuple(Path(path).resolve() for path in lineage_patches),
    )


def initialize_hyperdsh(run_dir: str | Path) -> CandidateSnapshot:
    """Create a clean lineage from fixed optimizer code and immutable harness seed."""

    resolved = Path(run_dir).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"HyperDSH run directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    bootstrap = resolved / "bootstrap_source"
    shutil.copytree(
        HYPERAGENTS_SEED,
        bootstrap,
        ignore=shutil.ignore_patterns(
            ".git", ".claude", "outputs", "analysis", "misc", "baselines",
            "domains", "__pycache__", "*.pyc", "*.png", "outputs_os*",
        ),
    )
    for source in OPTIMIZER_SEED.iterdir():
        if source.is_file():
            shutil.copy2(source, bootstrap / source.name)
    shutil.copytree(HARNESS_SEED, bootstrap / "harness")

    workspace = resolved / "gen_initial" / "hyperdsh"
    workspace.parent.mkdir(parents=True)
    shutil.copytree(bootstrap, workspace)
    _copy_benchmark_contracts(workspace)
    (workspace / "Dockerfile").write_text(_CANDIDATE_DOCKERFILE, encoding="utf-8")
    (workspace / "Dockerfile.runtime").write_text(
        _RUNTIME_DOCKERFILE, encoding="utf-8"
    )
    (resolved / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "hyperdsh_profile_evolution",
                "immutable_seed": str(HARNESS_SEED),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _git(workspace, "init")
    _git(workspace, "add", "--all")
    _git(
        workspace,
        "-c",
        "user.name=HyperDSH",
        "-c",
        "user.email=hyperdsh@local",
        "commit",
        "-m",
        "Initialize immutable HyperDSH seed",
    )
    root_commit = _git_output(workspace, "rev-parse", "HEAD")
    snapshot = CandidateSnapshot(
        candidate_id="initial",
        run_dir=resolved,
        workspace=workspace,
        root_commit=root_commit,
    )
    persist_version(snapshot, resolved / "versions" / "generation_0000_initial")
    _write_snapshot(resolved / "initial_snapshot.json", snapshot)
    _write_snapshot(resolved / "active_snapshot.json", snapshot)
    return snapshot


def materialize_snapshot(snapshot: CandidateSnapshot, destination: str | Path) -> Path:
    """Reconstruct one exact candidate in a new host-side directory."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Snapshot destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot.workspace, target, symlinks=True)
    _git(target, "reset", "--hard", snapshot.root_commit)
    _git(target, "clean", "-fd")
    for patch_path in snapshot.lineage_patches:
        patch = Path(patch_path).resolve()
        if not patch.is_file():
            raise FileNotFoundError(f"HyperDSH lineage patch is missing: {patch}")
        _git(target, "apply", "--check", str(patch))
        _git(target, "apply", str(patch))
    return target


def persist_version(snapshot: CandidateSnapshot, destination: str | Path) -> Path:
    """Save a complete, immutable-looking harness copy and its lineage manifest."""

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"HyperDSH version already exists: {target}")
    target.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="hyperdsh-materialize-", dir=target.parent) as temp:
        materialized = materialize_snapshot(snapshot, Path(temp) / "candidate")
        harness = materialized / "harness"
        validate_harness(harness)
        shutil.copytree(harness, target / "harness")
    manifest = {
        "schema_version": 1,
        "candidate_id": str(snapshot.candidate_id),
        "parent_id": str(snapshot.parent_id) if snapshot.parent_id is not None else None,
        "root_commit": snapshot.root_commit,
        "lineage_patches": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(Path(path)),
            }
            for path in snapshot.lineage_patches
        ],
        "harness_tree_sha256": tree_digest(target / "harness"),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def validate_harness(harness: str | Path) -> dict[str, Any]:
    """Reject malformed or capability-expanding candidate control files."""

    root = Path(harness).resolve()
    required = (
        root / "prompt.md",
        root / "tools.json",
        root / "profile" / "package.json",
        root / "profile" / "cordis.patch.yml",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"HyperDSH candidate is missing required files: {missing}")
    symlinks = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"HyperDSH candidate must not contain symlinks: {symlinks}")
    if not (root / "prompt.md").read_text(encoding="utf-8").strip():
        raise ValueError("HyperDSH prompt.md must not be empty")

    tools = json.loads((root / "tools.json").read_text(encoding="utf-8"))
    enabled = tools.get("enabled") if isinstance(tools, dict) else None
    if not isinstance(enabled, list) or any(not isinstance(item, str) for item in enabled):
        raise ValueError("HyperDSH tools.json enabled must be a list of strings")
    if len(enabled) != len(set(enabled)):
        raise ValueError("HyperDSH tools.json enabled contains duplicate ids")
    unsupported = set(enabled) - SAFE_OPTIONAL_TOOLS
    if unsupported:
        raise ValueError(f"HyperDSH tools.json enables unsupported tools: {sorted(unsupported)}")

    manifest = json.loads(
        (root / "profile" / "package.json").read_text(encoding="utf-8")
    )
    bundles = manifest.get("dsh", {}).get("profile", {}).get("bundles")
    expected = ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-headless"]
    if bundles != expected:
        raise ValueError(f"HyperDSH profile bundles must remain exactly {expected}")
    _validate_profile_patch(root / "profile" / "cordis.patch.yml")
    return tools


def _validate_profile_patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    forbidden_syntax = ("!!js", " name:", "- insert:", "- remove:")
    found = [token.strip() for token in forbidden_syntax if token in lowered]
    if found:
        raise ValueError(
            f"HyperDSH profile patch uses forbidden composition syntax: {found}"
        )
    for match in re.finditer(
        r"(?m)^\s*-\s*id:\s*['\"]?([^'\"\s#]+)", text
    ):
        row_id = match.group(1)
        if row_id in PROTECTED_PROFILE_IDS:
            raise ValueError(f"HyperDSH profile patch targets protected row: {row_id}")


def validate_harness_patch(patch_path: str | Path) -> None:
    """Ensure every path in a MetaAgent-produced patch stays below harness/."""

    path = Path(patch_path)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            fields = line.split()
            changed = [item[2:] for item in fields[2:4] if item.startswith(("a/", "b/"))]
        elif line.startswith(("--- a/", "+++ b/")):
            changed = [line[6:]]
        else:
            continue
        if any(not item.startswith("harness/") for item in changed):
            raise RuntimeError(f"HyperDSH refinement escaped harness/: {changed}")


def snapshot_dict(snapshot: CandidateSnapshot) -> dict[str, Any]:
    return {
        "candidate_id": str(snapshot.candidate_id),
        "parent_id": str(snapshot.parent_id) if snapshot.parent_id is not None else None,
        "run_dir": str(snapshot.run_dir),
        "workspace": str(snapshot.workspace),
        "root_commit": snapshot.root_commit,
        "lineage_patches": [str(Path(path).resolve()) for path in snapshot.lineage_patches],
    }


def write_active_snapshot(run_dir: str | Path, snapshot: CandidateSnapshot) -> None:
    _write_snapshot(Path(run_dir).resolve() / "active_snapshot.json", snapshot)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_snapshot(path: Path, snapshot: CandidateSnapshot) -> None:
    path.write_text(
        json.dumps(snapshot_dict(snapshot), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git(repository: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _copy_benchmark_contracts(workspace: Path) -> None:
    project_root = PACKAGE_ROOT.parents[1]
    source = project_root / "benchmark"
    destination = workspace / "benchmark"
    backend_destination = destination / "backend"
    backend_destination.mkdir(parents=True)
    for filename in ("__init__.py", "Backend.py", "OpenAIBackend.py"):
        shutil.copy2(source / "backend" / filename, backend_destination / filename)
    shutil.copy2(source / "contracts.py", destination / "contracts.py")
    (destination / "__init__.py").touch(exist_ok=True)


__all__ = [
    "initialize_hyperdsh",
    "CandidateSnapshot",
    "materialize_snapshot",
    "persist_version",
    "snapshot_dict",
    "tree_digest",
    "validate_harness",
    "validate_harness_patch",
    "write_active_snapshot",
    "with_lineage",
]
