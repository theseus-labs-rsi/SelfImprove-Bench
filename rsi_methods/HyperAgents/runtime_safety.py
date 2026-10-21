"""Parent-owned, auditable transport fixes for the pinned HyperAgents source.

The upstream gitlink is intentionally left untouched. Runtime fixes are installed
on the seed before applying lineage patches. Evolved tools are never overwritten.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil


_OVERLAY = Path(__file__).parent / "runtime_overlay"
_ORIGINAL_SHA256 = {
    "agent/llm_withtools.py": "9dd804865516bd846922e007f5c2b1f933c7fec8f11786d55efe0981a44e63f3",
    "agent/tools/bash.py": "489c1cac70546f53375f1ec03e685d18f45f30532a06954f9b95ce3b0ff51b1e",
}
# The release bundles an already-adapted base, distinct from the upstream pin.
# Only these exact audited files may receive the same runtime safety overlay.
_BUNDLED_SHA256 = {
    "agent/llm_withtools.py": "11f761662184a56d04d0245b4f1bf3c3934e7cf0e31f8e44a074532767180f86",
    "agent/tools/bash.py": "2a20313afed0ffa5fd5c2c34312ddb0e9ad0b4d8952618145efe1adb20cd0867",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_runtime_safety(root_dir: str | Path) -> None:
    """Install fixed modules before committing a freshly seeded candidate."""
    root = Path(root_dir)
    for relative, original in _ORIGINAL_SHA256.items():
        destination = root / relative
        overlay = _OVERLAY / relative
        observed = _sha256(destination)
        if observed not in (original, _BUNDLED_SHA256[relative], _sha256(overlay)):
            raise RuntimeError(f"Unexpected HyperAgents runtime version: {relative} ({observed})")
        shutil.copy2(overlay, destination)


def install_runtime_safety(container, commit_hash: str, repo_path: str) -> str:
    """Install the audited seed baseline, before reconstructing evolution."""
    from .hyperagents.utils.docker_utils import copy_to_container

    changed = False
    for relative, original in _ORIGINAL_SHA256.items():
        overlay = _OVERLAY / relative
        current = container.exec_run(["sha256sum", f"{repo_path}/{relative}"], workdir=repo_path)
        if current.exit_code != 0:
            raise RuntimeError(f"Cannot verify HyperAgents runtime: {relative}")
        observed = current.output.decode("utf-8", errors="replace").split()[0]
        fixed = _sha256(overlay)
        if observed == fixed:
            continue
        if observed not in (original, _BUNDLED_SHA256[relative]):
            raise RuntimeError(f"Refusing to overwrite evolved runtime module: {relative} ({observed})")
        copy_to_container(container, overlay, f"{repo_path}/{relative}")
        changed = True
    if not changed:
        return commit_hash
    paths = list(_ORIGINAL_SHA256)
    for description, command in (
        ("stage runtime safety", ["git", "add", "--", *paths]),
        ("commit runtime safety", ["git", "-c", "user.name=user", "-c", "user.email=you@example.com",
                                   "commit", "-m", "Harness runtime safety overlay", "--", *paths]),
        ("read runtime safety commit", ["git", "rev-parse", "HEAD"]),
    ):
        result = container.exec_run(command, workdir=repo_path)
        if result.exit_code != 0:
            raise RuntimeError(f"{description} failed: {result.output.decode('utf-8', errors='replace')}")
    return result.output.decode("utf-8", errors="replace").strip()
