"""Pin optional GameWorld experiments to the RSI-Bench CSD worktree.

The CSD branch reorganizes RSI-Bench into a flat checkout and adds AHE.  It is
kept separate from the pinned main checkout used by the existing ACE and
HyperAgents experiments.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_CSD_COMMIT = "2540eebae49ce75e4ebc74009105db4cd3087997"
DEFAULT_CSD_ROOT = Path("/mnt/shared-storage-user/yangjie/RSI-Bench-CSD")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_csd_checkout(
    root: str | Path | None = None,
    *,
    expected_commit: str = EXPECTED_CSD_COMMIT,
) -> dict[str, Any]:
    """Verify the detached CSD worktree without touching the main checkout."""

    resolved = Path(
        root or os.environ.get("RSI_BENCH_CSD_ROOT") or DEFAULT_CSD_ROOT
    ).expanduser().resolve()
    if not (resolved / ".git").exists():
        raise FileNotFoundError(f"RSI-Bench CSD Git worktree not found: {resolved}")
    if not (resolved / "rsi_methods").is_dir():
        raise FileNotFoundError(f"CSD RSI method sources not found: {resolved / 'rsi_methods'}")

    commit = _git(resolved, "rev-parse", "HEAD")
    if commit != expected_commit and os.environ.get("RSI_BENCH_CSD_ALLOW_UNPINNED") != "1":
        raise RuntimeError(
            "RSI-Bench CSD commit mismatch: "
            f"expected {expected_commit}, found {commit}"
        )

    status = _git(resolved, "status", "--porcelain", "--untracked-files=all")
    source_changes = [
        line
        for line in status.splitlines()
        if "/__pycache__/" not in line and not line.endswith((".pyc", ".pyo"))
    ]
    if source_changes:
        raise RuntimeError(
            "RSI-Bench CSD checkout has source changes; refusing a strict run: "
            + "; ".join(source_changes)
        )

    ahe_source = resolved / "rsi_methods" / "AHE" / "agentic-harness-engineering"
    return {
        "root": str(resolved),
        "commit": commit,
        "expected_commit": expected_commit,
        "commit_matches": commit == expected_commit,
        "dirty": False,
        "ahe_upstream_present": ahe_source.is_dir() and any(ahe_source.iterdir()),
        "ahe_upstream_path": str(ahe_source),
    }


def configure_csd_import_paths(
    *,
    code_root: str | Path,
    csd_root: str | Path | None = None,
) -> dict[str, Any]:
    """Put GameWorld and the CSD checkout on ``sys.path`` in that order."""

    provenance = verify_csd_checkout(csd_root)
    gameworld_path = str(Path(code_root).expanduser().resolve())
    csd_path = str(Path(provenance["root"]).resolve())
    legacy_path = str(Path(gameworld_path).parent / "rsi")
    sys.path[:] = [entry for entry in sys.path if entry not in {legacy_path, csd_path}]
    sys.path.insert(0, csd_path)
    sys.path.insert(0, gameworld_path)
    return provenance


def assert_ahe_imports(csd_root: str | Path | None = None) -> dict[str, str]:
    """Verify that AHE and its provider-neutral contracts resolve from CSD."""

    import importlib

    provenance = verify_csd_checkout(csd_root)
    root = Path(provenance["root"])
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    origins: dict[str, str] = {}
    for name in ("benchmark.contracts", "rsi_methods.AHE.AHEAdaptor"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(root):
            raise RuntimeError(f"{name} imported from {origin}, not {root}")
        origins[name] = str(origin)
    return origins


def require_ahe_upstream(provenance: dict[str, Any]) -> None:
    """Fail before an experiment if the CSD AHE gitlink was not materialized."""

    if provenance.get("ahe_upstream_present"):
        return
    raise RuntimeError(
        "CSD AHE upstream source is missing: "
        f"{provenance.get('ahe_upstream_path')}. "
        "The CSD commit contains a gitlink but this checkout has no materialized "
        "agentic-harness-engineering source; no AHE game score can be produced."
    )


__all__ = [
    "DEFAULT_CSD_ROOT",
    "EXPECTED_CSD_COMMIT",
    "assert_ahe_imports",
    "configure_csd_import_paths",
    "require_ahe_upstream",
    "verify_csd_checkout",
]
