"""Filesystem isolation for AHE model-driven processes (Linux bubblewrap).

Only runtime dependencies and explicit evidence/workspaces enter the namespace.
Networking stays enabled for inference; this is not a network security boundary.
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]


def reject_symlinks(directory: Path) -> None:
    """Do not let a copied/evolved harness dereference host files on the next build."""
    if directory.is_symlink() or any(path.is_symlink() for path in directory.rglob("*")):
        raise RuntimeError(f"AHE sandbox rejects symlinks in a harness: {directory}")


def sandbox_command(
    command: Sequence[str], *, readable: Sequence[Path] = (), writable: Sequence[Path] = (),
    cwd: Path, python_paths: Sequence[Path] = (),
) -> list[str]:
    executable = shutil.which("bwrap")
    if sys.platform != "linux" or executable is None:
        raise RuntimeError(
            "AHE sandbox requires Linux bubblewrap (bwrap) with user namespaces; "
            "install bubblewrap and enable namespaces. No unsandboxed fallback is allowed."
        )
    # Virtualenv .pth files can add dependencies outside sys.prefix. Do not expose
    # arbitrary sys.path entries such as the host working directory.
    package_paths = list(dict.fromkeys(
        Path(value).absolute() for value in sys.path
        if value and Path(value).name in {"site-packages", "dist-packages"}
        and Path(value).is_dir()
    ))
    args = [executable, "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--setenv", "HOME", "/tmp", "--setenv", "TMPDIR", "/tmp",
            "--setenv", "XDG_CACHE_HOME", "/tmp/.cache",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "PYTHONNOUSERSITE", "1",
            "--setenv", "PYTHONPATH", ":".join(
                str(path) for path in (ROOT, *python_paths, *package_paths)),
            "--unsetenv", "PYTHONHOME"]
    runtime = [Path(p) for p in (
        "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/ssl/certs", "/etc/pki",
        "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/localtime",
    )]
    for prefix in {Path(sys.prefix), Path(sys.base_prefix)}:
        runtime.extend((prefix / "bin", prefix / "lib", prefix / "lib64", prefix / "pyvenv.cfg"))
    runtime.extend(Path(value) for key, value in sysconfig.get_paths().items()
                   if key in {"stdlib", "platstdlib", "purelib", "platlib"})
    runtime.extend(package_paths)
    runtime.extend((ROOT / "benchmark", ROOT / "rsi_methods" / "AHE"))
    package_init = ROOT / "rsi_methods" / "__init__.py"
    if package_init.is_file():
        runtime.append(package_init)
    mounted: set[Path] = set()
    for path in [*runtime, *readable]:
        path = path.absolute()
        if path.exists() and path not in mounted:
            args.extend(("--ro-bind", str(path.resolve()), str(path)))
            mounted.add(path)
    for path in writable:
        path = path.absolute()
        if path.is_symlink() or not path.exists():
            raise RuntimeError(f"AHE sandbox writable path must exist and not be a symlink: {path}")
        args.extend(("--bind", str(path.resolve()), str(path)))
    return [*args, "--chdir", str(cwd), "--", *command]


def training_history_paths(run_dir: Path) -> list[Path]:
    """Expose only completed training evidence, never the entire run or holdout."""
    import json

    paths = [run_dir / name for name in ("evolution_history.json", "evolution_history.md", "method_config.toml")]
    history = run_dir / "evolution_history.json"
    if history.is_symlink() or not history.resolve().is_relative_to(run_dir.resolve()):
        raise RuntimeError("AHE sandbox history path escapes the run directory")
    records = json.loads(history.read_text(encoding="utf-8")) if history.is_file() else []
    for record in records:
        relative = record.get("change_evaluation")
        if not relative:
            continue
        path = run_dir / relative
        if not path.resolve().is_relative_to(run_dir.resolve()) or path.is_symlink():
            raise RuntimeError("AHE sandbox history path escapes the run directory")
        paths.append(path)
        # Manifests are evidence, not writable candidate code or evaluation state.
        paths.extend(path.parent.glob("variant_*/change_manifest.json"))
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()):
            raise RuntimeError("AHE sandbox history path escapes the run directory")
    return [path for path in paths if path.is_file()]
