"""Repackage the installed, RECORD-verified NexAU 0.3.9 without host artifacts."""

from __future__ import annotations

import base64
import hashlib
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


VERSION = "0.3.9"
BUILD = "1rsibench"
COMMIT = "35ee1861546db3cb280a6e17e38a74060d7c96c3"


def main() -> None:
    import json

    dist = distribution("nexau")
    if dist.version != VERSION:
        raise RuntimeError(f"Expected NexAU {VERSION}, found {dist.version}")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != COMMIT:
        raise RuntimeError("Installed NexAU does not record the expected source commit")
    metadata_dir = f"nexau-{VERSION}.dist-info"
    metadata_files = {
        f"{metadata_dir}/{name}"
        for name in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt")
    }
    copied = set()
    destination = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="nexau-repack-") as temporary:
        root = Path(temporary)
        for record in dist.files or ():
            name = str(record)
            if not (
                name.startswith("nexau/") or name in metadata_files
                or name.startswith(f"{metadata_dir}/licenses/")
            ):
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe distribution path: {name}")
            source = dist.locate_file(record)
            if source.is_symlink() or not record.hash:
                raise RuntimeError(f"Unverifiable distribution file: {name}")
            digest = base64.urlsafe_b64encode(
                hashlib.new(record.hash.mode, source.read_bytes()).digest(),
            ).decode().rstrip("=")
            if digest != record.hash.value:
                raise RuntimeError(f"Installed distribution file was modified: {name}")
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.add(name)
        required = metadata_files | {
            "nexau/__init__.py", f"{metadata_dir}/licenses/LICENSE",
        }
        if not required.issubset(copied):
            raise RuntimeError(f"Missing distribution files: {sorted(required - copied)}")
        subprocess.run([
            sys.executable, "-m", "wheel", "pack", str(root),
            "--build-number", BUILD, "--dest-dir", str(destination),
        ], check=True)


if __name__ == "__main__":
    main()
