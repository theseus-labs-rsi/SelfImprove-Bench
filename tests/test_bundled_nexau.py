"""Check the redistributed runtime without importing or executing it."""

import base64
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = "nexau-0.3.9-1rsibench-py3-none-any.whl"
METADATA = "nexau-0.3.9.dist-info"


def test_bundled_nexau_is_the_configured_dependency():
    requirement = f"./third_party/nexau/{WHEEL_NAME}"
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    assert requirement in lines
    assert not any("git+https://github.com/nex-agi/NexAU" in line for line in lines)


def test_bundled_nexau_contents_and_record():
    with ZipFile(ROOT / "third_party/nexau" / WHEEL_NAME) as archive:
        names = archive.namelist()
        assert len(set(names)) == len(names)
        assert sum(name.startswith("nexau/") for name in names) == 211
        assert b"Version: 0.3.9\n" in archive.read(f"{METADATA}/METADATA")
        assert b"Build: 1rsibench\n" in archive.read(f"{METADATA}/WHEEL")
        assert b"Apache License" in archive.read(f"{METADATA}/licenses/LICENSE")
        assert f"{METADATA}/entry_points.txt" in names
        for name in names:
            assert name.startswith(("nexau/", f"{METADATA}/"))
            assert not {"..", "__pycache__"}.intersection(Path(name).parts)
            assert not name.endswith((".pyc", ".env", "direct_url.json", "uv_build.json"))
        records = list(csv.reader(io.StringIO(
            archive.read(f"{METADATA}/RECORD").decode(),
        )))
        assert {row[0] for row in records} == set(names)
        for name, digest, size in records:
            if name == f"{METADATA}/RECORD":
                assert digest == size == ""
                continue
            data = archive.read(name)
            algorithm, encoded = digest.split("=", 1)
            actual = base64.urlsafe_b64encode(
                hashlib.new(algorithm, data).digest(),
            ).decode().rstrip("=")
            assert actual == encoded
            assert len(data) == int(size)
