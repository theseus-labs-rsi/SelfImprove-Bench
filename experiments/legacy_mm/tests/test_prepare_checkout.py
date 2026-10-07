import hashlib
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_checkout", Path(__file__).resolve().parents[1] / "tools/prepare_checkout.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def checkout(root):
    templates = root / "configs/templates"
    templates.mkdir(parents=True)
    (templates / "test.json").write_text(
        '{"output_dir":"${RSI_ROOT}/runs/test",'
        '"nested":[{"gameworld_root":"${RSI_ROOT}/gameworld"}]}'
    )
    (templates / "test.toml").write_text(
        '[state]\nworkspace_root = "${RSI_ROOT}/runs/state"\n'
    )
    provenance = root / "provenance"
    provenance.mkdir()
    manifest = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in templates.iterdir()
    }
    (provenance / "source_sha256.json").write_text(json.dumps(manifest))
    return templates


def test_resolve_templates_and_seal(tmp_path):
    root = tmp_path / 'path with spaces "and quotes"'
    checkout(root)
    prepare.prepare(root)
    config = json.loads((root / "configs/test.json").read_text())
    assert config["output_dir"] == str(root / "runs/test")
    assert config["nested"][0]["gameworld_root"] == str(root / "gameworld")
    settings = tomllib.loads((root / "configs/test.toml").read_text())
    assert settings["state"]["workspace_root"] == str(root / "runs/state")
    seal = json.loads((root / "provenance/release_sha256.json").read_text())
    assert "configs/test.json" in seal and "configs/templates/test.json" in seal
    for relative, expected in seal.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


def test_changed_source_rejected(tmp_path):
    templates = checkout(tmp_path)
    (templates / "test.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Published source changed"):
        prepare.prepare(tmp_path)
    assert not (tmp_path / "configs/test.json").exists()


def test_existing_checkpoint_rejected(tmp_path):
    checkout(tmp_path)
    checkpoint = tmp_path / "runs/astray/ace/lineage/checkpoints/train_0001.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    with pytest.raises(RuntimeError, match="Existing checkpoints"):
        prepare.prepare(tmp_path)
    assert not (tmp_path / "configs/test.json").exists()
