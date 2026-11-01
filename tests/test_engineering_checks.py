"""Check the check tooling without downloads, model calls or Docker."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from scripts import check, test_assets


def test_offline_scope_includes_new_first_party_tests_without_vendor_discovery(tmp_path):
    wrappers = tmp_path / "rsi_methods/HyperAgents"
    wrappers.mkdir(parents=True)
    (wrappers / "test_new.py").touch()
    vendor = wrappers / "hyperagents/tests"
    vendor.mkdir(parents=True)
    (vendor / "test_vendor.py").touch()
    scope = check.test_scope(tmp_path)
    assert scope[0] == "tests"
    assert "rsi_methods/HyperAgents/test_new.py" in scope
    assert not any("test_vendor" in entry for entry in scope)


def test_asset_verification_never_downloads_implicitly(tmp_path):
    with patch.object(test_assets, "urlopen") as download:
        with pytest.raises(ValueError, match="explicitly --download"):
            test_assets.prepare_tokenizer(tmp_path)
    download.assert_not_called()


def test_asset_bad_source_is_not_copied_or_replaced_by_network(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / test_assets.TOKENIZER_CACHE_KEY).write_bytes(b"corrupt")
    target = tmp_path / "destination"
    with patch.object(test_assets, "urlopen") as download:
        with pytest.raises(ValueError, match="SHA-256"):
            test_assets.prepare_tokenizer(target, source_dir=source)
    assert not target.exists()
    download.assert_not_called()


def test_offline_environment_drops_model_credentials_and_proxy(tmp_path):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic", "HTTPS_PROXY": "http://proxy", "CUSTOM_MODEL_SECRET": "synthetic"}):
        env = check.offline_environment(tmp_path)
    assert "OPENAI_API_KEY" not in env and "HTTPS_PROXY" not in env and "CUSTOM_MODEL_SECRET" not in env
    assert env["RSI_OFFLINE_CHECK"] == env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_python_network_guard_blocks_connect_and_dns(tmp_path):
    script = """
import socket
for operation in [lambda: socket.getaddrinfo('example.invalid', 443),
                  lambda: socket.socket().connect(('127.0.0.1', 9))]:
    try:
        operation()
    except OSError as error:
        assert 'Network access is disabled' in str(error), str(error)
    else:
        raise AssertionError('network was not blocked')
print('blocked')
"""
    result = subprocess.run([sys.executable, "-c", script], env=check.offline_environment(tmp_path),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "blocked"


def test_lint_scope_excludes_vendors_but_checks_wrappers(tmp_path):
    wrapper = tmp_path / "rsi_methods/HyperAgents/wrapper.py"
    vendor = tmp_path / "rsi_methods/HyperAgents/hyperagents/vendor.py"
    wrapper.parent.mkdir(parents=True)
    vendor.parent.mkdir()
    wrapper.touch()
    vendor.touch()
    assert check.lint_scope(tmp_path) == ["rsi_methods/HyperAgents/wrapper.py"]
