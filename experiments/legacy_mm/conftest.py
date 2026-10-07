"""Offline harness for both new tests and selected unmodified legacy tests."""
import importlib
import re
import socket

import pytest


@pytest.fixture(autouse=True)
def offline_and_isolated(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network requests are forbidden in this offline suite")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-not-a-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    # Some legacy tests copy this default config and initialize a local workspace.
    # Keep those writes in pytest temp space, never under experiment runs/.
    module = importlib.import_module("rsi_methods.ACE.ACEAdaptor")
    config = tmp_path / "isolated_ace.toml"
    config.write_text(re.sub(
        r'^workspace_root\s*=.*$', f'workspace_root = "{tmp_path / "ace_state"}"',
        module.CONFIG_PATH.read_text(), flags=re.MULTILINE))
    monkeypatch.setattr(module, "CONFIG_PATH", config)
