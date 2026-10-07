"""Automatically start a local HTTP server for game folders."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def append_url_suffix(base_url: str, suffix: str | None) -> str:
    """Append a task URL suffix.

    Catalog tasks should use explicit `?query=value` or `#fragment` forms.
    Bare query strings are still tolerated for ad hoc CLI overrides.
    """
    clean_suffix = str(suffix or "").strip()
    if not clean_suffix:
        return base_url

    if clean_suffix.startswith("#"):
        return f"{base_url}{clean_suffix}"

    if clean_suffix.startswith(("?", "&")):
        clean_suffix = clean_suffix[1:]

    if "=" in clean_suffix:
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}{clean_suffix}"

    return f"{base_url}{clean_suffix}"


class GameLauncher:
    """Manages HTTP server for serving game files."""

    DEFAULT_PORT = 8101
    DEFAULT_HTML = "index.html"
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_BASE_DIR = _REPO_ROOT / "games" / "benchmark"

    def __init__(
        self,
        game_name: str,
        port: Optional[int] = None,
        base_dir: Path | str | None = None,
        html_file: Optional[str] = None,
    ):
        """Initialize a launcher for one local game directory."""
        self.game_name = game_name
        self.port = port or self.DEFAULT_PORT
        self.html_file = html_file or self.DEFAULT_HTML
        self.process: Optional[subprocess.Popen] = None

        self.game_dir = self.resolve_game_directory(game_name, base_dir=base_dir)
        self.base_dir = Path(base_dir) if base_dir is not None else self.DEFAULT_BASE_DIR

    @classmethod
    def resolve_game_directory(
        cls,
        game_name: str,
        base_dir: Path | str | None = None,
    ) -> Path:
        """Resolve a game folder from an explicit base directory or the benchmark root."""
        root = Path(base_dir) if base_dir is not None else cls.DEFAULT_BASE_DIR
        return root / game_name

    def _ensure_port_available(self) -> None:
        """Fail fast if another process is already listening on the target port."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", int(self.port)))
        except OSError as exc:
            raise RuntimeError(
                f"Port {self.port} is already in use. "
                "Pass a different --port instead of terminating the existing process."
            ) from exc

    def _wait_until_listening(self, timeout_s: float | None = None) -> None:
        if timeout_s is None:
            timeout_s = float(os.environ.get("GAMEWORLD_SERVER_TIMEOUT", "90.0"))
        deadline = time.monotonic() + timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"HTTP server exited before listening on port {self.port} "
                    f"(exit_code={self.process.returncode})"
                )
            try:
                with socket.create_connection(("127.0.0.1", int(self.port)), timeout=0.5):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError(
            f"HTTP server did not start listening on port {self.port} within {timeout_s}s"
        ) from last_error

    def start(self) -> str:
        """
        Start the HTTP server for the game.

        Returns:
            URL to access the game (e.g., "http://127.0.0.1:8101/index.html")
        """
        if not self.game_dir.exists():
            raise FileNotFoundError(
                f"Game directory not found: {self.game_dir}. Base dir: {self.base_dir}"
            )

        LOGGER.info("Starting local game server: %s (port=%s)", self.game_name, self.port)

        self._ensure_port_available()

        self.process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(self.port), "--bind", "127.0.0.1"],
            cwd=self.game_dir,
            stdout=subprocess.DEVNULL,
            # Avoid deadlock from unconsumed http.server access logs on stderr.
            stderr=subprocess.DEVNULL,
        )

        if self.process.poll() is not None:
            raise RuntimeError(
                f"Failed to start HTTP server on port {self.port} "
                f"(exit_code={self.process.returncode})"
            )
        self._wait_until_listening()

        url = f"http://127.0.0.1:{self.port}/{self.html_file}"
        LOGGER.info("Game server started at %s", url)

        return url

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self.process:
            LOGGER.info("Stopping local game server for %s", self.game_name)
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
