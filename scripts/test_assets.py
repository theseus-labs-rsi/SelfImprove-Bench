"""Explicit, checksum-verified preparation of the offline tokenizer resource.

The default only verifies. Network access requires --download; copying an
existing verified cache with --source-dir needs no network or model credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
# This is the filename used by tiktoken.load.read_file_cached for the URL above.
TOKENIZER_CACHE_KEY = hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()


def default_cache_dir() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return Path(os.environ.get("TIKTOKEN_CACHE_DIR", cache_root / "rsi-bench/tiktoken"))


def verified_tokenizer(cache_dir: Path) -> bool:
    target = cache_dir / TOKENIZER_CACHE_KEY
    try:
        return target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == TOKENIZER_SHA256
    except OSError:
        return False


def prepare_tokenizer(cache_dir: Path, *, source_dir: Path | None = None, download: bool = False) -> Path:
    target = cache_dir / TOKENIZER_CACHE_KEY
    if verified_tokenizer(cache_dir):
        return target
    if source_dir is not None:
        if not verified_tokenizer(source_dir):
            raise ValueError(f"Source tokenizer cache is missing or has a wrong SHA-256: {source_dir}")
        data = (source_dir / TOKENIZER_CACHE_KEY).read_bytes()
    elif download:
        # Public tokenizer data only. No model endpoint, API key, or inference.
        with urlopen(TOKENIZER_URL, timeout=30) as response:
            data = response.read()
    else:
        raise ValueError("Tokenizer cache missing/corrupt. Use --source-dir CACHE or explicitly --download.")
    if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer SHA-256 mismatch; nothing installed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-dir", type=Path, help="Copy an existing checksum-verified tiktoken cache")
    source.add_argument("--download", action="store_true", help="Explicitly download the public tokenizer data")
    args = parser.parse_args()
    try:
        target = prepare_tokenizer(args.cache_dir, source_dir=args.source_dir, download=args.download)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Tokenizer preparation failed: {error}\n")
    print(f"Verified tokenizer: {target} (SHA-256 {TOKENIZER_SHA256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
