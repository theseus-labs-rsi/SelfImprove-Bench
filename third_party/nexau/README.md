# Bundled NexAU Runtime

This directory supplies the fixed AHE runtime without requiring access to the
upstream Git repository.

- Package: `nexau`, version `0.3.9`.
- Recorded upstream: `https://github.com/nex-agi/NexAU.git`.
- Recorded revision: `v0.3.9`.
- Recorded commit: `35ee1861546db3cb280a6e17e38a74060d7c96c3`.
- Artifact: `nexau-0.3.9-1rsibench-py3-none-any.whl`.
- SHA-256: `8419853b0a756e68aa74ca09fb9c23c4e6c3871fc4e332499768f6f85d8885d6`.
- License: [Apache-2.0](LICENSE). The original license is also included in the wheel at
  `nexau-0.3.9.dist-info/licenses/LICENSE`.

## Provenance

This is an RSI-Bench repack of an existing installed distribution, not an
official upstream wheel or a fresh build independently verified against Git.
The source commit and revision above come from its installation metadata.
All 211 package source/resource files matched their installed RECORD hashes
before repacking. Their contents are unchanged.

The repack preserves package metadata, entry points, resources, and license
files. It regenerates the wheel RECORD and adds build tag `1rsibench`.
It omits generated launcher scripts, bytecode, caches, installer-specific
metadata, local configuration, and installation paths. No separate NOTICE file
was present in the installed distribution's file manifest.

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

The wheel preserves upstream dependency declarations. In particular, keep
`ag-ui-protocol==0.1.22` from the root requirements for AHE compatibility.
Other dependencies still require normal package-index access.

## Repacking

`repack.py` requires Python with `wheel` and the original Git-installed NexAU
distribution. It validates the recorded version, commit and file hashes before
copying an allowlist of distribution files and invoking `python -m wheel pack`.
It deliberately rejects modified installed files; it is not needed for normal
installation or benchmark execution.

```bash
python third_party/nexau/repack.py
```
