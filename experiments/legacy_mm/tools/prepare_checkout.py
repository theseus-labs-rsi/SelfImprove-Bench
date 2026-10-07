"""Resolve portable templates and seal a fresh checkout; never migrate old runs."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(value, root):
    if isinstance(value, dict):
        return {k: resolve(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, root) for v in value]
    if isinstance(value, str):
        return value.replace("${RSI_ROOT}", str(root))
    return value


def prepare(root=ROOT):
    if any((root / "runs").rglob("checkpoints/*.json")):
        raise RuntimeError("Existing checkpoints found; do not regenerate a running experiment.")
    manifest = json.loads((root / "provenance/source_sha256.json").read_text())
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Published source changed: {relative}")
    generated = []
    for template in sorted((root / "configs/templates").iterdir()):
        target = root / "configs" / template.name
        if template.suffix == ".json":
            target.write_text(json.dumps(resolve(json.loads(template.read_text()), root),
                                         ensure_ascii=False, indent=2) + "\n")
        elif template.suffix == ".toml":
            text = template.read_text()
            # JSON escaping also gives a valid TOML basic string for POSIX paths.
            text = re.sub(r'"(\$\{RSI_ROOT\}[^"]*)"',
                          lambda m: json.dumps(resolve(m.group(1), root)), text)
            target.write_text(text)
        else:
            continue
        generated.append(target.relative_to(root).as_posix())
    release = dict(manifest)
    for relative in generated:
        release[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    (root / "provenance/release_sha256.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(f"Prepared {len(generated)} configurations at {root}; no API calls.")


if __name__ == "__main__":
    prepare()
