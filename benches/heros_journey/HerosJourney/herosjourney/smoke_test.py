"""Small registration and generation smoke test for Hero's Journey tasks."""

from __future__ import annotations

import json
from pathlib import Path

from herosjourney import get_task
from herosjourney.core.elements import fill_elements, load_lexicons


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "difficulty_manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    difficulties = manifest["difficulties"]
    expected = {
        "2": ("proc_chain_deep", 6, 16, 2),
        "3": ("proc_chain_dual", 6, 16, 3),
        "4": ("proc_chain_nested", 6, 16, 4),
        "5": ("proc_chain_quint", 6, 16, 5),
    }

    sem_lex, nonce_lex = load_lexicons()
    for level, (task_type, expected_source, expected_gen, expected_blocks) in expected.items():
        assert difficulties[level]["task_type"] == task_type
        if expected_source is not None:
            assert difficulties[level]["logic_version"] == "ordered_extensions_v1"
        spec = get_task(task_type)
        rule = json.loads(Path(spec.rules).read_text(encoding="utf-8"))
        elements = fill_elements(
            rule, sem_lex, nonce_lex, seed=int(level), split_spec=spec.split
        )
        source = spec.gen_fn(elements, split="source", use_nonce=False)
        generated = spec.gen_fn(elements, split="gen", use_nonce=False)
        assert source and generated
        if expected_source is not None:
            assert len(source) == expected_source
            assert len(generated) == expected_gen
            logic = elements["_variant_info"]["seeded_logic"]
            assert logic["block_count"] == expected_blocks
            assert len(logic["templates"]) == 4
            assert len(elements["_extension_specs"]) == expected_blocks
            assert len(logic["fingerprint"]) == 64
        print(f"PASS d{level}: {task_type} source={len(source)} gen={len(generated)}")


if __name__ == "__main__":
    main()
