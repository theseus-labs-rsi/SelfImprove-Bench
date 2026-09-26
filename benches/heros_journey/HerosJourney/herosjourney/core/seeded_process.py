"""Seeded, identifiable hidden logic for procedural Hero tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from itertools import permutations
from typing import Any, Callable, Dict, List, Sequence, Tuple


def _consistent_inputs(
    rows: Sequence[Sequence[int]], values: Sequence[int], names: Sequence[str]
) -> List[str]:
    consistent: List[str] = []
    for column, name in enumerate(names):
        observed: Dict[int, int] = {}
        for row, value in zip(rows, values):
            key = int(row[column])
            if key in observed and observed[key] != value:
                break
            observed[key] = value
        else:
            consistent.append(name)
    return consistent


def _sample_stage(
    rng: random.Random,
    names: Sequence[str],
    sizes: Sequence[int],
    rows: Sequence[Sequence[int]],
    output: str,
    table_factory: Callable[[int], List[int]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for _ in range(10_000):
        column = rng.randrange(len(names))
        table = table_factory(sizes[column])
        values = [table[int(row[column])] for row in rows]
        if _consistent_inputs(rows, values, names) == [names[column]]:
            stage = {"output": output, "inputs": [{"attr": names[column]}], "table": table}
            return stage, {"controller": names[column], "table": table}
    raise ValueError(f"Could not sample identifiable mapping for {output!r}")


def _binary_table(rng: random.Random, size: int) -> List[int]:
    while True:
        table = [rng.randrange(2) for _ in range(size)]
        if len(set(table)) == 2:
            return table


def _permutation_table(rng: random.Random, size: int) -> List[int]:
    table = list(range(size))
    rng.shuffle(table)
    return table


def _template_has_coverage(templates: Sequence[Tuple[Tuple[int, ...], int]], count: int) -> bool:
    for block in range(count):
        sides = {block in order[:cut] for order, cut in templates}
        if len(sides) != 2:
            return False
    for left in range(count):
        for right in range(left + 1, count):
            orientations = {order.index(left) < order.index(right) for order, _ in templates}
            if len(orientations) != 2:
                return False
    return True


def _sample_templates(rng: random.Random, count: int) -> List[Tuple[Tuple[int, ...], int]]:
    family = [(order, cut) for order in permutations(range(count)) for cut in range(count + 1)]
    for _ in range(10_000):
        selected = rng.sample(family, 4)
        if _template_has_coverage(selected, count):
            return selected
    raise ValueError(f"Could not sample four covering order templates for {count} blocks")


def _extension_spec(block: int, templates: Sequence[Tuple[Tuple[int, ...], int]]) -> Dict[str, Any]:
    positions = []
    sequence = []
    for order, cut in templates:
        positions.append({"before": "go_entity" if block in order[:cut] else "defeat_0"})
        sequence.append(order.index(block))
    return {
        "count": {"from_dim": f"block_{block}_count", "values": [2, 3, 4, 5]},
        "action": {"from_dim": f"block_{block}_action", "values": ["perform", "drink"]},
        "pool": {"by_action": {
            "perform": f"activity.logic_{block}_ritual",
            "drink": f"activity.logic_{block}_potion",
        }},
        "position": {"from_dim": "order_template", "values": positions},
        "sequence": {"from_dim": "order_template", "values": sequence},
        "is_hidden": True,
    }


def instantiate_seeded_process_logic(
    output: Dict[str, Any],
    inputs: Sequence[Dict[str, Any]],
    source_rows: Sequence[Sequence[int]],
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Instantiate one reproducible rule from the ordered_extensions_v1 family."""
    config = output.get("seeded_logic")
    if not config:
        return output, {}
    if config.get("family") != "ordered_extensions_v1":
        raise ValueError(f"Unknown seeded process family: {config.get('family')!r}")

    block_count = int(config["block_count"])
    if block_count not in (2, 3, 4, 5):
        raise ValueError("ordered_extensions_v1 requires 2, 3, 4, or 5 blocks")
    names = [item["name"] for item in inputs]
    sizes = [int(item["n_values"]) for item in inputs]
    if any(set(int(row[i]) for row in source_rows) != set(range(size)) for i, size in enumerate(sizes)):
        raise ValueError("Source demonstrations must cover every value of every input attribute")

    rng = random.Random(seed ^ 0x5EEDC0DE)
    templates = _sample_templates(rng, block_count)
    stages: List[Dict[str, Any]] = []
    mappings: Dict[str, Any] = {}

    order_stage, mappings["order_template"] = _sample_stage(
        rng, names, sizes, source_rows, "order_template",
        lambda size: _permutation_table(rng, size),
    )
    stages.append(order_stage)
    outputs = [{"name": "order_template", "n_values": 4}]

    for block in range(block_count):
        for suffix, factory in (
            ("action", lambda size: _binary_table(rng, size)),
            ("count", lambda size: _permutation_table(rng, size)),
        ):
            name = f"block_{block}_{suffix}"
            stage, mappings[name] = _sample_stage(
                rng, names, sizes, source_rows, name, factory,
            )
            stages.append(stage)
            outputs.append({"name": name, "n_values": 2 if suffix == "action" else 4})

    realized = copy.deepcopy(output)
    realized["outputs"] = outputs
    realized["functions"] = []
    realized["composition"] = {"fn": "pipeline", "stages": stages}
    realized["extensions"] = [_extension_spec(i, templates) for i in range(block_count)]
    realized.pop("extension", None)
    extra_pools = {
        "activity.ritual": {"name_keys": ["rituals"]},
        "activity.potion": {"name_keys": ["potions"]},
    }
    extra_pools.update({
        f"activity.logic_{i}_{kind}": {
            "source": f"activity.{kind}",
            "name_keys": ["rituals" if kind == "ritual" else "potions"],
        }
        for i in range(block_count)
        for kind in ("ritual", "potion")
    })
    realized["extra_pools"] = extra_pools

    template_records = [
        {"order": list(order), "cut": cut, "before": list(order[:cut]), "after": list(order[cut:])}
        for order, cut in templates
    ]
    metadata = {
        "family": config["family"],
        "block_count": block_count,
        "mappings": mappings,
        "templates": template_records,
    }
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata["fingerprint"] = hashlib.sha256(payload).hexdigest()
    return realized, metadata


__all__ = ["instantiate_seeded_process_logic"]
