import json
import os
import random
from pathlib import Path

WORLD_INFO_PATH = Path(__file__).resolve().parents[2] / "world_info" / "lexicons"

FUNCTION_NAME_CANDIDATES = [
    "f_{input}_{output}",
    "f_base_{output}",
]

OUTPUT_SPECS = [
    {"name": "action_type", "n_values": 2},
    {"name": "ordering", "n_values": 2},
]


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def list_json_files(directory):
    return [path for path in Path(directory).glob("*.json")]


def collect_meta_paths(input_meta: dict) -> set:
    """Extract lexicon paths already used in input_meta for later filtering."""
    used_paths = set()
    for key in ("name", "repeat_name"):
        used_paths.update(input_meta.get(key, []))
    used_paths.update(input_meta.get("properties", {}).values())
    return used_paths


def get_valid_pools(data, meta_paths: set, threshold: int, current_path="") -> list:
    """Recursively find a lexicon pool that is long enough and is not metadata."""
    valid_pools = []
    if not isinstance(data, dict):
        return valid_pools
    for key, value in data.items():
        path = f"{current_path}.{key}" if current_path else key
        if isinstance(value, list):
            if path not in meta_paths and len(value) >= threshold:
                valid_pools.append({"name": key, "path": path, "size": len(value)})
        elif isinstance(value, dict):
            valid_pools.extend(get_valid_pools(value, meta_paths, threshold, path))
    return valid_pools


def get_nested_node(data: dict, category_path: str):
    """Get a lexicon node by dotted path."""
    current = data
    for key in category_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return {}
        current = current[key]
    return current


def _find_input_pools(input_dimensions: int, input_n: int, seed: int) -> list:
    meta = {
        "name": ["name_pools.first", "name_pools.last"],
        "repeat_name": ["name_pools.repeat_first", "name_pools.repeat_last"],
        "properties": {"location": "locations"},
    }
    candidates = []
    for path in list_json_files(WORLD_INFO_PATH):
        if "semantic" not in path.name or "distractor" in path.name:
            continue
        lexicon = load_json(path)
        node = get_nested_node(lexicon, "entity.npc") or lexicon
        candidates.extend(get_valid_pools(node, collect_meta_paths(meta), input_n))
    unique = {(item["path"], item["name"]): item for item in candidates}
    candidates = list(unique.values())
    if len(candidates) < input_dimensions:
        raise ValueError(
            f"Not enough input pools: need {input_dimensions}, only {len(candidates)} available."
        )
    return random.Random(seed).sample(candidates, input_dimensions)


def construct(
    input_dimensions: int = 2,
    input_n: int = 3,
    output_dimensions: int = 2,
    output_n: int = 5,
    seed: int = 1,
    order_enabled: bool = False,
):

    if input_dimensions != 2:
        raise ValueError("The first version of single-level override requires input_dimensions=2.")
    if output_dimensions < 2 or output_dimensions > len(OUTPUT_SPECS):
        raise ValueError(
            f"output_dimensions must be between 2 and {len(OUTPUT_SPECS)}."
        )
    if input_n < 2 or output_n < 2:
        raise ValueError("input_n and output_n must be at least 2.")

    rng = random.Random(seed)
    input_pools = _find_input_pools(input_dimensions, input_n, seed)
    inputs = [
        {
            "name": f"attribute_{index}",
            "n_values": input_n,
            "pool": pool["path"],
            "nonce_pool": f"attribute_slots.{index}",
            "nonce_label": f"attribute_labels.{index}",
        }
        for index, pool in enumerate(input_pools)
    ]
    if output_dimensions != len(OUTPUT_SPECS):
        raise ValueError("process override requires both action_type and ordering outputs.")
    output_specs = [dict(spec) for spec in rng.sample(OUTPUT_SPECS, output_dimensions)]
    functions = []
    for output_spec in output_specs:
        input_idx = rng.randint(0, input_dimensions - 1)
        input_name = f"attribute_{input_idx}"
        output_name = output_spec["name"]
        function_name = next(
            candidate.format(input=input_name, output=output_name)
            for candidate in FUNCTION_NAME_CANDIDATES
            if candidate.format(input=input_name, output=output_name)
            not in {function["name"] for function in functions}
        )
        output_size = output_spec["n_values"]
        output_map = list(range(output_size))
        rng.shuffle(output_map)
        functions.append(
            {
                "name": function_name,
                "input": input_name,
                "output": output_name,
                "map": [value % output_size for value in output_map[:input_n]],
            }
        )

    output = {
        "type": "process",
        "outputs": output_specs,
        "functions": functions,
        "composition": {
            "fn": "override",
            "base_fns": [function["name"] for function in functions],
            "override_attr": f"attribute_{rng.randint(0, input_dimensions - 1)}",
            "override_value": rng.randint(0, input_n - 1),
            "override_dims": {
                output_spec["name"]: rng.randint(0, output_spec["n_values"] - 1)
                for output_spec in output_specs
            },
        },
        "extra_pools": {
            "activity.ritual": {"name_keys": ["rituals"]},
            "activity.potion": {"name_keys": ["potions"]},
        },
        "extension": {
            "count": 1,
            "action": {
                "from_dim": "action_type",
                "values": ["perform", "drink"],
            },
            "pool": {
                "by_action": {
                    "perform": "activity.ritual",
                    "drink": "activity.potion",
                }
            },
            "position": (
                {
                    "from_dim": "ordering",
                    "values": [{"before": "buy_0"}, {"after": "buy_0"}],
                }
                if order_enabled
                else {"before": "buy_0"}
            ),
            "is_hidden": True,
        },
    }
    return {
        "input_category": "entity.npc",
        "input_meta": {
            "name": ["name_pools.first", "name_pools.last"],
            "repeat_name": ["name_pools.repeat_first", "name_pools.repeat_last"],
            "properties": {"location": "locations"},
        },
        "inputs": inputs,
        "output": output,
        "_generator": {
            "family": "single_layer_override_process",
            "seed": seed,
            "order_enabled": order_enabled,
            "order_mode": "ordered_v1" if order_enabled else "flat",
            "input_dimensions": input_dimensions,
            "input_n": input_n,
            "output_dimensions": output_dimensions,
            "output_n": output_n,
        },
    }


if __name__ == "__main__":
    print(json.dumps(construct(), indent=2, ensure_ascii=False))
