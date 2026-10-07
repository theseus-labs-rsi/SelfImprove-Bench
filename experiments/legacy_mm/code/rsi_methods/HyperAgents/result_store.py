import json
import math
import os
from dataclasses import asdict, is_dataclass
from numbers import Real


RESULT_FILENAME = "result.json"
RESULT_SCHEMA_VERSION = 1


def result_to_dict(result):
    """Convert a play result to the canonical serializable mapping."""
    if is_dataclass(result):
        result = asdict(result)
    elif hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, dict):
        raise TypeError(
            "play(agent) must return a dict, dataclass, or object with to_dict()."
        )
    return dict(result)


def normalize_result(result):
    """Validate required fields and add stable schema defaults."""
    result_data = result_to_dict(result)
    schema_version = result_data.get("schema_version", RESULT_SCHEMA_VERSION)
    if schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"result.schema_version must be {RESULT_SCHEMA_VERSION}."
        )
    result_data["schema_version"] = schema_version

    if not isinstance(result_data.get("valid"), bool):
        raise ValueError("result.valid must be a boolean.")

    score = result_data.get("selection_score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError("result.selection_score must be a finite number or null.")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("result.selection_score must be finite.")
        if not 0.0 <= score <= 1.0:
            raise ValueError("result.selection_score must be between 0 and 1.")
        result_data["selection_score"] = score
    if result_data["valid"] and score is None:
        raise ValueError("A valid result must provide selection_score.")

    result_data.setdefault("metrics", {})
    result_data.setdefault("feedback", {})
    result_data.setdefault("artifacts", {})
    result_data.setdefault("cost", {})
    result_data.setdefault("error", None)

    if not isinstance(result_data["metrics"], dict):
        raise ValueError("result.metrics must be a mapping.")
    if not isinstance(result_data["feedback"], (dict, list, str)):
        raise ValueError("result.feedback must be a mapping, list, or string.")
    if not isinstance(result_data["artifacts"], dict):
        raise ValueError("result.artifacts must be a mapping.")
    if not isinstance(result_data["cost"], dict):
        raise ValueError("result.cost must be a mapping.")
    if result_data["error"] is not None and not isinstance(result_data["error"], str):
        raise ValueError("result.error must be a string or null.")
    return result_data


def validate_artifacts(gen_output_dir, artifacts):
    """Require artifact values to reference existing files inside this node."""
    generation_root = os.path.abspath(gen_output_dir)
    for name, relative_path in artifacts.items():
        if not isinstance(name, str) or not isinstance(relative_path, str):
            raise ValueError("result.artifacts must map string names to relative paths.")
        if os.path.isabs(relative_path):
            raise ValueError("Artifact paths must be relative to the generation directory.")
        artifact_path = os.path.abspath(os.path.join(generation_root, relative_path))
        if os.path.commonpath([generation_root, artifact_path]) != generation_root:
            raise ValueError("Artifact paths cannot leave the generation directory.")
        if not os.path.isfile(artifact_path):
            raise ValueError(f"Artifact does not exist: {relative_path}")


def save_result(gen_output_dir, result):
    """Atomically write one generation's canonical result.json."""
    os.makedirs(gen_output_dir, exist_ok=True)
    result_path = os.path.join(gen_output_dir, RESULT_FILENAME)
    temp_path = f"{result_path}.tmp"
    try:
        result_data = normalize_result(result)
        validate_artifacts(gen_output_dir, result_data["artifacts"])
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, result_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return result_path


def write_result(run_dir, generation_id, result):
    """Public adapter interface: persist Result for a generation id."""
    return save_result(os.path.join(run_dir, f"gen_{generation_id}"), result)


def load_result(gen_output_dir):
    result_path = os.path.join(gen_output_dir, RESULT_FILENAME)
    try:
        with open(result_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def result_is_valid(result):
    if result is None:
        return False
    try:
        result_data = normalize_result(result)
    except (TypeError, ValueError):
        return False
    return result_data.get("valid") is True


def get_selection_score(run_dir, generation_id):
    """Read a valid node's scalar score used only for parent selection."""
    result = load_result(os.path.join(run_dir, f"gen_{generation_id}"))
    if not result_is_valid(result):
        return None
    score = result.get("selection_score")
    if isinstance(score, bool) or not isinstance(score, Real):
        return None
    score = float(score)
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None
