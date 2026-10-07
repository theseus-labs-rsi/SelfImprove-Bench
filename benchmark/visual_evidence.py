"""Transport-only visual evidence helpers; no rewards or learning decisions.

Images are the observations supplied BEFORE the associated action.  A bounded,
deterministic sample is sent to the model; a bundle keeps every referenced frame.
Missing referenced images are errors, never silently converted to text-only.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

# One image budget for every method's refinement input (ACE Reflector, AHE
# Evolver, HyperAgents visual inspection). tools/launch.py pins the value.
IMAGE_LIMIT_ENV = "RSI_MAX_REFINEMENT_IMAGES"
DEFAULT_IMAGE_LIMIT = 8


def refinement_image_limit(value: str | None = None) -> int | None:
    """Parse a positive integer or ``all`` (None = no sampling)."""
    raw = os.environ.get(IMAGE_LIMIT_ENV, str(DEFAULT_IMAGE_LIMIT)) if value is None else value
    text = str(raw).strip().lower()
    if text == "all":
        return None
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"{IMAGE_LIMIT_ENV} must be a positive integer or 'all', got {raw!r}")
    return int(text)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def image_file(path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024 * 1024:
        raise ValueError(f"Missing, empty or oversized visual evidence: {path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        raise ValueError(f"Unsupported visual evidence format: {path}")
    return path


def frames(feedback) -> list[dict]:
    result = []
    for index, step in enumerate(_field(feedback, "trajectory", ()) or ()):
        for image in _field(step, "image_records", ()) or ():
            path = image_file(_field(image, "path"))
            result.append({
                "task_id": str(_field(feedback, "task_id", "")),
                "step_index": _field(step, "step_index", index),
                "timing": "pre_action_observation",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    return result


def sample(items, limit: int | None = 8):
    if limit is None:
        return list(items)
    if limit < 1:
        raise ValueError("Visual sample limit must be positive")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def visual_note(items) -> str:
    if not items:
        return ""
    return (
        "\n\nVisual evidence (ordered image inputs; these may be sampled):\n"
        + "\n".join(
            f"Image {i}: task={item['task_id']}, step={item['step_index']}, "
            f"{item['timing']}, sha256={item['sha256']}"
            for i, item in enumerate(items, 1)
        )
        + "\nThese are observations before the indicated action, NOT its resulting state. "
        "Use the scored textual trajectory for actions/outcomes; do not invent unseen state."
    )


def bundle_feedback(feedback: dict, destination: Path) -> dict:
    """Keep all images, relocate feedback paths and label model-visible samples."""
    limit = refinement_image_limit()
    destination.mkdir(parents=True, exist_ok=True)
    copied = json.loads(json.dumps(feedback))
    entries = frames(feedback)
    images = destination / "images"
    images.mkdir(exist_ok=True)
    cursor = 0
    for step in copied.get("trajectory", ()):
        for image in step.get("image_records") or ():
            entry = dict(entries[cursor])
            source = Path(entry.pop("path"))
            relative = f"images/{cursor:04d}_{entry['sha256'][:12]}{source.suffix.lower()}"
            shutil.copy2(source, destination / relative)
            entry.update(path=relative, original_path=str(source))
            entries[cursor] = entry
            image["path"] = relative
            cursor += 1
    manifest = {
        "schema_version": 1, "path_base": ".", "all_frames": entries,
        "model_sample": sample(entries, limit), "model_sample_limit": limit or "all",
        "sampling": "uniform ordered including first and last",
    }
    (destination / "feedback.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2))
    (destination / "visual_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def chat_message(text: str, image_paths):
    """NexAU chat-completions message with actual image bytes, not path strings."""
    if not image_paths:
        return text
    from nexau.core.messages import Message, TextBlock, ImageBlock
    parts = [TextBlock(text=text)]
    for value in image_paths:
        path = image_file(value)
        parts.append(ImageBlock(
            base64=base64.b64encode(path.read_bytes()).decode("ascii"),
            mime_type=mimetypes.guess_type(path.name)[0] or "image/png",
        ))
    return [Message(role="user", content=parts)]
