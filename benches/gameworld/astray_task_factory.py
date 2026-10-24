"""Build seeded Astray task YAMLs using the existing browser maze generator."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from urllib.parse import urlencode

import yaml

MANIFEST = Path(__file__).with_name("astray_difficulty_manifest.json")


def materialize_task(task: dict, output_dir: str | Path) -> Path:
    """Write an episode-local catalog from its saved manifest definition."""
    definition = task["generated_task"]
    expected = generate_task(
        task["difficulty"], seed=task["seed"] % 2**32, task_id=task["game_task_id"],
    )
    expected["max_steps"] = task["max_steps"]
    if task["game_id"] != "03_astray" or definition != expected:
        raise ValueError("Generated Astray task does not match its difficulty/seed")
    root = Path(output_dir)
    directory = root / "astray"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{definition['task_id']}.yaml"
    if path.exists():
        if yaml.safe_load(path.read_text(encoding="utf-8")) != definition:
            raise ValueError("Existing generated Astray catalog differs from manifest")
    else:
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(definition, stream, sort_keys=False)
    return root


def generate_task(difficulty: int, *, seed: int, task_id: str) -> dict:
    """Return a catalog task; the browser generates its map from level and seed."""
    settings = json.loads(MANIFEST.read_text(encoding="utf-8"))["difficulties"]
    if isinstance(difficulty, bool) or not isinstance(difficulty, int) or difficulty not in range(1, 6):
        raise ValueError("difficulty must be an integer within 1..5")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if not task_id or task_id in {".", ".."} or any(c in task_id for c in "/\\"):
        raise ValueError("task_id must be a non-empty file stem")
    spec = settings[str(difficulty)]
    return {
        "task_id": task_id,
        "game_id": "03_astray",
        "game_url_suffix": "?" + urlencode({"level": spec["level"], "train_seed": seed}),
        "evaluator_id": "game_api_metric",
        "max_steps": spec["max_steps"],
        "task_target_score_field": 1.0,
        "task_start_score_field": 0,
        "pause_during_inference": True,
        "continue_on_fail": True,
        "task_prompt": f"Find the exit of this {spec['dimension']}x{spec['dimension']} maze.\n",
        "evaluator_config": {
            "terminal_status": "success",
            "score_field": "game_state.completion_progress",
            "metrics_fields": [
                "game_state.level",
                "game_state.completion_progress",
                "metrics.distance_to_goal",
                "terminal.outcome",
                "terminal.isTerminal",
            ],
        },
    }


def generate_batch(
    *, output_dir: str | Path, min_difficulty: int = 3, max_difficulty: int = 3,
    per_difficulty: int = 20, seed: int = 20260923,
) -> list[Path]:
    """Write a catalog root with an astray/ subdirectory; never replace tasks."""
    if not 1 <= min_difficulty <= max_difficulty <= 5:
        raise ValueError("difficulty range must be within 1..5")
    if per_difficulty < 1:
        raise ValueError("per_difficulty must be positive")
    destination = Path(output_dir) / "astray"
    tasks = []
    for difficulty in range(min_difficulty, max_difficulty + 1):
        for index in range(per_difficulty):
            task_id = f"astray-d{difficulty}-{index + 1:03d}"
            task_seed = seed + (difficulty - min_difficulty) * per_difficulty + index
            task = generate_task(difficulty, seed=task_seed, task_id=task_id)
            path = destination / f"{task_id}.yaml"
            if path.exists():
                raise FileExistsError(path)
            tasks.append((path, task))
    destination.mkdir(parents=True, exist_ok=True)
    for path, task in tasks:
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(task, stream, sort_keys=False)
    return [path for path, _ in tasks]


def add_cases_to_archive(
    archive: str | Path, *, min_difficulty: int = 1, max_difficulty: int = 5,
    per_difficulty: int = 20, seed: int = 20260923,
) -> list[str]:
    """Add JSON cases atomically, preserving every existing archive member."""
    if (type(min_difficulty) is not int or type(max_difficulty) is not int
            or not 1 <= min_difficulty <= max_difficulty <= 5):
        raise ValueError("difficulty range must be integers within 1..5")
    if type(per_difficulty) is not int or per_difficulty < 1:
        raise ValueError("per_difficulty must be a positive integer")
    cases = {}
    for difficulty in range(min_difficulty, max_difficulty + 1):
        for index in range(per_difficulty):
            task_id = f"astray_d{difficulty}_{index + 1:03d}"
            task_seed = seed + (difficulty - min_difficulty) * per_difficulty + index
            definition = generate_task(difficulty, seed=task_seed, task_id=task_id)
            definition.update(difficulty=difficulty, seed=task_seed)
            cases[f"case/astray/{task_id}.json"] = (
                json.dumps(definition, indent=2) + "\n"
            ).encode("utf-8")

    archive = Path(archive)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member names")
        for member in members:
            if member.name not in cases:
                continue
            if not member.isfile():
                raise ValueError(f"Existing case is not a file: {member.name}")
            with source.extractfile(member) as stream:
                existing = json.load(stream)
            if existing != json.loads(cases[member.name]):
                raise ValueError(f"Existing case differs: {member.name}")
        additions = {name: data for name, data in cases.items() if name not in names}
        if not additions:
            return []
        # Write beside the original so replacement is atomic on the same filesystem.
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".tar.gz", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with tarfile.open(temporary, "w:gz") as target:
                for member in members:
                    if member.isfile():
                        with source.extractfile(member) as stream:
                            target.addfile(member, stream)
                    else:
                        target.addfile(member)
                for name, data in additions.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    member.mode = 0o644
                    target.addfile(member, io.BytesIO(data))
            temporary.chmod(archive.stat().st_mode & 0o777)
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    return list(additions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", help="Task catalog root (contains astray/)")
    destination.add_argument("--archive", help="Existing .tar.gz to extend with JSON cases")
    parser.add_argument("--min-difficulty", type=int, default=3)
    parser.add_argument("--max-difficulty", type=int, default=3)
    parser.add_argument("--per-difficulty", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    options = vars(args).copy()
    archive = options.pop("archive")
    output_dir = options.pop("output_dir")
    try:
        if archive:
            paths = add_cases_to_archive(archive, **options)
        else:
            paths = generate_batch(output_dir=output_dir, **options)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(paths)} Astray tasks in {archive or output_dir}")


if __name__ == "__main__":
    main()
