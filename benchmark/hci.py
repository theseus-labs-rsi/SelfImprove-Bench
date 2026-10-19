"""Read-only G0--G3 HCI calculation from matched benchmark output directories."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any

from .metrics import execution_score as _score
from .tasks import task_fingerprint


PAIRED_GAMES = {
    "farm": "combat", "combat": "farm", "sweepmine": "hero",
    "hero": "sweepmine", "minecraft": "astray", "astray": "minecraft",
}
GAMEWORLD_IDS = {"18_minecraft-clone-glm": "minecraft", "03_astray": "astray"}
GAMEWORLD_GROUPS = {
    "minecraft": {"medium": {"18_02", "18_03"}, "hard": {"18_04", "18_05"}},
    "astray": {"medium": {"03_03"}, "hard": {"03_05"}},
}


def _number(value: Any) -> float | None:
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and isfinite(value) else None)


def headroom_closed_index(initial: float, final: float, reference: float = 1.0) -> float | None:
    """Return a ratio, not a percent; zero headroom is undefined."""
    if any(_number(value) is None for value in (initial, final, reference)):
        raise ValueError("HCI requires finite numeric scores")
    return (final - initial) / (reference - initial) if reference != initial else None


def _game(task: dict[str, Any]) -> str:
    game = task["game"]
    if game == "gameworld":
        game = GAMEWORLD_IDS.get(task["task"].get("game_id"))
    if game == "heros_journey":
        game = "hero"
    if game not in PAIRED_GAMES:
        raise ValueError(f"Unsupported game: {game}")
    return game


def _index(items: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for item in items:
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in result:
            raise ValueError(f"{label}: missing or duplicate run_id: {run_id}")
        result[run_id] = item
    return result


def _validate_run(result: dict[str, Any], manifest: dict[str, Any], label: str) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{label}: unsupported task manifest schema")
    for phase, field in (("training", "training_runs"), ("holdout", "holdout_runs")):
        planned = _index(manifest[phase], f"{label} {phase} manifest")
        records = _index(result[field], f"{label} {phase} results")
        if planned.keys() != records.keys():
            raise ValueError(
                f"{label} {phase}: incomplete/mismatched run IDs; "
                f"missing={sorted(planned.keys() - records.keys())}, "
                f"unexpected={sorted(records.keys() - planned.keys())}"
            )
        for run_id, task in planned.items():
            if task.get("task_fingerprint") != task_fingerprint(task["task"]):
                raise ValueError(f"{label}: task fingerprint changed: {run_id}")
            for key in ("game", "difficulty", "task_seed", "replay", "replay_of", "replay_gap"):
                if records[run_id].get(key) != task.get(key):
                    raise ValueError(f"{label}: record/manifest {key} mismatch: {run_id}")


def _summary(pairs: list[dict[str, Any]], target: str, reason: str | None = None) -> dict[str, Any]:
    missing = [pair for pair in pairs if pair["initial_score"] is None or pair["final_score"] is None]
    unavailable = reason or ("missing_pair_scores" if missing else None)
    if not pairs and not unavailable:
        unavailable = "no_tasks"
    initial = fmean(pair["initial_score"] for pair in pairs) if not unavailable else None
    final = fmean(pair["final_score"] for pair in pairs) if not unavailable else None
    value = headroom_closed_index(initial, final) if not unavailable else None
    return {
        "target_game": target,
        "status": "unavailable" if unavailable else "undefined" if value is None else "ok",
        "reason": unavailable or ("zero_headroom" if value is None else None),
        "task_count": len(pairs), "valid_pair_count": len(pairs) - len(missing),
        "initial_score": initial, "final_score": final, "reference_score": 1.0,
        "score_change": final - initial if initial is not None else None,
        "hci": value,
        "warning": "initial_above_reference" if initial is not None and initial > 1 else None,
        "pairs": pairs,
    }


def _group(tasks: list[dict[str, Any]], game: str, level: str) -> tuple[list[dict[str, Any]], str | None]:
    own = [task for task in tasks if _game(task) == game]
    if game in GAMEWORLD_GROUPS:
        expected = GAMEWORLD_GROUPS[game][level]
        selected = [task for task in own if task["task"].get("game_task_id") in expected]
        ids = [task["task"]["game_task_id"] for task in selected]
        if set(ids) != expected or len(ids) != len(expected):
            return selected, "missing_or_duplicate_group_tasks"
    else:
        difficulty = 3 if level == "medium" else 5 if game in {"farm", "combat"} else 4
        selected = [task for task in own if task["difficulty"] == difficulty]
    return selected, None


def calculate_hci(
    trained: dict[str, Any], initial: dict[str, Any],
    trained_manifest: dict[str, Any], initial_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate identities and aggregate scores before applying the HCI formula."""
    _validate_run(trained, trained_manifest, "trained")
    _validate_run(initial, initial_manifest, "initial")
    if initial_manifest["training"] or initial["config"].get("schedule"):
        raise ValueError("Initial baseline must be an untrained --initial run")
    agent = trained["config"]["agent"]
    if agent != initial["config"]["agent"]:
        raise ValueError("Initial and trained agent configurations differ")
    if agent.get("provider") not in {"ace", "ahe", "hyperagents"}:
        raise ValueError("Unsupported framework")
    normal = [task for task in trained_manifest["training"] if not task.get("replay")]
    games = {_game(task) for task in normal}
    if len(games) != 1:
        raise ValueError("HCI requires one training game per experiment")
    source = games.pop()
    holdout = _index(trained_manifest["holdout"], "trained holdout")
    before = _index(initial_manifest["holdout"], "initial holdout")
    if holdout.keys() != before.keys():
        raise ValueError("Initial and trained holdout task IDs differ")
    for run_id, task in holdout.items():
        other = before[run_id]
        if any(task.get(key) != other.get(key)
               for key in ("task_fingerprint", "game", "difficulty", "task_seed")):
            raise ValueError(f"Initial and trained holdout task identities differ: {run_id}")
    training_records = _index(trained["training_runs"], "training")
    originals = _index(normal, "original training tasks")
    replay_pairs = []
    original_rounds = {}
    for task in trained_manifest["training"]:
        if not task.get("replay"):
            original_rounds[task["run_id"]] = len(original_rounds) + 1
            continue
        original_id = task.get("replay_of")
        original = originals.get(original_id)
        gap = task.get("replay_gap")
        if original is None or task["task_fingerprint"] != original["task_fingerprint"]:
            raise ValueError(f"Replay does not match an original task: {task['run_id']}")
        if not isinstance(gap, int) or isinstance(gap, bool) or gap <= 0:
            raise ValueError(f"Replay has an invalid round gap: {task['run_id']}")
        if original_id not in original_rounds or len(original_rounds) - original_rounds[original_id] != gap:
            raise ValueError(f"Replay timing does not match its round gap: {task['run_id']}")
        record = training_records[original_id]
        baseline = _score(record)
        replay = training_records[task["run_id"]]
        replay_pairs.append({
            "original_run_id": original_id, "replay_run_id": task["run_id"],
            "original_round": original_rounds[original_id], "replay_after_round": len(original_rounds),
            "replay_gap": gap, "initial_score": baseline, "final_score": _score(replay),
            "initial_status": record.get("status"), "final_status": replay.get("status"),
        })
    groups = {"G0": _summary(replay_pairs, source)}
    final_records = _index(trained["holdout_runs"], "trained holdout results")
    initial_records = _index(initial["holdout_runs"], "initial holdout results")
    for name, target, level in (("G1", source, "medium"), ("G2", source, "hard"),
                                ("G3", PAIRED_GAMES[source], "medium")):
        selected, reason = _group(list(holdout.values()), target, level)
        pairs = []
        for task in selected:
            run_id = task["run_id"]
            start, end = initial_records[run_id], final_records[run_id]
            pairs.append({
                "run_id": run_id, "task_fingerprint": task["task_fingerprint"],
                "game_task_id": task["task"].get("game_task_id"),
                "initial_score": _score(start), "final_score": _score(end),
                "initial_status": start.get("status"), "final_status": end.get("status"),
            })
        groups[name] = {**_summary(pairs, target, reason), "difficulty_group": level}
    return {
        "schema_version": 1,
        "status": "incomplete" if any(g["status"] == "unavailable" for g in groups.values()) else "completed",
        "source_game": source, "method": agent["provider"],
        "formula": "(mean(final) - mean(initial)) / (1 - mean(initial))",
        "score_field": "relative_score", "hci_unit": "ratio", "failed_execution_score": 0,
        "g0_baseline": "pre_refinement_execution_score",
        "groups": groups,
    }


def _read_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = path.parent if path.is_file() else path
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "task_manifest.json").read_text(encoding="utf-8"))
    return result, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path, required=True, help="Trained output directory or result.json")
    parser.add_argument("--initial", type=Path, required=True, help="Matching --initial output directory or result.json")
    parser.add_argument("--output", type=Path, help="JSON output; defaults to stdout")
    args = parser.parse_args()
    try:
        trained, manifest = _read_run(args.trained)
        initial, initial_manifest = _read_run(args.initial)
        report = calculate_hci(trained, initial, manifest, initial_manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        inputs = [p.parent if p.is_file() else p for p in (args.trained, args.initial)]
        protected = {p.resolve() for root in inputs for p in (root / "result.json", root / "task_manifest.json")}
        if args.output.resolve() in protected:
            parser.error("Output must not overwrite source results or task manifests")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Saved G0--G3 HCI ({report['status']}) to {args.output}")
    else:
        print(content, end="")
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
