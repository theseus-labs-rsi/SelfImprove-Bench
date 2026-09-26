"""Read-only aggregation for unified results and legacy overall-control-loop runs.

Minimal time override schema::
    {"run_20240101_000000_000000/repeat_001": {
        "status": "external_interruption", "impute_from_peers": true
    }}

Keys may be repeat directory names or ``run_id/repeat_id`` paths.  The latter
avoids collisions when several legacy runs are aggregated together.  The
script reads result/summary files only and writes per-repeat and grouped rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "first_absolute_score", "last_absolute_score", "absolute_gain", "relative_gain",
    "replay_score_difference", "replay_difference_per_gap",
    "holdout_same_domain_same_difficulty", "holdout_same_domain_higher_difficulty",
    "holdout_cross_domain", "tokens_per_absolute_gain", "time_per_absolute_gain",
    "score_variance", "declining_round_fraction",
    "drop_20pct_round_count",
)
CHINESE_COLUMNS = {
    "first_absolute_score": "首轮绝对分数", "last_absolute_score": "末轮绝对分数",
    "absolute_gain": "（末轮-首轮）", "relative_gain": "（末轮-首轮）/首轮",
    "replay_score_difference": "重放得分差异", "replay_difference_per_gap": "重放分差/间隔",
    "holdout_same_domain_same_difficulty": "域内同难度上的留出评测",
    "holdout_same_domain_higher_difficulty": "域内难度提高的留出评测",
    "holdout_cross_domain": "跨域任务评测", "tokens_per_absolute_gain": "Token/绝对增益",
    "time_per_absolute_gain": "Time/绝对增益", "score_variance": "得分方差",
    "declining_round_fraction": "得分下降轮次占比",
    "drop_20pct_round_count": "回退至少20%的轮次数",
}
TEXT_FIELDS = (
    "source_path", "schema_kind", "run_id", "repeat_id", "method", "provider",
    "benchmark", "difficulty", "token_scope", "time_source", "time_confidence",
)
COUNT_FIELDS = (
    "observed_training_count", "observed_holdout_count", "observed_replay_pair_count",
    "time_observed_valid_n",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _reason(row: dict[str, Any], text: str) -> None:
    row.setdefault("missing_fields", []).append(text)


def _base(source: Path, kind: str, run_id: str, repeat_id: str) -> dict[str, Any]:
    return {
        "source_path": str(source.resolve()), "schema_kind": kind, "run_id": run_id,
        "repeat_id": repeat_id, "method": None, "provider": None, "benchmark": None,
        "difficulty": None, **{name: None for name in METRICS},
        "observed_training_count": 0, "observed_holdout_count": 0,
        "observed_replay_pair_count": 0, "time_observed_valid_n": 0,
        "time_observed_range_seconds": None, "missing_fields": [], "token_scope": "missing",
        "time_source": None, "time_confidence": None, "time_imputed": False,
        "token_total": None, "time_total_seconds": None,
    }


def _scores(records: list[dict[str, Any]]) -> list[float]:
    return [
        score for record in records
        if record.get("status") in {"ok", "skipped"}
        and (score := _num(record.get("relative_score"))) is not None
    ]


def _costs(records: list[dict[str, Any]], field: str) -> float | None:
    values = [_num(record.get(field)) for record in records]
    return sum(value for value in values if value is not None) if values and all(value is not None for value in values) else None


def _replay(records: list[dict[str, Any]], row: dict[str, Any]) -> None:
    by_id = {record.get("run_id"): _num(record.get("relative_score")) for record in records}
    deltas: list[float] = []
    normalized: list[float] = []
    for record in records:
        if not record.get("replay") or record.get("replay_of") not in by_id:
            continue
        original = by_id.get(record.get("replay_of"))
        gap = record.get("replay_gap")
        score = _num(record.get("relative_score"))
        if original is None or score is None or not isinstance(gap, int) or gap <= 0:
            continue
        deltas.append(score - original)
        normalized.append((score - original) / gap)
    row["observed_replay_pair_count"] = len(deltas)
    row["replay_score_difference"] = _mean(deltas)
    row["replay_difference_per_gap"] = _mean(normalized)
    if not deltas:
        _reason(row, "no valid replay pair")


def _evolution(row: dict[str, Any], scores: list[float]) -> None:
    row["observed_training_count"] = len(scores)
    if not scores:
        _reason(row, "no non-replay training scores")
        return
    first, last = scores[0], scores[-1]
    gain = last - first
    row.update(first_absolute_score=first, last_absolute_score=last, absolute_gain=gain)
    row["relative_gain"] = gain / first if first else None
    if not first:
        _reason(row, "first score is zero")
    row["score_variance"] = statistics.pvariance(scores) if scores else None
    declines = sum(current < previous for previous, current in zip(scores, scores[1:]))
    row["declining_round_fraction"] = declines / (len(scores) - 1) if len(scores) > 1 else 0.0
    row["drop_20pct_round_count"] = sum(
        previous > 0 and current <= 0.8 * previous
        for previous, current in zip(scores, scores[1:])
    )


def _holdout_new(row: dict[str, Any], config: dict[str, Any], records: list[dict[str, Any]]) -> None:
    schedule = config.get("schedule", [])
    train_pairs = {(item.get("game"), item.get("difficulty")) for item in schedule if isinstance(item, dict)}
    train_games = {game for game, _ in train_pairs}
    max_difficulty: dict[str, int] = {}
    for game, difficulty in train_pairs:
        if isinstance(game, str) and isinstance(difficulty, int):
            max_difficulty[game] = max(max_difficulty.get(game, 0), difficulty)
    buckets: dict[str, list[float]] = {"same": [], "higher": [], "cross": []}
    for record in records:
        if record.get("status") != "ok":
            continue
        game, difficulty = record.get("game"), record.get("difficulty")
        score = _num(record.get("relative_score"))
        if score is None:
            continue
        if (game, difficulty) in train_pairs:
            category = "same"
        elif game not in train_games:
            category = "cross"
        elif isinstance(difficulty, int) and difficulty > max_difficulty.get(game, 0):
            category = "higher"
        else:
            continue
        buckets[category].append(score)
    row["holdout_same_domain_same_difficulty"] = _mean(buckets["same"])
    row["holdout_same_domain_higher_difficulty"] = _mean(buckets["higher"])
    row["holdout_cross_domain"] = _mean(buckets["cross"])
    row["observed_holdout_count"] = sum(len(values) for values in buckets.values())
    for name, values in buckets.items():
        if not values:
            _reason(row, f"no classified holdout: {name}")


def _cost(row: dict[str, Any], training: list[dict[str, Any]]) -> None:
    tokens = _costs(training, "total_tokens")
    seconds = _costs(training, "total_seconds")
    gain = row.get("absolute_gain")
    if tokens is not None:
        row["token_scope"] = "measured"
        row["token_total"] = tokens
    else:
        _reason(row, "training total_tokens missing")
    if seconds is not None:
        row["time_source"] = "recorded"
        row["time_confidence"] = "high"
        row["time_total_seconds"] = seconds
    if _num(gain) is not None and gain > 0:
        row["tokens_per_absolute_gain"] = tokens / gain if tokens is not None else None
        row["time_per_absolute_gain"] = seconds / gain if seconds is not None else None
    else:
        _reason(row, "non-positive absolute gain; cost per gain is null")


def parse_unified(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    row = _base(path, "unified_runner", str(data.get("config_hash", path.stem)), "single")
    config = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
    agent = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    row.update(provider=agent.get("provider"), method=agent.get("provider"))
    training = [item for item in data.get("training_runs", []) if isinstance(item, dict)]
    holdout = [item for item in data.get("holdout_runs", []) if isinstance(item, dict)]
    non_replay = [item for item in training if not item.get("replay")]
    _evolution(row, _scores(non_replay))
    _replay(training, row)
    _holdout_new(row, config, holdout)
    _cost(row, non_replay)
    games = sorted({item.get("game") for item in training if item.get("game")})
    row["benchmark"] = "+".join(games) or None
    difficulties = sorted({item.get("difficulty") for item in training if isinstance(item.get("difficulty"), int)})
    row["difficulty"] = difficulties[0] if len(difficulties) == 1 else ("mixed" if difficulties else None)
    return [row]


def _legacy_metadata(run_root: Path) -> dict[str, Any]:
    path = run_root / "experiment.toml"
    if not path.is_file():
        return {}
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    run = value.get("run", {}) if isinstance(value.get("run"), dict) else {}
    benchmarks = value.get("benchmark", {}) if isinstance(value.get("benchmark"), dict) else {}
    names = list(benchmarks)
    label = run_root.name.lower()
    matches = [name for name in names if str(name).lower() in label]
    benchmark = names[0] if len(names) == 1 else (matches[0] if len(matches) == 1 else None)
    return {
        "method": run.get("rsi_method"), "provider": run.get("rsi_method"),
        "benchmark": benchmark, "difficulty": run.get("difficulty"),
        "metadata_complete": benchmark is not None and isinstance(run.get("difficulty"), int),
    }


def _run_start(root: Path) -> float | None:
    match = re.match(r"run_(\d{8})_(\d{6})_(\d{6})(?:_|$)", root.name)
    if not match:
        return None
    return datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S_%f").timestamp()


def _legacy_time(repeat_dir: Path, run_root: Path, index: int, previous_end: float | None) -> float | None:
    summary = repeat_dir / "summary.json"
    if not summary.is_file():
        return None
    end = os.path.getmtime(summary)
    start = _run_start(run_root) if index == 0 else previous_end
    if start is None:
        return None
    return end - start


def _legacy_repeat(repeat_dir: Path, run_root: Path) -> dict[str, Any]:
    summary_path = repeat_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    row = _base(summary_path if summary_path.exists() else repeat_dir, "legacy_overall_control_loop", run_root.name, repeat_dir.name)
    metadata = _legacy_metadata(run_root)
    label = run_root.name.lower()
    row["benchmark"] = metadata.get("benchmark") or next((name for name in ("sweepmine", "combat", "farm", "hero") if name in label), None)
    row["method"] = metadata.get("method") or next((name for name in ("hyperagents", "ace", "self_harness", "ahe") if name in label), None)
    row["provider"] = metadata.get("provider") or row["method"]
    row["difficulty"] = metadata.get("difficulty")
    metadata_complete = metadata.get("metadata_complete", False)
    held_in = summary.get("held_in", {}) if isinstance(summary.get("held_in"), dict) else {}
    initial = _num(held_in.get("initial", {}).get("score_mean"))
    final = _num(held_in.get("final", {}).get("score_mean"))
    scores = [value for value in (initial, final) if value is not None]
    _evolution(row, scores)
    row["drop_20pct_round_count"] = None  # Legacy summaries do not contain consecutive round scores.
    round_scores: list[float] = []
    holdout_scores: list[float] = []
    round_dirs = sorted(path for path in repeat_dir.glob("round_*") if path.is_dir())
    for round_dir in round_dirs:
        held_in_path = round_dir / "bench" / "held_in" / "summary.json"
        held_out_path = round_dir / "bench" / "held_out" / "summary.json"
        if held_in_path.is_file():
            value = _num(_read_json(held_in_path).get("score_mean"))
            if value is not None:
                round_scores.append(value)
        if held_out_path.is_file():
            value = _num(_read_json(held_out_path).get("score_mean"))
            if value is not None:
                holdout_scores.append(value)
    if len(round_scores) > 1:
        row["score_variance"] = statistics.pvariance(round_scores)
        row["declining_round_fraction"] = sum(b < a for a, b in zip(round_scores, round_scores[1:])) / (len(round_scores) - 1)
        row["observed_training_count"] = len(round_scores)
    if round_scores:
        row["drop_20pct_round_count"] = sum(
            previous > 0 and current <= 0.8 * previous
            for previous, current in zip(round_scores, round_scores[1:])
        )
    held_out_final = _num(summary.get("held_out", {}).get("final", {}).get("score_mean"))
    if metadata_complete and row["benchmark"] and row["difficulty"] is not None:
        row["holdout_same_domain_same_difficulty"] = held_out_final if held_out_final is not None else _mean(holdout_scores)
    else:
        _reason(row, "legacy holdout metadata does not prove active domain and difficulty")
    row["observed_holdout_count"] = 1 if row["holdout_same_domain_same_difficulty"] is not None else 0
    if row["holdout_same_domain_same_difficulty"] is None:
        _reason(row, "legacy holdout score missing")
    usage = summary.get("benchmark_usage", {}).get("held_in", {})
    tokens = _num(usage.get("total_tokens")) if isinstance(usage, dict) else None
    if tokens is not None:
        row["token_scope"] = "partial"
        row["token_total"] = tokens
        row["_legacy_tokens"] = tokens
    else:
        _reason(row, "legacy held_in total_tokens missing")
    row["_legacy_observed_time"] = None
    return row


def parse_legacy(path: Path, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    root = path if path.is_dir() else path.parent
    repeats = sorted(item for item in root.glob("repeat_*") if item.is_dir())
    rows = [_legacy_repeat(item, root) for item in repeats]
    previous_end = None
    for index, (row, repeat_dir) in enumerate(zip(rows, repeats)):
        row["_legacy_observed_time"] = _legacy_time(repeat_dir, root, index, previous_end)
        summary = repeat_dir / "summary.json"
        previous_end = os.path.getmtime(summary) if summary.is_file() else previous_end
        duration = row["_legacy_observed_time"]
        if duration is not None:
            row["time_source"] = "filesystem_estimate"
            row["time_confidence"] = "low"
            row["time_observed_valid_n"] = 1
            row["time_observed_range_seconds"] = [duration, duration]
        else:
            _reason(row, "legacy time boundary unavailable")
    overrides = overrides or {}
    for row in rows:
        relative_repeat = str(Path(row["source_path"]).parent.relative_to(root))
        key_options = (
            f"{root.name}/{row['repeat_id']}",
            f"{root.name}\\{row['repeat_id']}",
            row["repeat_id"],
            relative_repeat,
        )
        override = next((overrides.get(key) for key in key_options if key in overrides), None)
        if isinstance(override, dict) and override.get("status") == "external_interruption":
            row["_legacy_observed_time"] = None
            row["time_source"] = None
            row["time_confidence"] = None
            row["time_observed_valid_n"] = 0
            row["time_observed_range_seconds"] = None
            if override.get("impute_from_peers"):
                row["_needs_peer_time"] = True
            else:
                _reason(row, "explicit external interruption")
    valid = [row["_legacy_observed_time"] for row in rows if row.get("_legacy_observed_time") is not None]
    for row in rows:
        if row.get("_needs_peer_time"):
            if valid:
                row["_legacy_observed_time"] = statistics.fmean(valid)
                row["time_source"] = "peer_repeat_mean"
                row["time_confidence"] = "low"
                row["time_imputed"] = True
                row["time_observed_valid_n"] = len(valid)
                row["time_observed_range_seconds"] = [min(valid), max(valid)]
            else:
                _reason(row, "no valid peer repeat for time imputation")
        seconds = row.get("_legacy_observed_time")
        row["time_total_seconds"] = seconds
        gain = _num(row.get("absolute_gain"))
        tokens = row.pop("_legacy_tokens", None)
        row.pop("_needs_peer_time", None)
        if gain is not None and gain > 0:
            row["tokens_per_absolute_gain"] = tokens / gain if tokens is not None else None
            row["time_per_absolute_gain"] = seconds / gain if seconds is not None else None
        else:
            _reason(row, "non-positive absolute gain; cost per gain is null")
        row.pop("_legacy_observed_time", None)
    return rows


def _input_files(inputs: list[str]) -> list[Path]:
    result: list[Path] = []
    for value in inputs:
        path = Path(value).resolve()
        if path.is_file() and path.name == "result.json":
            result.append(path)
        elif path.is_dir() and (path / "result.json").is_file():
            result.append(path / "result.json")
        elif path.is_dir() and (path / "summary.json").is_file():
            result.append(path)
        else:
            raise ValueError(f"input is not a supported result or legacy run: {value}")
    return result


def aggregate(inputs: list[str], overrides: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in _input_files(inputs):
        if path.name == "result.json" and "training_runs" in _read_json(path):
            rows.extend(parse_unified(path))
        else:
            rows.extend(parse_legacy(path, overrides))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in ("schema_kind", "method", "provider", "benchmark", "difficulty"))
        groups.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for key, members in groups.items():
        item = {"schema_kind": key[0], "method": key[1], "provider": key[2], "benchmark": key[3], "difficulty": key[4], "repeat_count": len(members)}
        for metric in METRICS:
            values = [_num(row.get(metric)) for row in members]
            values = [value for value in values if value is not None]
            item[metric] = {"mean": _mean(values), "non_null_n": len(values)}
        for name in COUNT_FIELDS:
            values = [_num(row.get(name)) for row in members]
            values = [value for value in values if value is not None]
            item[name] = {"mean": _mean(values), "non_null_n": len(values)}
        time_values = [_num(row.get("time_per_absolute_gain")) for row in members]
        time_values = [value for value in time_values if value is not None]
        time_sources = sorted({row["time_source"] for row in members if row.get("time_source")})
        imputed_count = sum(bool(row.get("time_imputed")) for row in members)
        time_metric: dict[str, Any] = {"mean": _mean(time_values), "non_null_n": len(time_values)}
        if not time_values:
            time_metric["reason"] = "no valid positive-gain time observation"
        item["time_per_absolute_gain"] = time_metric
        item["time_sources"] = time_sources
        item["time_imputed_repeat_count"] = imputed_count
        confidences = {row.get("time_confidence") for row in members if row.get("time_confidence")}
        item["time_confidence"] = "low" if "low" in confidences else ("high" if confidences == {"high"} else None)
        item["missing_fields"] = sorted({
            reason
            for row in members
            for reason in row.get("missing_fields", [])
            if isinstance(reason, str)
        })
        aggregate_rows.append(item)
    return rows, aggregate_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items() if key in fields
            }
            for row in rows
        )


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, aggregate: bool) -> None:
    fields = ["schema_kind", "method", "benchmark", "difficulty"]
    labels = ["schema", "method", "benchmark", "difficulty"]
    if aggregate:
        fields.append("repeat_count")
        labels.append("重复数")
    fields.extend(METRICS)
    labels.extend(CHINESE_COLUMNS[field] for field in METRICS)
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if aggregate and isinstance(value, dict):
                mean = value.get("mean")
                value = f"{'N/A' if mean is None else mean} (n={value.get('non_null_n', 0)})"
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "per_repeat.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "aggregate.json").write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(directory / "per_repeat.csv", rows)
    _write_csv(directory / "aggregate.csv", aggregate_rows)
    _write_markdown(directory / "table.md", aggregate_rows, aggregate=True)
    _write_markdown(directory / "per_repeat_table.md", rows, aggregate=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RSI result summaries without changing inputs")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-overrides")
    args = parser.parse_args()
    overrides: dict[str, Any] = {}
    if args.time_overrides:
        source = args.time_overrides
        raw = source if source.lstrip().startswith("{") else Path(source).read_text(encoding="utf-8")
        overrides = json.loads(raw)
    rows, aggregate_rows = aggregate(args.input, overrides)
    write_outputs(rows, aggregate_rows, args.output_dir)
    print(f"Aggregated {len(rows)} repeat rows into {len(aggregate_rows)} groups")


if __name__ == "__main__":
    main()
