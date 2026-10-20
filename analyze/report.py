"""Evidence-linked, per-game reports from existing trajectory analysis JSON files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .llm import AnalysisLLMError, AnalysisOutput, _create_openai_client, _record_telemetry


GAMES = ("farm", "combat", "sweepmine", "hero", "minecraft", "astray")
METHODS = {"ACE": "ACE", "AHE": "AHE", "HyperAgent": "HyperAgents", "HyperAgents": "HyperAgents"}
DIMENSIONS = ("relations", "scope", "function", "surface")
LIMITATIONS = [
    "Game identity is checked when embedded in the input; otherwise --game is a user declaration, not an independent check.",
    "Node scores are the analyzer's native parent/current scores, not adjacent training-round differences or HCI.",
    "Score availability and comparability depend on the underlying artifacts; missing scores stay missing.",
    "Proposal labels are model annotations, not independent ground truth. Unknown labels remain unknown.",
    "Category/score associations count each node once per label; nodes can contain multiple labels and proposals.",
    "These are descriptive associations, not causal effects, independent replicates, or significance tests.",
    "Incomplete and failed analyses are included and flagged; unclassified nodes are not treated as having no proposals.",
]


class EvidenceStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    observation: str
    interpretation: str
    implication: str
    evidence_ids: list[str]
    next_experiment: str


class ReportInsights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[EvidenceStatement] = Field(max_length=3)
    hypotheses: list[EvidenceStatement] = Field(max_length=2)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


def _score_stats(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [node["score_delta"] for node in nodes if node["score_delta"] is not None]
    return {
        "node_count": len(nodes), "scored_node_count": len(deltas),
        "missing_score_count": len(nodes) - len(deltas),
        "positive": sum(value > 0 for value in deltas),
        "zero": sum(value == 0 for value in deltas),
        "negative": sum(value < 0 for value in deltas),
        "mean_score_delta": fmean(deltas) if deltas else None,
    }


def _labels(opinion: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "relations": [name for name in ("semantically_new", "contradiction_or_revision", "duplicate")
                      if opinion[name] is True],
        "scope": [opinion["scope"] or "undetermined"],
        "function": opinion["function"] or ["undetermined"],
        "surface": [opinion["surface"] or "undetermined"],
    }


def build_report(inputs: list[str | Path], game: str) -> dict[str, Any]:
    """Recompute descriptive statistics without trusting cached summary counters."""
    if game not in GAMES:
        raise ValueError(f"Unsupported game: {game}")
    if not inputs:
        raise ValueError("At least one trajectory analysis file is required")
    report: dict[str, Any] = {
        "schema_version": 1, "game": game, "sources": [], "runs": [],
        "nodes": [], "proposals": [], "evidence": {},
        "limitations": list(LIMITATIONS), "insights": None, "telemetry": {},
        "status": "statistics_only", "language": "en", "score_statistics": [], "score_sources": [],
    }
    seen = set()
    for run_index, source in enumerate(inputs, 1):
        path = Path(source).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate trajectory input: {path}")
        seen.add(digest)
        state = json.loads(raw)
        if not isinstance(state, dict) or state.get("method") not in METHODS:
            raise ValueError(f"Unsupported trajectory method: {path}")
        if state.get("schema_version") not in (5, 6, 7) or not isinstance(state.get("nodes"), list):
            raise ValueError(f"Expected trajectory schema 5, 6 or 7 with nodes: {path}")
        embedded_game = state.get("game")
        if embedded_game is not None and embedded_game != game:
            raise ValueError(f"Trajectory game mismatch: expected {game}, got {embedded_game}: {path}")
        method = METHODS[state["method"]]
        run_id = f"run_{run_index:03d}"
        report["sources"].append({"run_id": run_id, "method": method, "path": str(path), "sha256": digest,
                                  "input_schema_version": state["schema_version"],
                                  "game_identity": "embedded" if embedded_game is not None else "user_declared"})
        nodes = []
        proposals = []
        node_ids = set()
        for node_index, node in enumerate(state["nodes"], 1):
            if not isinstance(node, dict) or not isinstance(node.get("node_id"), str):
                raise ValueError(f"Invalid trajectory node in {path}")
            if node["node_id"] in node_ids:
                raise ValueError(f"Duplicate node_id in {path}: {node['node_id']}")
            node_ids.add(node["node_id"])
            ref = f"{run_id}/node_{node_index:04d}"
            parent, current = _number(node.get("parent_score")), _number(node.get("current_score"))
            delta = current - parent if parent is not None and current is not None else None
            # Never use a stored delta alone as proof of a complete score pair.
            stored_delta = _number(node.get("score_delta"))
            if delta is not None and stored_delta is not None and not math.isclose(delta, stored_delta, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Inconsistent score_delta: {path}: {node['node_id']}")
            entry = {
                "evidence_id": ref, "run_id": run_id, "method": method,
                "node_id": node["node_id"], "parent_id": node.get("parent_id"),
                "status": node.get("status", "unknown"),
                "accepted": node.get("accepted") if isinstance(node.get("accepted"), bool) else None,
                "selected": node.get("selected") if isinstance(node.get("selected"), bool) else None,
                "parent_score": parent, "current_score": current, "score_delta": delta,
                "proposal_count": None,
            }
            analysis = node.get("analysis")
            if analysis is not None:
                parsed = AnalysisOutput.model_validate(analysis)
                if parsed.method != state["method"] or parsed.round != node["node_id"]:
                    raise ValueError(f"Analysis/node identity mismatch: {path}: {node['node_id']}")
                entry["proposal_count"] = len(parsed.opinions)
                for opinion_index, opinion in enumerate(parsed.opinions):
                    proposal = {
                        "evidence_id": f"{ref}/proposal_{opinion_index:04d}",
                        "node_evidence_id": ref, "run_id": run_id, "method": method,
                        "node_id": node["node_id"], "proposal_index": opinion_index,
                        **opinion.model_dump(mode="json"),
                    }
                    proposals.append(proposal)
                    report["evidence"][proposal["evidence_id"]] = proposal
            nodes.append(entry)
            report["evidence"][ref] = entry
        counts = {dimension: Counter() for dimension in DIMENSIONS}
        unknown_relations = Counter()
        for proposal in proposals:
            for dimension, labels in _labels(proposal).items():
                counts[dimension].update(labels)
            unknown_relations.update(name for name in ("semantically_new", "contradiction_or_revision", "duplicate")
                                     if proposal[name] is None)
        associations = []
        for dimension in DIMENSIONS:
            for label in sorted(counts[dimension]):
                matching = {proposal["node_evidence_id"] for proposal in proposals
                            if label in _labels(proposal)[dimension]}
                matched_nodes = [node for node in nodes if node["evidence_id"] in matching]
                associations.append({"dimension": dimension, "label": label,
                                     **_score_stats(matched_nodes), "evidence_ids": sorted(matching)})
        status_counts = dict(Counter(node["status"] for node in nodes))
        report["runs"].append({
            "run_id": run_id, "method": method, "analysis_status": state.get("status", "unknown"),
            "node_status_counts": status_counts, **_score_stats(nodes),
            "classified_node_count": sum(node["proposal_count"] is not None for node in nodes),
            "proposal_count": len(proposals),
            "accepted_node_count": sum(node["accepted"] is True for node in nodes),
            "rejected_node_count": sum(node["accepted"] is False for node in nodes),
            "unknown_acceptance_count": sum(node["accepted"] is None for node in nodes),
            "accepted_negative_node_count": sum(node["accepted"] is True and node["score_delta"] is not None
                                                and node["score_delta"] < 0 for node in nodes),
            "classification_counts": {key: dict(value) for key, value in counts.items()},
            "unknown_relation_counts": dict(unknown_relations), "category_score_associations": associations,
        })
        report["nodes"].extend(nodes)
        report["proposals"].extend(proposals)
    return report


SCORE_NUMBERS = (
    "rounds", "possible_pairs", "valid_pairs", "pairs", "positive", "zero", "negative",
    "mean_delta", "positive_rate", "complement_pairs", "complement_positive_rate",
    "rate_difference", "relative_rate_gain", "error_transition_pairs", "clean_positive", "clean_pairs",
    "initial_score", "final_score", "reference_score", "hci_ratio", "task_count", "valid_pair_count",
)
SCORE_CATEGORIES = {
    "semantically_new", "contradiction_or_revision", "duplicate", "exploration", "reasoning",
    "planning", "execution", "prompt_memory", "scaffold_codebase", "meta", "code",
    "task_specific", "capability_category_general", "fully_general", "no_annotated_opinions",
}


def attach_score_statistics(report: dict[str, Any], inputs: list[str | Path]) -> None:
    """Import numeric summaries only, never arbitrary text or source-log contents."""
    method_names = {key.lower(): value for key, value in METHODS.items()}
    run_methods = {run["method"] for run in report["runs"]}
    seen_sources = {source["sha256"] for source in report["score_sources"]}
    seen_rows = {(row["method"], row["kind"], row.get("metric"), row.get("category"))
                 for row in report["score_statistics"]}
    for source in inputs:
        path = Path(source).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_sources:
            raise ValueError(f"Duplicate score statistics input: {path}")
        seen_sources.add(digest)
        if path.suffix.lower() == ".json":
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("groups"), dict):
                raise ValueError(f"Expected benchmark.hci JSON or supported summary CSV: {path}")
            if any(not isinstance(group, dict) for group in data["groups"].values()):
                raise ValueError(f"Invalid HCI group: {path}")
            rows = [{"game": data.get("source_game"), "method": data.get("method"),
                     **{key: value for key, value in group.items() if key != "pairs"},
                     "hci_ratio": group.get("hci"), "score_group": group.get("difficulty_group", "replay"),
                     "hci_status": group.get("status"), "metric": metric}
                    for metric, group in data["groups"].items()]
        elif path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
        else:
            raise ValueError(f"Expected benchmark.hci JSON or supported summary CSV: {path}")
        added = 0
        for index, original in enumerate(rows, 2):
            if original.get("game") != report["game"]:
                continue
            method = method_names.get(str(original.get("method", "")).lower())
            if method not in run_methods:
                raise ValueError(f"Score method has no matching trajectory: {path}:{index}")
            if "category" in original and {"pairs", "positive", "zero", "negative", "mean_delta"} <= original.keys():
                kind = "category_adjacent_training_association"
                if original["category"] not in SCORE_CATEGORIES:
                    raise ValueError(f"Unknown score category: {path}:{index}")
            elif {"valid_pairs", "positive", "zero", "negative", "mean_delta"} <= original.keys():
                kind = "adjacent_training_rounds"
            elif {"metric", "initial_score", "final_score"} <= original.keys():
                kind = "initial_final_evaluation"
                if original["metric"] not in ("G0", "G1", "G2", "G3"):
                    raise ValueError(f"Unsupported evaluation metric: {path}:{index}")
                if original.get("target_game") not in GAMES:
                    raise ValueError(f"Unsupported target game: {path}:{index}")
                if original.get("score_group") not in ("medium", "hard", "replay"):
                    raise ValueError(f"Unsupported score group: {path}:{index}")
            else:
                raise ValueError(f"Unsupported score summary columns: {path}:{index}")
            row: dict[str, Any] = {"kind": kind, "game": report["game"], "method": method}
            for key in SCORE_NUMBERS:
                if key in original:
                    value = original[key]
                    if value is None or value == "":
                        row[key] = None
                    else:
                        try:
                            number = _number(float(value)) if not isinstance(value, bool) else None
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError(f"Invalid score statistic {key}: {path}:{index}") from exc
                        if number is None:
                            raise ValueError(f"Non-finite score statistic {key}: {path}:{index}")
                        row[key] = number
            metadata = ("category",) if kind == "category_adjacent_training_association" else (
                ("metric", "target_game", "score_group") if kind == "initial_final_evaluation" else ())
            for key in metadata:
                if key in original:
                    row[key] = original[key]
            if kind == "initial_final_evaluation":
                status = original.get("hci_status")
                if status not in (None, "", "ok", "undefined", "undefined_zero_headroom", "unavailable"):
                    raise ValueError(f"Unsupported HCI status: {path}:{index}")
                row["hci_status"] = status or "unspecified"
            else:
                count = row.get("valid_pairs", row.get("pairs"))
                counts = [row.get(key) for key in ("positive", "zero", "negative")]
                if count is None or any(v is None or v < 0 or not v.is_integer() for v in [count, *counts]):
                    raise ValueError(f"Invalid score pair counts: {path}:{index}")
                if sum(counts) != count:
                    raise ValueError(f"Inconsistent score pair counts: {path}:{index}")
            identity = (method, kind, row.get("metric"), row.get("category"))
            if identity in seen_rows:
                raise ValueError(f"Ambiguous duplicate score summary: {path}:{index}")
            seen_rows.add(identity)
            row["evidence_id"] = f"score_{len(report['score_statistics']) + 1:04d}"
            report["score_statistics"].append(row)
            added += 1
        if not added:
            raise ValueError(f"No matching score rows for {report['game']}: {path}")
        report["score_sources"].append({"path": str(path), "sha256": digest, "row_count": added})


def build_statistics(report: dict[str, Any]) -> dict[str, Any]:
    """Construct the complete model input with an explicit statistics-only allowlist."""
    evidence = []
    for run in report["runs"]:
        ref = run["run_id"]
        overall = {key: run[key] for key in (
            "method", "analysis_status", "node_count", "classified_node_count", "proposal_count",
            "scored_node_count", "missing_score_count", "positive", "zero", "negative", "mean_score_delta",
            "accepted_node_count", "rejected_node_count", "unknown_acceptance_count", "accepted_negative_node_count",
        )}
        evidence.append({"evidence_id": f"{ref}/summary", "kind": "trajectory_classification_overview", **overall})
        total = run["proposal_count"]
        for dimension, counts in run["classification_counts"].items():
            for label, count in counts.items():
                evidence.append({"evidence_id": f"{ref}/classification/{dimension}/{label}",
                                 "kind": "proposal_classification", "method": run["method"],
                                 "dimension": dimension, "label": label, "count": count,
                                 "denominator": total, "proportion": count / total if total else None})
        for label, count in run["unknown_relation_counts"].items():
            evidence.append({"evidence_id": f"{ref}/unknown/{label}", "kind": "unknown_relation",
                             "method": run["method"], "label": label, "count": count})
        for association in run["category_score_associations"]:
            evidence.append({"evidence_id": f"{ref}/association/{association['dimension']}/{association['label']}",
                             "kind": "category_native_node_score_association", "method": run["method"],
                             **{key: value for key, value in association.items() if key != "evidence_ids"}})
    evidence.extend(report["score_statistics"])
    return {"schema_version": 1, "game": report["game"], "input_scope": "classification_and_score_statistics_only",
            "limitations": report["limitations"] + [
                "Imported score summaries retain their stated metric; no scores or HCI are recomputed here.",
                "Adjacent training scores may use different tasks. Their differences are not controlled improvement effects.",
                "Final evaluation and proposal proportions are method-level observations, not evidence of proposal-level effects.",
                "Score summaries are matched by declared game/method, not an independently verified shared run lineage.",
            ], "evidence_catalogue": evidence}


REPORT_INSTRUCTIONS = """Write a concise, human-readable analysis of classification and score statistics.
You receive only numeric summaries and taxonomy labels, NOT proposal text, code changes or raw logs.
Analyze framework differences in novelty/revision/duplication, scope, function and modification surface,
and relate these patterns to the supplied score outcomes. Do not guess what individual proposals said,
diagnose specific code bugs, or recommend changes to game-playing strategies.
Separate observed findings from possible framework characteristics (hypotheses). Cite exact evidence_ids
for every statement. Prefer interesting, concrete comparisons with numbers over generic warnings.
Return at most three findings and two hypotheses. Write all report text in English and keep it within
roughly 450 words. Use short observation/interpretation fields; implications and next experiments
should concern statistical interpretation or controlled validation, not concrete agent modifications.
Use only supplied statistics; do not invent scores, counts or missing values. Missing is not zero.
Category/score associations are not causal effects. Labels can overlap; proportions need not sum to one.
The same node can have several labels. No significance, independent-seed or cross-game claims.
Distinguish native parent/current node deltas, adjacent-training-round deltas and initial/final evaluation
gains. Adjacent rounds may use different tasks. Acceptance is not a game score or proof of improvement.
Method-level final scores cannot identify which proposal category caused a gain. Check score coverage.
Classification labels are model annotations, not independent ground truth. Return the requested JSON schema.
"""


def add_insights(
    report: dict[str, Any], *, model: str, timeout: float | None = None,
    client: Any = None, max_input_chars: int = 240_000,
) -> None:
    """Attach model interpretation only after validating every evidence reference."""
    payload = build_statistics(report)
    prefix = REPORT_INSTRUCTIONS + "\nRequested language: en\n"
    prompt = prefix + json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(prompt) > max_input_chars:
        raise ValueError("Statistics exceed the model input character limit; split inputs")
    if client is None:
        client = _create_openai_client(None, None, timeout)
    allowed_refs = {entry["evidence_id"] for entry in payload["evidence_catalogue"]}
    try:
        response = client.responses.parse(model=model, input=prompt, text_format=ReportInsights)
        _record_telemetry(response, "game_report", report["telemetry"])
        output = getattr(response, "output_parsed", None)
        if output is None:
            raise ValueError("No structured report returned")
        parsed = ReportInsights.model_validate(output)
        for statement in (*parsed.findings, *parsed.hypotheses):
            if not statement.evidence_ids or any(ref not in allowed_refs for ref in statement.evidence_ids):
                raise ValueError(f"Unknown or missing evidence reference: {statement.title}")
            if len(set(statement.evidence_ids)) != len(statement.evidence_ids):
                raise ValueError(f"Duplicate evidence reference: {statement.title}")
    except Exception as exc:
        raise AnalysisLLMError(f"Game report interpretation failed: {exc}") from exc
    report["insights"] = parsed.model_dump(mode="json")
    report["model"] = model
    report["returned_model"] = getattr(response, "model", None)
    report["language"] = "en"
    report["status"] = "interpreted"
    report["interpretation_scope"] = payload["input_scope"]


def _cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {report['game']} Classification and Score Report", ""]
    if report.get("data_note"):
        lines += [_cell(report["data_note"]), ""]
    lines += ["## Classification Statistics", "",
              "| Method | Classified / nodes | Proposals | Status |", "| --- | --- | --- | --- |"]
    for run in report["runs"]:
        row = [run["method"], f"{run['classified_node_count']} / {run['node_count']}",
               run["proposal_count"], run["analysis_status"]]
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    lines += ["", "| Dimension / label | " + " | ".join(run["method"] for run in report["runs"]) + " |",
              "| --- | " + " | ".join("---" for _ in report["runs"]) + " |"]
    for dimension in DIMENSIONS:
        labels = sorted({label for run in report["runs"] for label in run["classification_counts"][dimension]})
        for label in labels:
            values = []
            for run in report["runs"]:
                count = run["classification_counts"][dimension].get(label, 0)
                total = run["proposal_count"]
                values.append(f"{count} ({count / total:.1%})" if total else "N/A")
            name = f"{dimension} / {label}"
            lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "Proportions use each framework's annotated proposals; relation/function labels may overlap. Unknowns: statistics.json.", "",
              "## Score Statistics"]
    adjacent = [row for row in report["score_statistics"] if row["kind"] == "adjacent_training_rounds"]
    if adjacent:
        lines += ["", "Adjacent training rounds (may span tasks, not controlled gains):", "",
                  "| Method | Pairs | + / 0 / - | Mean delta |", "| --- | --- | --- | --- |"]
        for row in adjacent:
            values = [row["method"], row["valid_pairs"], f"{row['positive']:g} / {row['zero']:g} / {row['negative']:g}", row["mean_delta"]]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    evaluations = [row for row in report["score_statistics"] if row["kind"] == "initial_final_evaluation"]
    if evaluations:
        lines += ["", "| Method | Metric | Target / group | Initial | Final | HCI |", "| --- | --- | --- | --- | --- | --- |"]
        for item in evaluations:
            row = [item["method"], item["metric"], f"{item['target_game']} / {item['score_group']}",
                   item.get("initial_score"), item.get("final_score"), item.get("hci_ratio")]
            lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    if any(run["scored_node_count"] for run in report["runs"]):
        lines += ["", "Native parent/current deltas (not adjacent training deltas):", "",
                  "| Method | Scored / nodes | + / 0 / - | Mean delta |", "| --- | --- | --- | --- |"]
        for run in report["runs"]:
            row = [run["method"], f"{run['scored_node_count']} / {run['node_count']}",
                   f"{run['positive']} / {run['zero']} / {run['negative']}", run["mean_score_delta"]]
            lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    elif not adjacent and not evaluations:
        lines += ["", "N/A: no scores available."]
    insights = report["insights"]
    for key, title in (("findings", "Findings"),
                       ("hypotheses", "Possible Characteristics (Hypotheses)")):
        lines += ["", f"## {title}", ""]
        if insights is None:
            lines.append("No report model was called; statistics only.")
            continue
        if not insights[key]:
            lines.append("Insufficient evidence.")
        for index, statement in enumerate(insights[key], 1):
            lines += [f"{index}. **{_cell(statement['title'])}** " + " ".join(
                _cell(statement[field]) for field in ("observation", "interpretation", "implication") if statement[field]), ""]
    lines += ["", "Labels come from existing LLM annotations; associations are descriptive, not causal. N/A is not zero.",
              "Interpretation is limited to statistics.json summaries, not proposals, code or logs. Evidence references, validation ideas and provenance remain in JSON."]
    if report.get("interpretation_error"):
        lines += ["", "Model interpretation failed; statistics retained: "
                  + _cell(report["interpretation_error"])]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    protected = {Path(source["path"]).resolve() for source in report["sources"] + report["score_sources"]}
    paths = [directory / name for name in ("report.json", "report.md", "nodes.csv", "proposals.csv", "statistics.json")]
    if any(path.resolve() in protected for path in paths):
        raise ValueError("Report output must not overwrite an input trajectory")
    paths[0].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths[1].write_text(render_markdown(report), encoding="utf-8")
    paths[4].write_text(json.dumps(build_statistics(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for path, key, fallback in ((paths[2], "nodes", "node_id"), (paths[3], "proposals", "normalized_opinion")):
        rows = report[key]
        fields = list(rows[0]) if rows else ["evidence_id", fallback]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                                 for name, value in row.items()})


def run_report(args: Any) -> None:
    report = build_report(args.inputs, args.game)
    attach_score_statistics(report, getattr(args, "score_stats", None) or [])
    report.update(requested_model=args.model)
    write_report(report, args.output_dir)
    if args.model:
        try:
            add_insights(report, model=args.model, timeout=args.timeout)
        except (ValueError, AnalysisLLMError) as exc:
            report.update(status="interpretation_failed", interpretation_error=str(exc))
            write_report(report, args.output_dir)
            raise
        write_report(report, args.output_dir)
    print(f"Saved {args.game} trajectory report ({report['status']}) to {args.output_dir}")
