"""Command-line entry point for the analyze package."""

import argparse
import json
import os
from typing import Any

from .ace import run_ace_analysis
from .ahe import run_ahe_analysis
from .hyper import run_hyper_analysis
from .report import GAMES, run_report


def _positive_int(value: str) -> int:
    """Parse a node count that must be greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _build_parser() -> argparse.ArgumentParser:
    """Create the method-level subcommands."""
    parser = argparse.ArgumentParser(description="Analyze the improvement artifacts of an RSI method")
    commands = parser.add_subparsers(dest="command", required=True)
    ace = commands.add_parser("ace", help="Analyze ACE tasks in order")
    ace.add_argument("experiment_dir", help="ACE experiment directory")
    ace.add_argument("--output", default="ace_analysis.json", help="Output path for the trace state")
    ace.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="Model name")
    ace.add_argument("--limit", type=_positive_int, help="Analyze only the first N nodes")
    ace.add_argument("--timeout", type=float, help="Timeout in seconds for a single API request")
    hyper = commands.add_parser("hyper", help="Analyze HyperAgent candidates by parent-child tree")
    hyper.add_argument("experiment_dir", help="HyperAgent experiment directory")
    hyper.add_argument("--output", default="hyper_analysis.json", help="Output path for the trace state")
    hyper.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="Model name")
    hyper.add_argument("--limit", type=_positive_int, help="Analyze only the first N nodes in topological order")
    hyper.add_argument("--timeout", type=float, help="Timeout in seconds for a single API request")
    ahe = commands.add_parser("ahe", help="Analyze AHE iterations in order")
    ahe.add_argument("experiment_dir", help="AHE experiment directory")
    ahe.add_argument("--output", default="ahe_analysis.json", help="Output path for the trace state")
    ahe.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="Model name")
    ahe.add_argument("--limit", type=_positive_int, help="Analyze only the first N iterations")
    ahe.add_argument("--timeout", type=float, help="Timeout in seconds for a single API request")
    report = commands.add_parser("report", help="Create an evidence-linked, per-game trajectory report")
    report.add_argument("--game", required=True, choices=GAMES, help="Game shared by all input trajectories")
    report.add_argument("--input", dest="inputs", nargs="+", required=True, help="Existing trajectory analysis JSON files")
    report.add_argument("--output-dir", required=True, help="Directory for Markdown, JSON and CSV reports")
    report.add_argument("--model", help="Optional interpretation model; omit for offline statistics only")
    report.add_argument("--language", choices=("zh", "en"), default="zh", help="Report language")
    report.add_argument("--timeout", type=float, help="Timeout in seconds for the report API request")
    return parser


def _print_node(node: dict[str, Any]) -> None:
    """Print the single-round state and the four statistic dimensions."""
    print(
        f"[{node['node_id']}] parent_score={node.get('parent_score')} "
        f"current_score={node.get('current_score')} score_delta={node.get('score_delta')}"
    )
    if node["status"] != "completed":
        print(f"[{node['node_id']}] {node['status']}")
        return
    summary = node["summary"]
    print(f"[{node['node_id']}] opinions={summary['opinion_count']}")
    for dimension in ("relations", "scope", "function", "surface"):
        content = json.dumps(summary[dimension], ensure_ascii=False)
        print(f"  {dimension}: {content}")


def main() -> None:
    """Parse arguments and run the control loop of the selected method."""
    args = _build_parser().parse_args()
    if args.command == "report":
        try:
            run_report(args)
        except (OSError, ValueError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
    elif args.command == "ace":
        if not args.model:
            raise SystemExit("Specify a model via --model or OPENAI_MODEL")
        state = run_ace_analysis(
            args.experiment_dir,
            args.output,
            model=args.model,
            limit=args.limit,
            timeout=args.timeout,
            on_node=_print_node,
        )
        if state["status"] != "completed":
            raise SystemExit(
                f"ACE analysis {state['status']}: "
                f"{state.get('missing_artifact_count', 0)} missing artifacts; "
                f"partial results saved to {args.output}"
            )
        print(f"Done: {len(state['nodes'])} nodes written to {args.output}")
    elif args.command == "hyper":
        if not args.model:
            raise SystemExit("Specify a model via --model or OPENAI_MODEL")
        state = run_hyper_analysis(
            args.experiment_dir,
            args.output,
            model=args.model,
            limit=args.limit,
            timeout=args.timeout,
            on_node=_print_node,
        )
        print(f"Done: {len(state['nodes'])} nodes written to {args.output}")
    elif args.command == "ahe":
        if not args.model:
            raise SystemExit("Specify a model via --model or OPENAI_MODEL")
        state = run_ahe_analysis(
            args.experiment_dir,
            args.output,
            model=args.model,
            limit=args.limit,
            timeout=args.timeout,
            on_node=_print_node,
        )
        print(f"Done: {len(state['nodes'])} nodes written to {args.output}")


if __name__ == "__main__":
    main()
