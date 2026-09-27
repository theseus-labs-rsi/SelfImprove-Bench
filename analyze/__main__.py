"""analyze 包的命令行入口。"""

import argparse
import json
import os
from typing import Any

from .ace import run_ace_analysis
from .ahe import run_ahe_analysis
from .hyper import run_hyper_analysis


def _positive_int(value: str) -> int:
    """解析大于零的结点数量。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def _build_parser() -> argparse.ArgumentParser:
    """创建方法级子命令。"""
    parser = argparse.ArgumentParser(description="分析 RSI 方法的改进产物")
    commands = parser.add_subparsers(dest="command", required=True)
    ace = commands.add_parser("ace", help="顺序分析 ACE 任务")
    ace.add_argument("experiment_dir", help="ACE 实验目录")
    ace.add_argument("--output", default="ace_analysis.json", help="追踪状态输出路径")
    ace.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="模型名称")
    ace.add_argument("--limit", type=_positive_int, help="仅分析前 N 个结点")
    ace.add_argument("--timeout", type=float, help="单次 API 请求超时秒数")
    hyper = commands.add_parser("hyper", help="按父子树分析 HyperAgent 候选")
    hyper.add_argument("experiment_dir", help="HyperAgent 实验目录")
    hyper.add_argument("--output", default="hyper_analysis.json", help="追踪状态输出路径")
    hyper.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="模型名称")
    hyper.add_argument("--limit", type=_positive_int, help="仅分析拓扑序前 N 个结点")
    hyper.add_argument("--timeout", type=float, help="单次 API 请求超时秒数")
    ahe = commands.add_parser("ahe", help="顺序分析 AHE iteration")
    ahe.add_argument("experiment_dir", help="AHE 实验目录")
    ahe.add_argument("--output", default="ahe_analysis.json", help="追踪状态输出路径")
    ahe.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="模型名称")
    ahe.add_argument("--limit", type=_positive_int, help="仅分析前 N 个 iteration")
    ahe.add_argument("--timeout", type=float, help="单次 API 请求超时秒数")
    return parser


def _print_node(node: dict[str, Any]) -> None:
    """输出单轮状态和四个统计维度。"""
    if node["status"] != "completed":
        print(f"[{node['node_id']}] {node['status']}")
        return
    summary = node["summary"]
    print(f"[{node['node_id']}] opinions={summary['opinion_count']}")
    for dimension in ("relations", "scope", "function", "surface"):
        content = json.dumps(summary[dimension], ensure_ascii=False)
        print(f"  {dimension}: {content}")


def main() -> None:
    """解析参数并调用相应方法的控制循环。"""
    args = _build_parser().parse_args()
    if args.command == "ace":
        if not args.model:
            raise SystemExit("请通过 --model 或 OPENAI_MODEL 指定模型")
        state = run_ace_analysis(
            args.experiment_dir,
            args.output,
            model=args.model,
            limit=args.limit,
            timeout=args.timeout,
            on_node=_print_node,
        )
        print(f"完成：{len(state['nodes'])} 个结点，输出到 {args.output}")
    elif args.command == "hyper":
        if not args.model:
            raise SystemExit("请通过 --model 或 OPENAI_MODEL 指定模型")
        state = run_hyper_analysis(
            args.experiment_dir,
            args.output,
            model=args.model,
            limit=args.limit,
            timeout=args.timeout,
            on_node=_print_node,
        )
        print(f"完成：{len(state['nodes'])} 个结点，输出到 {args.output}")
    elif args.command == "ahe":
        if not args.model:
            raise SystemExit("请通过 --model 或 OPENAI_MODEL 指定模型")
        state = run_ahe_analysis(
            args.experiment_dir,
            args.output,
            model=args.model,
            limit=args.limit,
            timeout=args.timeout,
            on_node=_print_node,
        )
        print(f"完成：{len(state['nodes'])} 个结点，输出到 {args.output}")


if __name__ == "__main__":
    main()
