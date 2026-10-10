"""读取并组织各 RSI method 的原始改进产物。"""

import difflib
import filecmp
import json
import math
import warnings
from pathlib import Path
from typing import Any


# ===== 辅助方法 =====

def _read_utf8_text(path: str | Path) -> str:
    """读取 UTF-8 文本，并为编码错误补充文件位置。"""
    source = Path(path)
    try:
        return source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"无法按 UTF-8 读取文件：{source}") from exc


def _read_json_object(path: str | Path) -> dict[str, Any]:
    """读取 JSON 对象，并拒绝不符合预期的顶层类型。"""
    source = Path(path)
    try:
        content = json.loads(_read_utf8_text(source))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无效 JSON：{source}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"JSON 顶层必须是对象：{source}")
    return content


def _workspace_files(root: Path) -> dict[Path, Path]:
    """列出 workspace 文件，排除候选自带的 Git 元数据。"""
    return {
        path.relative_to(root): path
        for path in root.rglob("*")
        if path.is_file()
        and not {".git", "__pycache__"}.intersection(path.relative_to(root).parts)
        and path.suffix.lower() not in {".pyc", ".pyo"}
    }


def _read_workspace_text(path: Path | None) -> str:
    """读取 workspace 文件；缺失路径表示新增或删除文件。"""
    return "" if path is None else _read_utf8_text(path)


def _read_ahe_workspace_diff(candidate: Path) -> str:
    """比较同轮基线与 AHE 候选 workspace，不依赖 Git 历史。"""
    iteration = candidate.parent.name
    state = candidate.parents[2]
    baseline = state / "runs" / iteration / "input" / "workspace"
    evolved = candidate / "workspace"
    for root in (baseline, evolved):
        if not root.is_dir():
            raise FileNotFoundError(f"AHE workspace 不存在：{root}")

    old_files = _workspace_files(baseline)
    new_files = _workspace_files(evolved)
    chunks: list[str] = []
    for relative in sorted(old_files.keys() | new_files.keys()):
        old_path, new_path = old_files.get(relative), new_files.get(relative)
        if old_path and new_path and filecmp.cmp(old_path, new_path, shallow=False):
            continue
        from_name = f"a/{relative.as_posix()}" if old_path else "/dev/null"
        to_name = f"b/{relative.as_posix()}" if new_path else "/dev/null"
        try:
            old_text = _read_workspace_text(old_path)
            new_text = _read_workspace_text(new_path)
        except ValueError:
            chunks.append(f"Binary files {from_name} and {to_name} differ\n")
            continue
        chunks.extend(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=from_name,
            tofile=to_name,
        ))
    return "".join(chunks)


# ===== 单个产物读取方法 =====

def read_ace_operations(path: str | Path) -> list[dict[str, Any]]:
    """从指定的 ACE JSONL 文件读取原始操作记录。"""
    source = Path(path)
    records: list[dict[str, Any]] = []

    # 逐行解析，不改写或归一化记录中的字段。
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                operation = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid ACE JSONL at {source}:{line_number}") from exc
            if not isinstance(operation, dict):
                raise ValueError(f"ACE operation must be an object at {source}:{line_number}")
            records.append(operation)

    return records


def _read_ace_curator_operations(task_dir: Path) -> list[dict[str, Any]] | None:
    """从 trace task 的 curator 响应读取本轮操作；缺失文件返回 None。"""
    sources = sorted(task_dir.glob("curator_refine_*.json"))
    if not sources:
        return None
    if len(sources) != 1:
        raise ValueError(f"ACE curator 响应文件不唯一：{task_dir}")

    record = _read_json_object(sources[0])
    response = record.get("response")
    if not isinstance(response, str):
        raise ValueError(f"ACE curator response 必须是字符串：{sources[0]}")
    try:
        content = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ACE curator response 不是有效 JSON：{sources[0]}") from exc
    operations = content.get("operations") if isinstance(content, dict) else None
    if not isinstance(operations, list) or not all(isinstance(item, dict) for item in operations):
        raise ValueError(f"ACE curator operations 必须是对象列表：{sources[0]}")
    return operations


def read_hyperagent_patch(path: str | Path) -> str:
    """从指定的 HyperAgent model_patch.diff 读取原始补丁。"""
    return _read_utf8_text(path)


def read_ahe_candidate(candidate_dir: str | Path) -> dict[str, str]:
    """读取指定 AHE 候选的概述和 workspace 详细差异。"""
    candidate = Path(candidate_dir)
    return {
        "evolve_result": _read_utf8_text(candidate / "evolve_result.json"),
        "workspace_diff": _read_ahe_workspace_diff(candidate),
    }


# ===== 方法级组织方法 =====

def _score_fields(
    report: dict[str, Any] | None, candidate_id: str | None = None,
) -> dict[str, Any]:
    """读取实际父代与候选评分；缺失或非有限数值保留为空。"""
    report = report or {}
    parent = report.get("parent_score")
    scores = report.get("candidate_scores", {})
    current = scores.get(candidate_id) if isinstance(scores, dict) else None
    if "candidate_score" in report and (
        candidate_id is None or report.get("selected_candidate_id") == candidate_id
    ):
        current = report["candidate_score"]
    valid = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    parent = parent if valid(parent) else None
    current = current if valid(current) else None
    return {
        "parent_score": parent, "current_score": current,
        "score_delta": current - parent if current is not None and parent is not None else None,
    }


def collect_ace_scores(experiment_dir: str | Path) -> dict[str, dict[str, Any]]:
    """按 task 读取逐轮选择结果，不改变 ACE 产物读取接口。"""
    root = Path(experiment_dir).resolve()
    tasks = root.glob("task_*") if (root / "batch_manifest.json").is_file() else root.glob("rsi/ace/*/task_*")
    return {task.name: _score_fields(_read_hyper_selection(task)) for task in tasks if task.is_dir()}


def _evaluation_scores(
    candidate_dir: Path, root: Path, fallback: dict[str, Any],
) -> dict[str, Any]:
    """优先读取候选 result，并按 parent_run_id 匹配同轮父代结果。"""
    path = candidate_dir / "evaluation" / "result.json"
    if not path.is_file():
        return fallback
    result = _read_json_object(path)
    current = result.get("relative_score") if result.get("status") == "ok" else None
    parent = fallback["parent_score"]
    run_id = result.get("parent_run_id")
    if isinstance(run_id, str):
        matches = [path for path in (root / "runs" / "train").glob("*.json") if path.stem.partition("_")[2] == run_id]
        if len(matches) > 1:
            raise ValueError(f"父代评估结果不唯一：{run_id}")
        if matches:
            record = _read_json_object(matches[0])
            parent = record.get("relative_score") if record.get("status") == "ok" else None
    return _score_fields({"parent_score": parent, "candidate_score": current})

def collect_ace_artifacts(
    experiment_dir: str | Path,
) -> list[tuple[str, list[dict[str, Any]] | None]]:
    """从 ACE 实验目录读取各 task 的原始改进记录。"""
    source_root = Path(experiment_dir).resolve()
    if (source_root / "batch_manifest.json").is_file():
        task_dirs = sorted(path for path in source_root.iterdir() if _is_task_dir(path))
        if not task_dirs:
            raise FileNotFoundError(f"ACE trace task 目录不存在：{source_root}")
        return [
            (task_dir.name, _read_ace_curator_operations(task_dir))
            for task_dir in task_dirs
        ]

    ace_root = source_root / "rsi" / "ace"
    if not ace_root.is_dir():
        raise FileNotFoundError(f"ACE 运行目录不存在：{ace_root}")

    run_dirs = sorted(path for path in ace_root.iterdir() if path.is_dir())
    if len(run_dirs) != 1:
        names = [path.name for path in run_dirs]
        raise ValueError(f"ACE 运行目录应当唯一，实际找到 {len(run_dirs)} 个：{names}")

    task_dirs = sorted(
        path for path in run_dirs[0].iterdir()
        if path.is_dir() and path.name.startswith("task_")
    )
    if not task_dirs:
        raise FileNotFoundError(f"ACE task 目录不存在：{run_dirs[0]}")

    artifacts: list[tuple[str, list[dict[str, Any]] | None]] = []
    for task_dir in task_dirs:
        source = task_dir / "candidates" / "curator_operations_diff.jsonl"
        content = read_ace_operations(source) if source.is_file() else None
        artifacts.append((task_dir.name, content))
    return artifacts


def collect_hyperagent_artifacts(
    experiment_dir: str | Path,
) -> list[dict[str, Any]]:
    """读取唯一有效 HyperAgent run，并按显式 parent_id 排序候选。"""
    return _collect_hyper_tree_artifacts(
        experiment_dir, method_dir="hyperagents", method_name="HyperAgent",
    )


def collect_hyperdsh_artifacts(
    experiment_dir: str | Path,
) -> list[dict[str, Any]]:
    """读取唯一有效 HyperDSH run，并按显式 parent_id 排序候选。"""
    return _collect_hyper_tree_artifacts(
        experiment_dir, method_dir="hyperdsh", method_name="HyperDSH",
    )


def _collect_hyper_tree_artifacts(
    experiment_dir: str | Path,
    *,
    method_dir: str,
    method_name: str,
) -> list[dict[str, Any]]:
    """按方法目录读取 HyperAgent 风格的候选树。"""
    source_root = Path(experiment_dir).resolve()
    if (source_root / "batch_manifest.json").is_file():
        return _read_hyper_trace_batch(source_root, method_name)

    root = source_root / "rsi" / method_dir
    if not root.is_dir():
        raise FileNotFoundError(f"{method_name} 运行目录不存在：{root}")

    run_dirs = [path for path in root.iterdir() if _has_hyper_tasks(path)]
    if len(run_dirs) != 1:
        names = [path.name for path in run_dirs]
        raise ValueError(f"{method_name} 有效 run 应当唯一，实际找到 {len(run_dirs)} 个：{names}")

    nodes: list[dict[str, Any]] = []
    task_dirs = sorted(path for path in run_dirs[0].iterdir() if _is_task_dir(path))
    for task_dir in task_dirs:
        nodes.extend(_read_hyper_task(task_dir, run_dirs[0].name, method_name))
    return _order_hyper_nodes(nodes, method_name)


def _read_hyper_trace_batch(root: Path, method_name: str) -> list[dict[str, Any]]:
    """从已选定单候选的 HyperAgent trace batch 重建父子链。"""
    manifest = _read_json_object(root / "batch_manifest.json")
    run_id = manifest.get("batch_id")
    if not isinstance(run_id, str):
        raise ValueError(f"{method_name} batch 缺少 batch_id：{root}")

    nodes: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in root.iterdir() if _is_task_dir(path)):
        report = _read_json_object(task_dir / "refinement_report.json")
        metadata = report.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"HyperAgent refinement metadata 无效：{task_dir}")
        nodes.append(_build_hyper_trace_node(task_dir, run_id, report, metadata))
    if not nodes:
        raise FileNotFoundError(f"{method_name} trace task 目录不存在：{root}")
    return _order_hyper_nodes(nodes, method_name)


def _build_hyper_trace_node(
    task_dir: Path,
    run_id: str,
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """把单候选 task 报告转换为统一 HyperAgent 结点。"""
    candidate_id = report.get("candidate_id")
    raw_parent = metadata.get("parent_genid")
    if not isinstance(candidate_id, str):
        raise ValueError(f"HyperAgent trace 缺少 candidate_id：{task_dir}")
    if raw_parent not in (None, "initial") and not isinstance(raw_parent, str):
        raise ValueError(f"HyperAgent trace parent_genid 无效：{candidate_id}")

    patch_path = task_dir / "agent_output" / "model_patch.diff"
    accepted = metadata.get("refine_successful")
    return {
        "node_id": candidate_id,
        **_evaluation_scores(task_dir / "agent_output", task_dir.parent,
                             _score_fields(_read_hyper_selection(task_dir), candidate_id)),
        "parent_id": None if raw_parent in (None, "initial") else raw_parent,
        "task_id": task_dir.name,
        "run_id": run_id,
        "changed": bool(report.get("changed")),
        "selected": True,
        "accepted": accepted if isinstance(accepted, bool) else None,
        "artifact_path": str(patch_path),
        "artifact": read_hyperagent_patch(patch_path) if patch_path.is_file() else None,
    }


def _is_task_dir(path: Path) -> bool:
    """判断目录是否为方法生成的 task 目录。"""
    return path.is_dir() and path.name.startswith("task_")


def _has_hyper_tasks(path: Path) -> bool:
    """空 run 不参与唯一性判断。"""
    return path.is_dir() and any(_is_task_dir(child) for child in path.iterdir())


def _read_hyper_task(
    task_dir: Path,
    run_id: str,
    method_name: str,
) -> list[dict[str, Any]]:
    """读取一个 task 中的全部候选及其选择状态。"""
    batch_path = task_dir / "candidate_batch.json"
    if not batch_path.is_file():
        warnings.warn(
            f"跳过缺少 candidate_batch.json 的 {method_name} task：{task_dir}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    batch = _read_json_object(batch_path)
    candidates = batch.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"HyperAgent candidates 必须是列表：{batch_path}")

    selection = _read_hyper_selection(task_dir)
    return [
        _build_hyper_node(task_dir, run_id, candidate, selection)
        for candidate in candidates
    ]


def _read_hyper_selection(task_dir: Path) -> dict[str, Any] | None:
    """读取 task 的候选选择结果；缺失时保留未知状态。"""
    paths = sorted(task_dir.glob("selection_stage_*/selection.json"))
    if len(paths) > 1:
        raise ValueError(f"HyperAgent selection 文件不唯一：{task_dir}")
    return _read_json_object(paths[0]) if paths else None


def _build_hyper_node(
    task_dir: Path,
    run_id: str,
    candidate: Any,
    selection: dict[str, Any] | None,
) -> dict[str, Any]:
    """使用本地目录重建候选路径，避免依赖元数据中的旧绝对路径。"""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
        raise ValueError(f"HyperAgent candidate 缺少 candidate_id：{task_dir}")
    candidate_id = candidate["candidate_id"]
    raw_parent = candidate.get("parent_id")
    parent_id = None if raw_parent in (None, "initial") else raw_parent
    if parent_id is not None and not isinstance(parent_id, str):
        raise ValueError(f"HyperAgent parent_id 无效：{candidate_id}")

    patch_path = task_dir / "candidates" / candidate_id / "agent_output" / "model_patch.diff"
    selected_id = selection.get("selected_candidate_id") if selection else None
    selected = candidate_id == selected_id if selected_id else None
    accepted = selection.get("accepted") if selected is True else (False if selected_id else None)
    return {
        "node_id": candidate_id,
        **_evaluation_scores(patch_path.parent, task_dir.parents[3],
                             _score_fields(selection, candidate_id)),
        "parent_id": parent_id,
        "task_id": task_dir.name,
        "run_id": run_id,
        "changed": bool(candidate.get("changed")),
        "selected": selected,
        "accepted": accepted,
        "artifact_path": str(patch_path),
        "artifact": read_hyperagent_patch(patch_path) if patch_path.is_file() else None,
    }


def _order_hyper_nodes(
    nodes: list[dict[str, Any]],
    method_name: str = "HyperAgent",
) -> list[dict[str, Any]]:
    """按 parent_id 拓扑排序，同时检查重复、缺失父结点和环。"""
    by_id = {node["node_id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError(f"{method_name} 包含重复 candidate_id")
    ordered: list[dict[str, Any]] = []
    states: dict[str, int] = {}

    def visit(node_id: str) -> None:
        state = states.get(node_id, 0)
        if state == 1:
            raise ValueError(f"{method_name} parent_id 形成环：{node_id}")
        if state == 2:
            return
        states[node_id] = 1
        parent_id = by_id[node_id]["parent_id"]
        if parent_id is not None:
            if parent_id not in by_id:
                raise ValueError(f"{method_name} 父结点不存在：{node_id} -> {parent_id}")
            visit(parent_id)
        states[node_id] = 2
        ordered.append(by_id[node_id])

    for node in nodes:
        visit(node["node_id"])
    return ordered


def _ahe_scores(
    run_dir: Path, iteration: str, variant_dir: Path,
    report: dict[str, Any] | None, experiment_dir: Path,
) -> dict[str, Any]:
    """缺少选择报告时，只从同轮输入结果补读可确认的父代分数。"""
    scores = _score_fields(report, variant_dir.name)
    if report is None:
        baseline = run_dir / "state" / "runs" / iteration / "input" / "benchmark"
        results = sorted(baseline.glob("*/*/result.json"))
        if len(results) == 1:
            record = _read_json_object(results[0])
            scores = _score_fields({"parent_score": record.get("score")})
    scores = _evaluation_scores(variant_dir, experiment_dir, scores)
    if report is None and scores["current_score"] is None:
        warnings.warn(
            f"AHE 候选缺少评分记录，当前分数保留为空：{iteration}:{variant_dir.name}",
            RuntimeWarning, stacklevel=2,
        )
    return scores


def collect_ahe_artifacts(
    experiment_dir: str | Path,
) -> list[dict[str, Any]]:
    """读取唯一有效 AHE run，并按 iteration 顺序组织单 variant 链。"""
    root = Path(experiment_dir).resolve() / "rsi" / "ahe"
    if not root.is_dir():
        raise FileNotFoundError(f"AHE 运行目录不存在：{root}")

    run_dirs = [path for path in root.iterdir() if _has_ahe_candidates(path)]
    if len(run_dirs) != 1:
        names = [path.name for path in run_dirs]
        raise ValueError(f"AHE 有效 run 应当唯一，实际找到 {len(run_dirs)} 个：{names}")

    candidates_root = run_dirs[0] / "state" / "candidates"
    iteration_dirs = sorted(
        path for path in candidates_root.iterdir()
        if path.is_dir() and path.name.startswith("iteration_")
    )
    if not iteration_dirs:
        raise FileNotFoundError(f"AHE iteration 目录不存在：{candidates_root}")

    nodes: list[dict[str, Any]] = []
    parent_id: str | None = None
    reports = {}
    for path in run_dirs[0].glob("task_*/selection_stage_*/ahe_selection.json"):
        report = _read_json_object(path)
        reports.setdefault(report.get("refinement_id"), []).append(report)
    for iteration_dir in iteration_dirs:
        variant_dir = _single_ahe_variant(iteration_dir)
        node_id = f"{iteration_dir.name}:{variant_dir.name}"
        matching = reports.get(iteration_dir.name, [])
        if len(matching) > 1:
            raise ValueError(f"AHE 评分结果不唯一：{iteration_dir.name}")
        nodes.append({
            "node_id": node_id,
            **_ahe_scores(run_dirs[0], iteration_dir.name, variant_dir,
                          matching[0] if matching else None, Path(experiment_dir).resolve()),
            "parent_id": parent_id,
            "iteration": iteration_dir.name,
            "variant": variant_dir.name,
            "run_id": run_dirs[0].name,
            "artifact_dir": str(variant_dir),
            "artifact": read_ahe_candidate(variant_dir),
        })
        parent_id = node_id
    return nodes


def _has_ahe_candidates(path: Path) -> bool:
    """判断 run 是否包含 AHE candidates 目录。"""
    candidates = path / "state" / "candidates"
    return candidates.is_dir() and any(candidates.iterdir())


def _single_ahe_variant(iteration_dir: Path) -> Path:
    """当前缺少选择元数据，因此只接受唯一 variant。"""
    variants = sorted(
        path for path in iteration_dir.iterdir()
        if path.is_dir() and path.name.startswith("variant_")
    )
    if len(variants) != 1:
        names = [path.name for path in variants]
        raise ValueError(
            f"AHE iteration 缺少唯一 variant，无法确定父子链：{iteration_dir} -> {names}"
        )
    return variants[0]
