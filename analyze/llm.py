"""调用 LLM 分析单轮 RSI 改进产物。"""

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .prompt import build_analysis_prompt


class AnalysisLLMError(RuntimeError):
    """LLM 调用或结构化结果校验失败。"""


class ScopeLabel(str, Enum):
    TASK_SPECIFIC = "task_specific"
    CAPABILITY_CATEGORY_GENERAL = "capability_category_general"
    FULLY_GENERAL = "fully_general"


class FunctionLabel(str, Enum):
    EXPLORATION = "exploration"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"


class SurfaceLabel(str, Enum):
    PROMPT_MEMORY = "prompt_memory"
    SCAFFOLD_CODEBASE = "scaffold_codebase"
    META = "meta"


class ImprovementOpinion(BaseModel):
    """一条由 LLM 拆分并分类的原子改进意见。"""

    normalized_opinion: str
    source_evidence: list[str]
    scope: ScopeLabel | None
    function: list[FunctionLabel]
    surface: SurfaceLabel | None
    semantically_new: bool | None
    contradiction_or_revision: bool | None
    duplicate: bool | None
    replaces_prior_opinion_ids: list[str]
    replaces_current_opinion_indices: list[int]

    class Config:
        extra = "forbid"


class AnalysisOutput(BaseModel):
    """单轮分析的结构化输出。"""

    method: str
    round: str
    opinions: list[ImprovementOpinion]

    class Config:
        extra = "forbid"


def _create_openai_client(
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
) -> Any:
    """按显式参数和环境变量创建 OpenAI 客户端。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AnalysisLLMError("OpenAI SDK 未安装") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise AnalysisLLMError("未提供 OPENAI_API_KEY")
    options: dict[str, Any] = {"api_key": key}
    endpoint = base_url or os.getenv("OPENAI_BASE_URL")
    if endpoint:
        options["base_url"] = endpoint
    if timeout is not None:
        options["timeout"] = timeout
    return OpenAI(**options)


def _record_telemetry(
    response: Any,
    stage: str,
    telemetry: dict[str, Any] | None,
) -> None:
    """保留单次分析响应的标识和 token 用量。"""
    if telemetry is None:
        return
    usage = getattr(response, "usage", None)
    record = {
        "stage": stage,
        "response_id": getattr(response, "id", None),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
    }
    telemetry.setdefault("responses", []).append(record)


def _validate_output(
    output: AnalysisOutput,
    method: str,
    round_id: str | int,
    prior_opinions: Any,
) -> None:
    """校验 schema 之外的跨字段约束。"""
    if output.method != method:
        raise AnalysisLLMError(f"method 不匹配：期望 {method}，实际 {output.method}")
    if output.round != str(round_id):
        raise AnalysisLLMError(f"round 不匹配：期望 {round_id}，实际 {output.round}")

    prior_ids = _extract_prior_ids(prior_opinions)
    for index, opinion in enumerate(output.opinions):
        if not opinion.source_evidence or any(
            not evidence.strip() for evidence in opinion.source_evidence
        ):
            raise AnalysisLLMError(f"第 {index} 条意见缺少有效 source_evidence")
        if len(opinion.function) != len(set(opinion.function)):
            raise AnalysisLLMError(f"第 {index} 条意见包含重复 function 标签")
        _validate_relations(output, index, prior_ids)


def _extract_prior_ids(prior_opinions: Any) -> set[str]:
    """提取父快照 ID，并拒绝无法用于追踪的历史格式。"""
    if prior_opinions is None:
        return set()
    if not isinstance(prior_opinions, list):
        raise AnalysisLLMError("prior_opinions 必须是意见列表")

    ids: list[str] = []
    for index, opinion in enumerate(prior_opinions):
        if not isinstance(opinion, dict) or not isinstance(opinion.get("opinion_id"), str):
            raise AnalysisLLMError(f"父快照第 {index} 条意见缺少 opinion_id")
        ids.append(opinion["opinion_id"])
    if len(ids) != len(set(ids)):
        raise AnalysisLLMError("父快照包含重复 opinion_id")
    return set(ids)


def _validate_relations(
    output: AnalysisOutput,
    index: int,
    prior_ids: set[str],
) -> None:
    """校验关系标签与替换目标是否能够安全更新快照。"""
    opinion = output.opinions[index]
    if opinion.duplicate is True and (
        opinion.semantically_new is True or opinion.contradiction_or_revision is True
    ):
        raise AnalysisLLMError(f"第 {index} 条重复意见同时被标记为新增或修订")

    relations = (
        opinion.semantically_new,
        opinion.contradiction_or_revision,
        opinion.duplicate,
    )
    if all(value is False for value in relations):
        raise AnalysisLLMError(f"第 {index} 条意见的三个关系标签均为 false")
    _validate_replacement_targets(output, index, prior_ids)


def _validate_replacement_targets(
    output: AnalysisOutput,
    index: int,
    prior_ids: set[str],
) -> None:
    """修订目标只能引用父快照或本轮已接受的较早意见。"""
    opinion = output.opinions[index]
    prior_targets = opinion.replaces_prior_opinion_ids
    current_targets = opinion.replaces_current_opinion_indices
    if len(prior_targets) != len(set(prior_targets)):
        raise AnalysisLLMError(f"第 {index} 条意见包含重复的父快照替换 ID")
    if len(current_targets) != len(set(current_targets)):
        raise AnalysisLLMError(f"第 {index} 条意见包含重复的本轮替换索引")
    unknown_ids = set(prior_targets) - prior_ids
    if unknown_ids:
        raise AnalysisLLMError(f"第 {index} 条意见引用未知父快照 ID：{sorted(unknown_ids)}")

    for target in current_targets:
        if target < 0 or target >= index:
            raise AnalysisLLMError(f"第 {index} 条意见引用无效的本轮索引：{target}")
        replaced = output.opinions[target]
        if replaced.duplicate is True or not (
            replaced.semantically_new is True
            or replaced.contradiction_or_revision is True
        ):
            raise AnalysisLLMError(f"第 {index} 条意见引用了未进入快照的本轮索引：{target}")

    has_targets = bool(prior_targets or current_targets)
    if opinion.contradiction_or_revision is True and not has_targets:
        raise AnalysisLLMError(f"第 {index} 条修订意见没有指定替换目标")
    if opinion.contradiction_or_revision is not True and has_targets:
        raise AnalysisLLMError(f"第 {index} 条非修订意见包含替换目标")


def analyze_artifact(
    method: str,
    round_id: str | int,
    artifacts: Any,
    prior_opinions: Any = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    client: Any = None,
    telemetry: dict[str, Any] | None = None,
) -> AnalysisOutput:
    """构造提示词、调用模型并返回经过校验的单轮分析结果。"""
    selected_model = model or os.getenv("OPENAI_MODEL")
    if not selected_model:
        raise AnalysisLLMError("请通过 model 参数或 OPENAI_MODEL 指定模型")
    if client is None:
        client = _create_openai_client(api_key, base_url, timeout)

    prompt = build_analysis_prompt(method, round_id, artifacts, prior_opinions)
    try:
        response = client.responses.parse(
            model=selected_model,
            input=prompt,
            text_format=AnalysisOutput,
        )
    except Exception as exc:
        raise AnalysisLLMError(f"LLM 分析请求失败：{exc}") from exc

    output = getattr(response, "output_parsed", None)
    if output is None:
        raise AnalysisLLMError("LLM 未返回可解析的结构化结果")
    _validate_output(output, method, round_id, prior_opinions)
    _record_telemetry(response, "analysis", telemetry)
    return output
