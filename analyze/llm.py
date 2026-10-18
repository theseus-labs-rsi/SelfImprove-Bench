"""Call the LLM to analyze a single round of RSI improvement artifacts."""

import json
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .prompt import build_analysis_prompt


class AnalysisLLMError(RuntimeError):
    """An LLM call or a structured-result validation failed."""


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
    """One atomic improvement opinion split and classified by the LLM."""

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
    """Structured output of a single-round analysis."""

    method: str
    round: str
    opinions: list[ImprovementOpinion]

    class Config:
        extra = "forbid"


def _recover_output_from_validation_error(exc: Exception) -> AnalysisOutput | None:
    """Extract the unique schema-valid JSON object from a structured parsing error."""
    errors = exc.errors() if hasattr(exc, "errors") else []
    raw_values = [item.get("input") for item in errors if isinstance(item, dict)]
    decoder = json.JSONDecoder()
    valid: list[AnalysisOutput] = []

    for raw in raw_values:
        if not isinstance(raw, str):
            continue
        for start, character in enumerate(raw):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(raw, start)
                valid.append(AnalysisOutput.model_validate(candidate))
            except (json.JSONDecodeError, ValueError):
                continue
    return valid[0] if len(valid) == 1 else None


def _create_openai_client(
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
) -> Any:
    """Create an OpenAI client from explicit arguments and environment variables."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AnalysisLLMError("OpenAI SDK is not installed") from exc

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise AnalysisLLMError("OPENAI_API_KEY is not provided")
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
    """Record the identifier and token usage of a single analysis response."""
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
    """Validate cross-field constraints beyond the schema."""
    if output.method != method:
        raise AnalysisLLMError(f"method mismatch: expected {method}, got {output.method}")
    if output.round != str(round_id):
        raise AnalysisLLMError(f"round mismatch: expected {round_id}, got {output.round}")

    prior_ids = _extract_prior_ids(prior_opinions)
    for index, opinion in enumerate(output.opinions):
        if not opinion.source_evidence or any(
            not evidence.strip() for evidence in opinion.source_evidence
        ):
            raise AnalysisLLMError(f"Opinion {index} is missing valid source_evidence")
        if len(opinion.function) != len(set(opinion.function)):
            raise AnalysisLLMError(f"Opinion {index} contains duplicate function labels")
        _validate_relations(output, index, prior_ids)


def _extract_prior_ids(prior_opinions: Any) -> set[str]:
    """Extract parent-snapshot IDs and reject legacy formats that cannot be tracked."""
    if prior_opinions is None:
        return set()
    if not isinstance(prior_opinions, list):
        raise AnalysisLLMError("prior_opinions must be a list of opinions")

    ids: list[str] = []
    for index, opinion in enumerate(prior_opinions):
        if not isinstance(opinion, dict) or not isinstance(opinion.get("opinion_id"), str):
            raise AnalysisLLMError(f"Parent-snapshot opinion {index} is missing opinion_id")
        ids.append(opinion["opinion_id"])
    if len(ids) != len(set(ids)):
        raise AnalysisLLMError("Parent snapshot contains duplicate opinion_id")
    return set(ids)


def _validate_relations(
    output: AnalysisOutput,
    index: int,
    prior_ids: set[str],
) -> None:
    """Validate that the relation labels and replacement targets can safely update the snapshot."""
    opinion = output.opinions[index]
    if opinion.duplicate is True and (
        opinion.semantically_new is True or opinion.contradiction_or_revision is True
    ):
        raise AnalysisLLMError(f"Duplicate opinion {index} is also marked as new or revised")

    relations = (
        opinion.semantically_new,
        opinion.contradiction_or_revision,
        opinion.duplicate,
    )
    if all(value is False for value in relations):
        raise AnalysisLLMError(f"All three relation labels of opinion {index} are false")
    _validate_replacement_targets(output, index, prior_ids)


def _validate_replacement_targets(
    output: AnalysisOutput,
    index: int,
    prior_ids: set[str],
) -> None:
    """A revision target may only reference the parent snapshot or an earlier accepted opinion in this round."""
    opinion = output.opinions[index]
    prior_targets = opinion.replaces_prior_opinion_ids
    current_targets = opinion.replaces_current_opinion_indices
    if len(prior_targets) != len(set(prior_targets)):
        raise AnalysisLLMError(f"Opinion {index} contains duplicate parent-snapshot replacement IDs")
    if len(current_targets) != len(set(current_targets)):
        raise AnalysisLLMError(f"Opinion {index} contains duplicate current-round replacement indices")
    unknown_ids = set(prior_targets) - prior_ids
    if unknown_ids:
        raise AnalysisLLMError(f"Opinion {index} references unknown parent-snapshot IDs: {sorted(unknown_ids)}")

    for target in current_targets:
        if target < 0 or target >= index:
            raise AnalysisLLMError(f"Opinion {index} references an invalid current-round index: {target}")
        replaced = output.opinions[target]
        if replaced.duplicate is True or not (
            replaced.semantically_new is True
            or replaced.contradiction_or_revision is True
        ):
            raise AnalysisLLMError(f"Opinion {index} references a current-round index that did not enter the snapshot: {target}")

    has_targets = bool(prior_targets or current_targets)
    if opinion.contradiction_or_revision is True and not has_targets:
        raise AnalysisLLMError(f"Revision opinion {index} does not specify a replacement target")
    if opinion.contradiction_or_revision is not True and has_targets:
        raise AnalysisLLMError(f"Non-revision opinion {index} contains a replacement target")


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
    """Build the prompt, call the model, and return the validated single-round analysis result."""
    selected_model = model or os.getenv("OPENAI_MODEL")
    if not selected_model:
        raise AnalysisLLMError("Specify a model via the model argument or OPENAI_MODEL")
    if client is None:
        client = _create_openai_client(api_key, base_url, timeout)

    prompt = build_analysis_prompt(method, round_id, artifacts, prior_opinions)
    response = None
    try:
        response = client.responses.parse(
            model=selected_model,
            input=prompt,
            text_format=AnalysisOutput,
        )
    except Exception as exc:
        output = _recover_output_from_validation_error(exc)
        if output is None:
            raise AnalysisLLMError(f"LLM analysis request failed: {exc}") from exc
        if telemetry is not None:
            telemetry["structured_parse_recovered"] = True
    else:
        output = getattr(response, "output_parsed", None)

    if output is None:
        raise AnalysisLLMError("LLM did not return a parseable structured result")
    _validate_output(output, method, round_id, prior_opinions)
    if response is not None:
        _record_telemetry(response, "analysis", telemetry)
    return output
