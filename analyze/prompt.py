"""构造改进产物分析所需的 LLM 提示词。"""

import json
from typing import Any


ANALYSIS_INSTRUCTIONS = """You analyze improvement artifacts produced by RSI methods.
Your goal is to describe, in a structured form, how the system was improved. Do not attempt to reconstruct the hidden process that generated an artifact.

Perform the analysis in two phases, in this exact order. Finish Phase A for the entire artifact before starting Phase B.

Phase A — Semantic decomposition:
1. Decompose the entire current-round artifact before normalization, comparison, or classification. Extract every occurrence of an improvement proposal and establish one fixed opinions item for each atomic proposal, in source order.
   - A proposal is atomic when it can be adopted or rejected independently from neighboring changes.
   - Determine proposal boundaries from meaning, not presentation. A JSONL record, field, paragraph, bullet, operation, file, or diff hunk may contain zero, one, or multiple proposals. Conversely, several adjacent fragments may jointly express one proposal.
   - Do not treat the number of source records, operations, bullets, or changed lines as the expected number of proposals.
   - Split a record, paragraph, bullet, or diff hunk when it contains multiple independently meaningful changes, even if they share the same topic.
   - Merge fragments within the same occurrence when they express the rule, condition, rationale, example, or implementation detail of one inseparable change.
   - Split independently adoptable semantic changes that modify different surfaces. Code that only stores, concatenates, or injects prompt or memory text is a delivery mechanism for that text, not a separate proposal and not a scaffold_codebase change.
   - Do not merge repeated proposals. Preserve every occurrence so that later occurrences can be classified as duplicates.
   - Descriptions of existing behavior, observations, outcomes, rationale, examples, warnings, and implementation details are not proposals by themselves. Keep them with the proposal they support unless they independently prescribe a change. Code used only to deliver a prompt change is such an implementation detail.
   - Imperative wording does not by itself prove that text contains a separate proposal, and descriptive wording does not by itself prove that no proposal exists. Judge the prescribed semantic change.
   - For a replacement in a diff, describe the semantic change from the removed behavior to the added behavior as one proposal. For a pure addition, describe the added behavior. For a pure deletion, describe the removal. Ignore unchanged diff context.
   - If the artifact contains no improvement proposal, return an empty opinions list.

Before Phase B, silently verify the fixed proposal list by asking for each item: what independently adoptable change does it prescribe? Merge or remove an item when no independent change can be identified. Split an item only when it contains multiple independently adoptable changes. Once Phase B begins, do not add, remove, merge, or split opinions items.

Phase B — Normalization, comparison, and classification:
2. Write a concise English normalized_opinion for each proposal. Preserve the modified object, applicable conditions, and essential change.
   For each opinion, copy the exact source fragments that express that occurrence into source_evidence. Do not paraphrase evidence. Include enough text to audit why the fragments form one semantic unit. Do not include carrier code that only delivers prompt or memory text unless the carrier itself is an independent scaffold proposal.
3. The prior-round opinions are the parent's refined snapshot. Each prior opinion has an opinion_id. Compare each current proposal with that snapshot and with earlier proposals from the current round. Judge these three properties independently:
   - semantically_new: the proposal is semantically distinct. This does not mean that it improves performance.
   - contradiction_or_revision: the proposal contradicts or substantially revises an existing proposal.
   - duplicate: the proposal is semantically repeated.
   semantically_new and contradiction_or_revision may both be true. duplicate cannot be true together with either of them.
   - When contradiction_or_revision is true, list every overwritten parent-snapshot ID in replaces_prior_opinion_ids. If it overwrites an earlier proposal from this response, list that proposal's zero-based opinions index in replaces_current_opinion_indices.
   - Replacement indices must refer only to earlier items in the current opinions array. Replacement IDs must exactly match IDs supplied in the parent snapshot.
   - When contradiction_or_revision is false, both replacement lists must be empty.
   - A duplicate is counted but discarded from the refined snapshot; its replacement lists must be empty.
4. Classify scope as a single value:
   - task_specific: the proposal depends on the content, terminology, structure, objective, answer format, or solution mechanism of one concrete task or named task family. An improvement that applies to multiple Hero's Journey instances is still task_specific when it depends on the Hero's Journey structure or recipe.
   - capability_category_general: the proposal applies across distinct concrete tasks because they require the same broad capability, such as numerical reasoning, constraint solving, causal inference, evidence reconciliation, information retrieval, or sequential decision making. It must not depend on the mechanics of one named task family.
   - fully_general: the proposal is independent of both a concrete task and a particular capability category, and remains applicable across substantially different tasks and capabilities.
   Use semantic dependency rather than the number of tasks mentioned. A named task family is not automatically capability_category_general. Select the broadest scope directly supported by the proposal, not the broadest scope for which an application can be imagined. Use null when the evidence is insufficient.
5. Classify function as a list containing every function directly changed by the proposal:
   - exploration: changes how evidence or observations are acquired through search, inspection, experimentation, or feedback.
   - reasoning: changes how existing evidence is interpreted or used to derive beliefs, values, or conclusions. This includes calculation, estimation, comparison, inference, and constraint solving.
   - planning: changes how future actions are selected, ordered, budgeted, allocated, or adapted to contingencies.
   - execution: changes how selected actions are carried out, submitted, retried, formatted, or operationally validated.
   Do not assign planning merely because reasoning is eventually used to act. Assign planning only when action selection, ordering, resource allocation, or contingency handling is directly changed. If a record contains separable proposals affecting different functions, split them into separate opinions before classification.
6. Classify surface as a single value:
   - prompt_memory: changes instructions, supplied context, or stored memory.
   - scaffold_codebase: changes executable logic, workflow, or other operational codebase components.
   - meta: changes the improver or the improvement process itself.
   Classify the content actually changed, rather than its filename. A prompt embedded in a code file is still prompt_memory. Concatenating, storing, or injecting that prompt does not create an additional scaffold_codebase opinion when the code has no independently meaningful behavioral change beyond delivering the text. Use scaffold_codebase only when executable logic or workflow changes independently of the delivered prompt content. Use meta only when the improver or improvement process is directly changed. If one semantic occurrence contains independently adoptable changes to multiple surfaces, split those changes before classification. Use null when the evidence is insufficient.
7. The labels above are closed sets. Every label must exactly match an allowed value. Do not invent, rename, translate, abbreviate, or combine labels. Function lists must contain unique values only, must not contain null, and must be empty when evidence is insufficient.
8. semantically_new, contradiction_or_revision, and duplicate must each be true, false, or null. Do not encode these values as strings. Use null when the evidence is insufficient.
9. Do not infer changes from file extensions, possible modification capabilities, downstream consequences, or predicted effects.
10. Before returning, verify that every item fixed in Phase A has exactly one corresponding output item and that Phase B did not change the Phase A item count or boundaries.
11. Return valid JSON with exactly the specified fields. Do not return explanations, Markdown, code fences, or additional fields.

The opinions array may contain zero, one, or many items. Its length must equal the number of atomic proposal occurrences extracted from the artifact.

Calibrated semantic-boundary example:
A patch adds one guidance block and concatenates that block before the task instruction. Decompose the guidance into exactly the following five opinions because each can be adopted or rejected without adopting the others:
1. Treat demonstrations as constraints and infer target-attribute mappings by contrasting examples. Labels: capability_category_general; reasoning; prompt_memory.
2. Compose only minimally supported action subsequences, without unions, maximized counts, speculative backups, or unsupported additivity. Labels: capability_category_general; planning; prompt_memory.
3. Preserve demonstrated counts, relative order, and placement around travel, and solve their ordering constraints. Labels: task_specific; reasoning and planning; prompt_memory.
4. Audit planned actions against evidence, reserve budget for travel and the goal, and prefer the shortest supported sequence. Labels: task_specific; reasoning and planning; prompt_memory.
5. Follow the requested output format exactly. Labels: fully_general; execution; prompt_memory.
The concatenation code is only how these prompt changes are delivered. Do not create a sixth opinion for it and do not label it scaffold_codebase. This example calibrates semantic boundaries and does not prescribe five opinions for other artifacts.

Output structure:
{
  "method": "method name",
  "round": "round identifier",
  "opinions": [
    {
      "normalized_opinion": "concise normalized improvement proposal",
      "source_evidence": ["exact supporting source fragment"],
      "scope": "task_specific",
      "function": ["reasoning", "planning"],
      "surface": "prompt_memory",
      "semantically_new": true,
      "contradiction_or_revision": false,
      "duplicate": false,
      "replaces_prior_opinion_ids": [],
      "replaces_current_opinion_indices": []
    },
    {
      "normalized_opinion": "another independently extracted improvement proposal",
      "source_evidence": ["exact supporting source fragment"],
      "scope": "capability_category_general",
      "function": ["exploration"],
      "surface": "scaffold_codebase",
      "semantically_new": true,
      "contradiction_or_revision": true,
      "duplicate": false,
      "replaces_prior_opinion_ids": [],
      "replaces_current_opinion_indices": [0]
    }
  ]
}"""


def _serialize(value: Any) -> str:
    """保留纯文本；其他 reader 结果转成便于 LLM 阅读的 JSON。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_analysis_prompt(
    method: str,
    round_id: str | int,
    artifacts: Any,
    prior_opinions: Any = None,
) -> str:
    """将分析规则、历史意见和本轮原始产物拼接为一次 LLM 输入。"""
    history = [] if prior_opinions is None else prior_opinions
    return "\n\n".join(
        [
            ANALYSIS_INSTRUCTIONS,
            "Analysis input:",
            f"method: {method}",
            f"round: {round_id}",
            f"Refined opinion snapshot from the parent node:\n{_serialize(history)}",
            f"Raw artifacts from the current round:\n{_serialize(artifacts)}",
        ]
    )
