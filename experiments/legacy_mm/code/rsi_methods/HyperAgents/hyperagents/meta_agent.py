# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from pathlib import Path

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def _inspect_game_screenshots(self, eval_path):
        """Make staged game images visible to the model, not just to its shell."""
        root = Path(eval_path).resolve()
        result_path = root / "result.json"
        if not result_path.is_file():
            return ""
        result = json.loads(result_path.read_text(encoding="utf-8"))
        references = []
        for case in result.get("cases", []):
            for step in (case.get("feedback") or {}).get("trajectory") or []:
                for image in step.get("image_records") or []:
                    path = root / image["path"]
                    resolved = path.resolve()
                    if root not in resolved.parents or not resolved.is_file():
                        raise FileNotFoundError(f"Staged evaluation screenshot is inaccessible: {path}")
                    references.append((image["path"], resolved))
        if not references:
            return ""
        # All images remain in the evidence folder. Bound the multimodal call
        # for long trajectories and sample across the entire ordered trace.
        if len(references) > 8:
            references = [references[round(i * (len(references) - 1) / 7)] for i in range(8)]
        prompt = (
            "These are ordered screenshots from an agent's completed game evaluation. "
            "Describe the visible state and any objective-relevant change for each image "
            "in order. State uncertainty; do not invent hidden goals, scores, or actions. "
            "The corresponding evidence paths, in image order, are: "
            + ", ".join(name for name, _ in references)
        )
        history = chat_with_agent(
            prompt, model=self.model, msg_history=[], logging=self.log,
            tools_available=[], image_paths=[str(path) for _, path in references],
            reasoning_effort=self.reasoning_effort, backend=self.backend,
            api_mode=self.api_mode, api_key_env=self.api_key_env,
            base_url_env=self.base_url_env,
        )
        last = history[-1] if history else {}
        observation = last.get("content") or last.get("text") or ""
        if not isinstance(observation, str) or not observation.strip():
            raise RuntimeError("Screenshot inspection returned no visual observations")
        output = Path(self.logger_manager.log_file).parent / "screenshot_analysis.json"
        output.write_text(json.dumps({
            "image_paths": [name for name, _ in references],
            "observation": observation, "usage": last.get("usage"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return observation

    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        instruction = f"""Improve the HyperAgent implementation in `{repo_path}`.

Your objective is to make a generalizable code change that increases the
`selection_score` produced by the evaluator. Evaluation history is available
read-only at `{eval_path}`.

Before editing code:
1. Recursively inspect the evaluation-history folder. Read
   `evidence_manifest.json` when present, then read `result.json` and relevant
   artifacts. Older histories may instead contain `run_config.json`,
   `archive.jsonl`, and per-candidate metadata.
   Artifact paths in `result.json` are relative to `{eval_path}`. Verify the
   referenced files before diagnosing a failure pattern, and ground the
   diagnosis in concrete actions and observations.
2. A batch `result.json` contains aggregate metrics plus a `cases` list. Each
   case carries its own `metrics`, `feedback`, and `artifacts`; `case_id`
   identifies one evaluated case. Compare cases instead of treating the
   aggregate score as a single trace.
3. Identify failure patterns that recur across cases. Distinguish a
   systematic weakness from stochastic one-off failures, and do not optimize
   for one case ID, prompt, or expected answer.
4. Inspect the current codebase and connect the evidence-backed pattern to code
   that can be improved.

Then edit the codebase at `{repo_path}`. Do not modify evaluation-history files,
the evaluator, or its recorded results. Do not hard-code a solution for one
task instance. Preserve the public TaskAgent and MetaAgent interfaces. Prefer a
small, testable change and run lightweight checks when useful.

Framework boundary: `exp/overall_conrtoll_loop/` is fixed model-transport
infrastructure. Do not edit files under that directory.

Strategy surface: if the failure pattern is an agent-strategy weakness, edit
`task_agent.py::forward()`. You may add general action-policy guidance to the
instruction that is forwarded to the model (for example: observe before acting;
do not repeat an action without a new observation). Keep the forwarded input
concise: do not dump the full evaluation history, long transcripts, or result
payloads into the model input; include only the current observation and a short
relevant context. Do not leak task names, answers, ground-truth boards, or
seeds, and do not special-case one board or seed. You do not need to run the
benchmark evaluator yourself; a submitted change will be evaluated externally.

Future MetaAgent invocations remaining after this one: {iterations_left}.
You must make the code edits yourself using the available tools."""

        visual_observations = self._inspect_game_screenshots(eval_path)
        if visual_observations:
            instruction += ("\nThe screenshots were inspected by the same model in a "
                            "separate multimodal call. Visual observations (not ground "
                            "truth) follow; check them against the scored trajectory "
                            "before changing code:\n" + visual_observations[:6000])
        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            reasoning_effort=self.reasoning_effort,
            backend=self.backend,
            api_mode=self.api_mode,
            api_key_env=self.api_key_env,
            base_url_env=self.base_url_env,
        )
        return new_msg_history
