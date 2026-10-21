# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from visual_evidence import collect_game_screenshots


class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        instruction = f"Modify any part of the codebase at `{repo_path}`."
        image_paths, image_context = collect_game_screenshots(eval_path)
        instruction += "\n\n" + (
            image_context or f"Evaluation records: {eval_path}/result.json"
        )
        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available="all",
            image_paths=image_paths,
            reasoning_effort=self.reasoning_effort,
            backend=self.backend,
            api_mode=self.api_mode,
            api_key_env=self.api_key_env,
            base_url_env=self.base_url_env,
        )
        return new_msg_history
