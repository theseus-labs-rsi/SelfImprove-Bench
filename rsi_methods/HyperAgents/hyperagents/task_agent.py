from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class TaskAgent(AgentSystem):
    def act(self, prompt, image_paths=None, max_tokens=16384):
        """Expose the common Agent interface while retaining ``forward``."""
        if image_paths:
            raise NotImplementedError(
                "HyperAgents TaskAgent cannot read container image paths yet."
            )
        return self.forward(
            {
                "instruction": prompt,
                "max_tokens": max_tokens,
            }
        )

    def forward(self, inputs):
        """
        An agent that solves a given task.

        Args:
            inputs (dict): A dictionary with input data for the task.

        Returns:
            tuple:
                - prediction (str): The prediction made by the agent.
                - new_msg_history (list): A list of messages representing the message history of the interaction.
        """
        instruction = inputs["instruction"]
        max_attempts = 3
        prediction = None
        new_msg_history = None
        for attempt in range(max_attempts):
            candidate_history = chat_with_agent(
                instruction,
                model=self.model,
                msg_history=[],
                logging=self.log,
                reasoning_effort=self.reasoning_effort,
                backend=self.backend,
                api_mode=self.api_mode,
                api_key_env=self.api_key_env,
                base_url_env=self.base_url_env,
                max_tokens=inputs.get("max_tokens", 16384),
            )
            # Preserve the task-defined output format for the benchmark to parse.
            if candidate_history and isinstance(candidate_history[-1], dict):
                candidate_text = candidate_history[-1].get("text")
                if isinstance(candidate_text, str) and candidate_text.strip():
                    prediction = candidate_text
                    new_msg_history = candidate_history
                    break
            self.log(
                f"TaskAgent received an empty model response "
                f"(attempt {attempt + 1}/{max_attempts}); retrying."
            )

        if prediction is None or new_msg_history is None:
            raise RuntimeError(
                "TaskAgent received an empty model response "
                f"after {max_attempts} attempts."
            )
        return prediction, new_msg_history
