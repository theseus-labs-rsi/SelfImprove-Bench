from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class TaskAgent(AgentSystem):
    def act(self, prompt, image_paths=None, max_tokens=16384):
        """Expose the common Agent interface while retaining ``forward``."""
        return self.forward(
            {
                "instruction": prompt,
                "image_paths": list(image_paths or ()),
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
        attempt_usages = []
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
                image_paths=inputs.get("image_paths") or (),
            )
            # Count *all* successful provider attempts, including empty answers retried here.
            last = candidate_history[-1] if candidate_history else {}
            attempt_usages.append(last.get("usage") if isinstance(last, dict) else None)
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
        fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")
        if all(isinstance(item, dict) and isinstance(item.get("total_tokens"), int)
               for item in attempt_usages):
            new_msg_history[-1]["usage"] = {
                field: sum(item[field] for item in attempt_usages)
                       if all(isinstance(item.get(field), int) for item in attempt_usages) else None
                for field in fields
            }
        else:
            new_msg_history[-1]["usage"] = None
        new_msg_history[-1]["usage_attempts"] = attempt_usages
        return prediction, new_msg_history
