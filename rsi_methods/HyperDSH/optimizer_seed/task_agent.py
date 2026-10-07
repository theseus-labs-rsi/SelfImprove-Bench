"""Compatibility stub; HyperDSH tasks are executed by DeepSeek Harness."""


class TaskAgent:
    def __init__(self, **kwargs):
        self.config = dict(kwargs)

    def act(self, *args, **kwargs):
        raise RuntimeError(
            "HyperDSH task execution must use the host HyperDSHAgent bridge"
        )
