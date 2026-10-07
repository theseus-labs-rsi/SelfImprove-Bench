from abc import ABC, abstractmethod
from utils.thread_logger import ThreadLoggerManager


class AgentSystem(ABC):
    def __init__(
        self,
        model,
        chat_history_file='./outputs/chat_history.md',
        reasoning_effort="medium",
        backend="openai",
        api_mode="responses",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.backend = backend
        self.api_mode = api_mode
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env

        # Initialize logger and store it in thread-local storage
        self.logger_manager = ThreadLoggerManager(log_file=chat_history_file)
        self.log = self.logger_manager.log

        # Clear the log file
        with open(chat_history_file, 'w') as f:
            f.write('')

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
