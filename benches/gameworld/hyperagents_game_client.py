"""Legacy import and constructor alias for the shared GameWorld transport."""

from typing import Any

from .actor_input import TRANSPORT_NOTE
from .rsi_game_client import RSIGameClient


class HyperAgentsGameClient(RSIGameClient):
    agent_label = "HyperAgents"

    def __init__(self, config: Any, runtime_agent: Any, **shared_tools: Any) -> None:
        super().__init__(config, runtime_agent, **shared_tools)
        self._runtime_agent = runtime_agent


__all__ = ["HyperAgentsGameClient", "TRANSPORT_NOTE"]
