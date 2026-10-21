"""Legacy import and constructor alias for the shared GameWorld transport."""

from typing import Any

from .actor_input import TRANSPORT_NOTE
from .rsi_game_client import RSIGameClient


class ACEGameClient(RSIGameClient):
    agent_label = "ACE"

    def __init__(self, config: Any, ace_agent: Any, **shared_tools: Any) -> None:
        super().__init__(config, ace_agent, **shared_tools)
        self._ace_agent = ace_agent


__all__ = ["ACEGameClient", "TRANSPORT_NOTE"]
