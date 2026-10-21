"""Legacy import and constructor alias for the shared GameWorld transport."""

from typing import Any

from .actor_input import TRANSPORT_NOTE
from .rsi_game_client import RSIGameClient


class AHEGameClient(RSIGameClient):
    agent_label = "AHE"

    def __init__(self, config: Any, ahe_agent: Any, **shared_tools: Any) -> None:
        super().__init__(config, ahe_agent, **shared_tools)
        self._ahe_agent = ahe_agent


__all__ = ["AHEGameClient", "TRANSPORT_NOTE"]
