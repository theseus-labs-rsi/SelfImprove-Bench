"""Outer HyperAgents controller with compatibility for the vendored source tree."""

from pathlib import Path
import sys


_VENDORED_SOURCE = str(Path(__file__).resolve().parent / "hyperagents")
if _VENDORED_SOURCE not in sys.path:
    sys.path.insert(0, _VENDORED_SOURCE)

from .candidate import CandidateSnapshot, initialize_agent, with_lineage
from .candidate_registry import (
    CandidateRecord,
    CandidateRegistry,
    load_candidate_snapshot,
)
from .HyperAgentsAdaptor import HyperAgentsAdaptor
from .lifecycle import (
    HyperAgentRuntime,
    InstantiatedAgent,
    RefineOutput,
    evaluate_agent,
    instantiate_agent,
    reconstruct_agent,
    refine,
)

__all__ = [
    "CandidateSnapshot",
    "CandidateRecord",
    "CandidateRegistry",
    "HyperAgentRuntime",
    "HyperAgentsAdaptor",
    "InstantiatedAgent",
    "RefineOutput",
    "evaluate_agent",
    "initialize_agent",
    "instantiate_agent",
    "load_candidate_snapshot",
    "reconstruct_agent",
    "refine",
    "with_lineage",
]
