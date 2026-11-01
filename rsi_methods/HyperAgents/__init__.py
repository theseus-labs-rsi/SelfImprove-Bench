"""Outer controller; host imports use the nested vendored package explicitly."""

# Candidate scripts retain upstream imports inside their container; the host
# does not expose their top-level agent/utils modules on sys.path.

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
