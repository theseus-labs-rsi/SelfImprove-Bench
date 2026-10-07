"""HyperAgents-style evolution over a DeepSeek Harness execution surface."""

from .HyperDSHAdaptor import (
    HyperDSHAdaptor,
    HyperDSHCandidateBatch,
    HyperDSHCandidateSpec,
    HyperDSHSelectionOutput,
    HyperDSHSelectionReport,
)
from .runtime_agent import HyperDSHAgent

__all__ = [
    "HyperDSHAdaptor",
    "HyperDSHAgent",
    "HyperDSHCandidateBatch",
    "HyperDSHCandidateSpec",
    "HyperDSHSelectionOutput",
    "HyperDSHSelectionReport",
]
