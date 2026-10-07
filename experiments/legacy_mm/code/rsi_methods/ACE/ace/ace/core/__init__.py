"""
Core module for ACE system.
Contains the three main agent classes: Generator, Reflector, and Curator.
"""

from .generator import Generator
from .reflector import Reflector
from .curator import Curator

__all__ = ['Generator', 'Reflector', 'Curator', 'BulletpointAnalyzer', 'DEDUP_AVAILABLE']


def __getattr__(name):
    """Load optional bulletpoint-analysis dependencies only when requested."""
    if name not in {"BulletpointAnalyzer", "DEDUP_AVAILABLE"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    bulletpoint_analyzer = import_module(f"{__name__}.bulletpoint_analyzer")
    value = getattr(bulletpoint_analyzer, name)
    globals()[name] = value
    return value
