"""Hero's Journey benchmark adapter with lazy runtime imports."""

from .task_factory import HeroJourneyDifficulty

__all__ = [
    "HeroJourneyBenchmark",
    "HeroJourneyDifficulty",
    "HeroJourneyTask",
]


def __getattr__(name: str):
    if name in {"HeroJourneyBenchmark", "HeroJourneyTask"}:
        from .benchmark import HeroJourneyBenchmark, HeroJourneyTask

        return {"HeroJourneyBenchmark": HeroJourneyBenchmark, "HeroJourneyTask": HeroJourneyTask}[name]
    raise AttributeError(name)
