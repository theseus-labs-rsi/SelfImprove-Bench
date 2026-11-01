"""Load the bundled Hero runtime without adding its siblings to sys.path."""

from pathlib import Path
from types import ModuleType

from benchmark.imports import load_vendor_package


def ensure_hero_package() -> ModuleType:
    return load_vendor_package(
        "herosjourney", Path(__file__).with_name("HerosJourney") / "herosjourney",
    )
