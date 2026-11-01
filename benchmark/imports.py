"""Explicit import bridges for bundled upstream packages and worker tools."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType

_PACKAGE_LOCK = RLock()

def load_vendor_package(name: str, package_root: Path) -> ModuleType:
    if not name.isidentifier():
        raise ValueError("A bundled package must have a top-level package name")
    package_root = Path(package_root).resolve()
    initializer = package_root / "__init__.py"
    if not initializer.is_file():
        raise ImportError(f"Bundled package initializer not found: {initializer}")
    with _PACKAGE_LOCK:
        existing = sys.modules.get(name)
        if existing is not None:
            origin = getattr(existing, "__file__", None)
            if origin is None or Path(origin).resolve() != initializer:
                raise ImportError(f"Package {name!r} is already loaded from {origin!r}")
            return existing
        spec = importlib.util.spec_from_file_location(name, initializer, submodule_search_locations=[str(package_root)])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load bundled package {name!r} from {initializer}")
        before = set(sys.modules)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for loaded in set(sys.modules) - before:
                if loaded == name or loaded.startswith(name + "."):
                    del sys.modules[loaded]
            raise
        return module

def prepend_legacy_paths(*source_roots: Path) -> None:
    roots = list(dict.fromkeys(str(Path(root).resolve()) for root in source_roots))
    for root in roots:
        if not Path(root).is_dir():
            raise FileNotFoundError(f"Import source directory not found: {root}")
    sys.path[:] = roots + [entry for entry in sys.path if entry not in roots]
