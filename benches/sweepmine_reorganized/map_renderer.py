"""Render a board with state-dependent local cores checked point by point."""

from bisect import bisect_right
from collections.abc import Sequence
from typing import Any


def _validate_core(core: Any) -> None:
    if type(core) is not list or not core:
        raise ValueError("core must be a non-empty square 2D grid")
    size = len(core)
    if size % 2 == 0 or any(type(row) is not list or len(row) != size for row in core):
        raise ValueError("core must be an odd square 2D grid")
    if any(type(value) is not int or value < 0 for row in core for value in row):
        raise ValueError("core weights must be non-negative integers")


def _validate_layers(layers: Any) -> None:
    if type(layers) is not list or not layers:
        raise ValueError("core must contain at least one layer")
    for index, layer in enumerate(layers):
        if type(layer) is not dict or set(layer) != {"thresholds", "cores"}:
            raise ValueError(f"layer {index} must contain thresholds and cores")
        thresholds = layer["thresholds"]
        cores = layer["cores"]
        if index == 0 and thresholds is not None:
            raise ValueError("the first layer must not have thresholds")
        if thresholds is not None and (
            type(thresholds) is not list
            or any(type(value) is not int for value in thresholds)
            or thresholds != sorted(set(thresholds))
        ):
            raise ValueError("thresholds must be strictly increasing integers")
        if type(cores) is not list or len(cores) != 1 + len(thresholds or []):
            raise ValueError("a layer needs one core per threshold interval")
        for candidate in cores:
            _validate_core(candidate)


def _window_value(board: list[list[int]], row: int, col: int, core: list[list[int]]) -> int:
    radius = len(core) // 2
    value = 0
    for core_row, weights in enumerate(core):
        for core_col, weight in enumerate(weights):
            source_row = row + core_row - radius
            source_col = col + core_col - radius
            if 0 <= source_row < len(board) and 0 <= source_col < len(board[0]):
                value += board[source_row][source_col] * weight
    return value


def _point_value(board: list[list[int]], row: int, col: int, layers: list[dict[str, Any]]) -> int:
    value = _window_value(board, row, col, layers[0]["cores"][0])
    for layer in layers[1:]:
        branch = bisect_right(layer["thresholds"], value)
        value = _window_value(board, row, col, layer["cores"][branch])
    return value


def render_map(
    map: Sequence[Sequence[int]], core: list[dict[str, Any]], zero_core: Any = None
) -> list[list[int | str]]:
    """Check each point against the original board using layered local cores."""
    if zero_core is not None:
        raise ValueError("zero_core is not supported by the layered renderer")
    board = [list(row) for row in map]
    if not board or any(len(row) != len(board[0]) for row in board):
        raise ValueError("map must be a non-empty rectangular binary grid")
    if any(value not in (0, 1) for row in board for value in row):
        raise ValueError("map must be a binary grid")
    _validate_layers(core)
    return [
        ["M" if board[row][col] else _point_value(board, row, col, core)
         for col in range(len(board[0]))]
        for row in range(len(board))
    ]
