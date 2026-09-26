"""Build a layered, state-dependent collection of local convolution cores."""

import random


def core_size(difficulty: int) -> int:
    """Return the composed core size for a pair of difficulty levels."""
    if type(difficulty) is not int or difficulty < 1:
        raise ValueError("difficulty must be a positive integer")
    return 2 * ((difficulty + 1) // 2) + 1


def _convolve(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    """Sum the weights of all paths that reach each output offset."""
    size = len(left) + len(right) - 1
    result = [[0 for _ in range(size)] for _ in range(size)]
    for left_row, values in enumerate(left):
        for left_col, left_weight in enumerate(values):
            if left_weight == 0:
                continue
            for right_row, right_values in enumerate(right):
                for right_col, right_weight in enumerate(right_values):
                    result[left_row + right_row][left_col + right_col] += (
                        left_weight * right_weight
                    )
    return result


def _weighted_core(active_count: int, seed: int) -> list[list[int]]:
    """Sample non-center positions and assign deterministic integer weights."""
    if type(active_count) is not int or not 1 <= active_count <= 8:
        raise ValueError("active_count must be between 1 and 8")
    positions = [
        (row, col)
        for row in range(3)
        for col in range(3)
        if (row, col) != (1, 1)
    ]
    rng = random.Random(seed)
    selected = rng.sample(positions, active_count)
    core = [[0 for _ in range(3)] for _ in range(3)]
    for row, col in selected:
        core[row][col] = rng.randint(1, 2)
    return core


def generate_core(
    difficulty: int, active_count: int, seed: int
) -> list[dict[str, object]]:
    """Generate one fixed first layer and threshold-selected later layers."""
    layers = (core_size(difficulty) - 1) // 2
    result: list[dict[str, object]] = []
    result.append({"thresholds": None, "cores": [_weighted_core(active_count, seed)]})

    for layer in range(1, layers):
        thresholds = [2]
        cores = [
            _weighted_core(active_count, seed + layer * 100 + branch)
            for branch in range(2)
        ]
        result.append({"thresholds": thresholds, "cores": cores})
    return result
