"""Score non-mine cells without grading reasoning text."""

from collections.abc import Sequence
from typing import Any


def score_details(
    predicted: Sequence[dict[str, Any] | None],
    answer_maps: Sequence[Sequence[Sequence[int | str]]],
) -> list[dict[str, Any]]:
    """Return cell-level accuracy details for each query board."""
    if not answer_maps or len(predicted) != len(answer_maps):
        raise ValueError("predictions and answer maps must have the same nonzero count")
    details = []
    for answer, expected in zip(predicted, answer_maps):
        board = answer["board"] if answer is not None else None
        cells = [
            (row, col, value, expected[row][col])
            for row, values in enumerate(expected)
            for col, value in enumerate(values)
            if value != "M"
        ]
        near_cells = [
            cell for cell in cells
            if any(
                0 <= cell[0] + row_delta < len(expected)
                and 0 <= cell[1] + col_delta < len(expected[0])
                and expected[cell[0] + row_delta][cell[1] + col_delta] == "M"
                for row_delta in (-1, 0, 1)
                for col_delta in (-1, 0, 1)
                if row_delta or col_delta
            )
        ]
        def correct(items):
            return sum(
                board is not None and board[row][col] == value
                for row, col, value, _ in items
            )
        details.append({
            "correct_cell_count": correct(cells),
            "scored_cell_count": len(cells),
            "near_mine_correct_cell_count": correct(near_cells),
            "near_mine_scored_cell_count": len(near_cells),
            "exact_board_correct": board is not None and board == expected,
        })
    return details


def score_answers(
    predicted: Sequence[dict[str, Any] | None],
    answer_maps: Sequence[Sequence[Sequence[int | str]]],
) -> float:
    """Return percentage accuracy over non-mine cells."""
    details = score_details(predicted, answer_maps)
    correct = sum(item["correct_cell_count"] for item in details)
    scored = sum(item["scored_cell_count"] for item in details)
    return 100.0 * correct / scored if scored else 0.0
