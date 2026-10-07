"""Model-facing prompt construction for standard-number Sweepmine."""

from __future__ import annotations

# Standard number rule, with the benchmark's existing restart and win rules.
_INITIAL_PROMPT = "\n".join(
    [
        "General:",
        "You are playing Minesweeper. Each opened safe cell shows the exact "
        "number of mines among its eight surrounding cells; off-board cells "
        "are ignored. Opening a zero reveals its connected zero region and "
        "neighboring numbered cells. Your goal is to mark exactly every mine "
        "and no safe cells on one board.",
        "",
        "Available Acts:",
        "You take one act per turn, addressing a cell by zero-based "
        "(row, col) coordinates:",
        " - click(row, col): open an unmarked cell;",
        " - mark(row, col): mark a hidden cell as a mine;",
        " - unmark(row, col): remove a mark from a cell.",
        "Clicking a mine deals a new board under the same rules. Marked cells "
        "cannot be opened until unmarked. Out-of-range or repeated acts do "
        "not reveal new information.",
        "",
        "Response Format:",
        'Your entire response must be exactly one line of JSON with only the '
        'keys "act", "row", "col". Do not add explanations, code fences, or '
        "any other text. After every act you automatically receive the "
        "current board status.",
        "",
    ]
)


def build_initial_prompt() -> str:
    """Return the initial (static) instruction prompt for an episode."""
    return _INITIAL_PROMPT


def build_turn_prompt(statement: str, trace: list[str], board: str) -> str:
    """Render one turn with history scoped to the current board only."""
    trace_text = (
        "\n".join(f"{index}. {line}" for index, line in enumerate(trace, start=1))
        if trace
        else "No actions have been taken on the current board."
    )
    return "\n\n".join((
        "## Statement\n" + statement.strip(),
        "## Trace\n" + trace_text,
        "## Board\n" + board,
    ))
