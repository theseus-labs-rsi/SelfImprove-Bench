"""Minimal single-file Sweepmine: ground-truth construction + interaction core."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import random


@dataclass(frozen=True)
class SweepmineConfig:
    """Construction parameters for one ground-truth board."""

    rows: int = 5
    cols: int = 5
    mine_count: int = 4
    seed: int = 0


def validate_config(config: SweepmineConfig) -> None:
    """Validate the minimal board-generation parameters."""
    if config.rows < 1 or config.cols < 1:
        raise ValueError("rows and cols must be positive.")
    if config.mine_count < 1:
        raise ValueError("mine_count must be positive.")
    if config.mine_count >= config.rows * config.cols:
        raise ValueError("mine_count must be smaller than rows * cols.")


def generate_task(
    *,
    seed: int = 0,
    rows: int = 5,
    cols: int = 5,
    mine_count: int = 4,
) -> list[list[int]]:
    """Generate a reproducible ground-truth board.

    ``-1`` denotes a mine and ``0`` denotes a non-mine cell. Derived numbers
    are computed by ``SweepmineCore`` during initialization.
    """
    config = SweepmineConfig(rows, cols, mine_count, seed)
    validate_config(config)
    rng = random.Random(config.seed)
    cells = [(row, col) for row in range(rows) for col in range(cols)]
    mines = set(rng.sample(cells, mine_count))

    board = [[0 for _ in range(cols)] for _ in range(rows)]
    for row, col in mines:
        board[row][col] = -1
    return board


class SweepmineState(IntEnum):
    """Persistent episode state of the current board."""

    FAIL = 0
    ONGOING = 1
    SUCCESS = 2


class SweepmineEvent(IntEnum):
    """Per-action outcome code returned by click()/mark()/unmark()."""

    FAIL = 0
    NORMAL_ACTION = 10
    ABNORMAL_ACTION = 11
    SUCCESS = 20
    RESTART = 30


class SweepmineCore:
    """Core state and click/inspect operations for one board."""

    def build_task(self, seed: int | None = None):
        """Build a fresh board. Default seed = episode seed; callers may pass a
        derived seed (e.g. restart counter) to change the layout deterministically."""
        board_seed = self.seed if seed is None else seed
        self.groundtruth = generate_task(
            seed=board_seed, rows=self.rows, cols=self.cols, mine_count=self.mine_count
        )
        self._validate_groundtruth()
        self.numbers = self._build_numbers()
        self.visibility = [
            [False for _ in row]
            for row in self.groundtruth
        ]
        self.status = SweepmineState.ONGOING

    def __init__(self, config) -> None:
        self.task_id = config.get("task_id", f"sweepmine-seed-{config['seed']}")
        self.difficulty = config.get("difficulty")
        self.seed = config["seed"]
        self.rows = config["rows"]
        self.cols = config["cols"]
        self.mine_count = config["mine_count"]
        self.max_steps = config["max_steps"]
        self.steps = self.max_steps
        self.restart_count = 0
        self.build_task()

    def click(self, row: int, col: int) -> SweepmineEvent:
        """Reveal a cell or its zero-region and return the action outcome.

        Every attempted action (normal, repeated, out-of-bounds, restart)
        consumes one step, including the winning action.
        """
        if self.status != SweepmineState.ONGOING:
            return SweepmineEvent.FAIL
        if not self._validate_position(row, col):
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.ABNORMAL_ACTION
        if self.visibility[row][col]:
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.NORMAL_ACTION
        if self.numbers[row][col] == "M":
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.ABNORMAL_ACTION
        if self.groundtruth[row][col] == -1:
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            self.visibility[row][col] = True
            # Hit a mine: restart with a freshly derived board seed so the
            # layout CHANGES deterministically (episode seed + restart count);
            # cross-restart memory of old mine positions must not help.
            self.restart_count += 1
            self.build_task(seed=self.seed + self.restart_count)
            return SweepmineEvent.RESTART

        queue = deque([(row, col)])
        while queue:
            current_row, current_col = queue.popleft()
            if self.visibility[current_row][current_col]:
                continue
            if self.groundtruth[current_row][current_col] == -1:
                continue
            if self.numbers[current_row][current_col] == "M":
                continue
            self.visibility[current_row][current_col] = True
            if self._number_at(current_row, current_col) != 0:
                continue
            for neighbor in self._neighbors(current_row, current_col):
                neighbor_row, neighbor_col = neighbor
                if not self.visibility[neighbor_row][neighbor_col] and self.numbers[neighbor_row][neighbor_col] != "M":
                    queue.append(neighbor)

        if self._spend_and_fail():
            return SweepmineEvent.FAIL
        return SweepmineEvent.NORMAL_ACTION

    def mark(self, row: int, col: int) -> SweepmineEvent:
        """Mark one hidden cell as a mine."""
        if self.status != SweepmineState.ONGOING:
            return SweepmineEvent.FAIL
        if not self._validate_position(row, col):
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.ABNORMAL_ACTION
        if self.visibility[row][col]:
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.ABNORMAL_ACTION
        if self.numbers[row][col] == "M":
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.ABNORMAL_ACTION
        self.numbers[row][col] = "M"
        if self._mines_exactly_marked():
            self.steps -= 1
            self.status = SweepmineState.SUCCESS
            return SweepmineEvent.SUCCESS
        if self._spend_and_fail():
            return SweepmineEvent.FAIL
        return SweepmineEvent.NORMAL_ACTION

    def unmark(self, row: int, col: int) -> SweepmineEvent:
        """Remove a previously placed mark; no-op on an unmarked cell."""
        if self.status != SweepmineState.ONGOING:
            return SweepmineEvent.FAIL
        if not self._validate_position(row, col):
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.ABNORMAL_ACTION
        if self.visibility[row][col]:
            if self._spend_and_fail():
                return SweepmineEvent.FAIL
            return SweepmineEvent.ABNORMAL_ACTION
        if self.numbers[row][col] == "M":
            self.numbers[row][col] = self._number_at(row, col)
        if self._mines_exactly_marked():
            self.steps -= 1
            self.status = SweepmineState.SUCCESS
            return SweepmineEvent.SUCCESS
        if self._spend_and_fail():
            return SweepmineEvent.FAIL
        return SweepmineEvent.NORMAL_ACTION

    def inspect(self) -> list[list[int | None]]:
        """Return visible numbers while hiding unrevealed cells."""
        return [
            [
                self.numbers[row][col]
                if self.visibility[row][col] or self.numbers[row][col] == "M"
                else None
                for col in range(len(self.groundtruth[row]))
            ]
            for row in range(len(self.groundtruth))
        ]

    def _validate_groundtruth(self) -> None:
        if not self.groundtruth or not self.groundtruth[0]:
            raise ValueError("groundtruth must be a non-empty table.")
        width = len(self.groundtruth[0])
        if any(len(row) != width for row in self.groundtruth):
            raise ValueError("groundtruth must be rectangular.")
        if any(value not in (-1, 0) for row in self.groundtruth for value in row):
            raise ValueError("groundtruth values must be -1 or 0.")

    def _validate_position(self, row: int, col: int) -> bool:
        if not 0 <= row < len(self.groundtruth):
            return False
        if not 0 <= col < len(self.groundtruth[0]):
            return False
        return True

    def _spend_and_fail(self) -> bool:
        """Consume one step; mark FAIL and return True when budget exhausted.

        Every attempted action spends one step, so the budget also bounds
        abnormal/restart attempts. Callers return FAIL when this is True.
        """
        self.steps = self.steps - 1
        if self.steps <= 0:
            self.status = SweepmineState.FAIL
            return True
        return False

    def _build_numbers(self) -> list[list[int]]:
        rows = len(self.groundtruth)
        cols = len(self.groundtruth[0])
        return [
            [self._number_at(row, col) for col in range(cols)]
            for row in range(rows)
        ]

    def _number_at(self, row: int, col: int) -> int:
        """Read the derived number without relying on a mutable mark."""
        if self.groundtruth[row][col] == -1:
            return -1
        return sum(
            self.groundtruth[nr][nc] == -1
            for nr, nc in self._neighbors(row, col)
        )

    def _neighbors(self, row: int, col: int):
        rows = len(self.groundtruth)
        cols = len(self.groundtruth[0])
        for neighbor_row in range(max(0, row - 1), min(rows, row + 2)):
            for neighbor_col in range(max(0, col - 1), min(cols, col + 2)):
                if (neighbor_row, neighbor_col) != (row, col):
                    yield neighbor_row, neighbor_col

    def _mines_exactly_marked(self) -> bool:
        """True only when marked cells exactly equal the true mine set.

        Precision matters: marking every real mine still fails if any safe
        cell was wrongly marked, and vice versa.
        """
        return all(
            (self.numbers[row][col] == "M") == (self.groundtruth[row][col] == -1)
            for row in range(len(self.groundtruth))
            for col in range(len(self.groundtruth[0]))
        )


__all__ = [
    "SweepmineCore",
    "SweepmineState",
    "SweepmineEvent",
    "SweepmineConfig",
    "generate_task",
    "validate_config",
]
