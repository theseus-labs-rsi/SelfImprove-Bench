"""Model-free regressions for browser configuration and wrapped episode errors."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchmark import runner
from benchmark.config import AgentConfig, BenchmarkConfig
from benchmark.tasks import PlannedTask


@pytest.fixture
def gameworld_core(monkeypatch):
    core = Path(__file__).resolve().parents[1] / "benches/gameworld/assets/core/code"
    monkeypatch.syspath_prepend(str(core))


def test_software_renderer_needs_no_custom_browser(gameworld_core, monkeypatch):
    from env.browser_manager import BrowserConfig, BrowserGameManager

    launch = AsyncMock()
    browser = launch.return_value
    browser.new_context.return_value.new_page = AsyncMock()
    browser_type = SimpleNamespace(launch=launch)
    playwright = SimpleNamespace(chromium=browser_type)
    monkeypatch.setattr("env.browser_manager.async_playwright", lambda: SimpleNamespace(
        start=AsyncMock(return_value=playwright)))
    monkeypatch.setenv("GAMEWORLD_BROWSER", "chromium")
    monkeypatch.setenv("GAMEWORLD_RENDERER", "swiftshader")
    monkeypatch.delenv("GAMEWORLD_BROWSER_EXECUTABLE", raising=False)
    manager = BrowserGameManager(BrowserConfig(game_url="http://localhost", headless=True))
    asyncio.run(manager._launch_browser())
    assert "--use-angle=swiftshader" in launch.call_args.kwargs["args"]
    assert "executable_path" not in launch.call_args.kwargs


def test_default_renderer_and_invalid_setting(gameworld_core, monkeypatch):
    from env.browser_manager import BrowserGameManager

    monkeypatch.delenv("GAMEWORLD_RENDERER", raising=False)
    assert "--use-angle=swiftshader" not in BrowserGameManager._browser_launch_args()
    monkeypatch.setenv("GAMEWORLD_RENDERER", "typo")
    with pytest.raises(ValueError, match="GAMEWORLD_RENDERER"):
        BrowserGameManager._browser_launch_args()


def test_wrapped_connection_failure_stops_without_refinement(
    gameworld_core, monkeypatch, tmp_path,
):
    import catalog  # Match the production catalog-before-runtime import order.
    from runtime.coordinator import Coordinator

    # Isolated synthetic transport failure: no browser, model or paid calls.
    coordinator = Coordinator.__new__(Coordinator)
    coordinator._stop_event = asyncio.Event()
    coordinator._fatal_error = None
    coordinator.agent_loggers = {}
    coordinator.agents = [SimpleNamespace(agent_id="synthetic-agent")]
    coordinator.env = SimpleNamespace(start=AsyncMock(), close_game=AsyncMock())
    coordinator._run_agent_step = AsyncMock(side_effect=ConnectionError("synthetic offline failure"))
    monkeypatch.setattr(runner, "evaluate_task", lambda *args, **kwargs: asyncio.run(coordinator.run()))
    session = SimpleNamespace(refinement_telemetry=None, pending_batch=None)
    controller = SimpleNamespace(
        config=BenchmarkConfig(seed=1, schedule=(), holdout=(),
                               agent=AgentConfig(provider="rsi", adapter="examples.minimal_rsi:MinimalAdapter"),
                               continue_on_error=True),
        provider="rsi", session=session, active_agent=object(),
        begin=Mock(), request=Mock(), refine=Mock(),
    )
    task = PlannedTask("synthetic-episode", "train", "gameworld", 3,
                       {"task_id": "synthetic-episode"}, 1, 0)
    record = runner._run_one(task, controller=controller, ordinal=1, output_dir=tmp_path)
    assert record["status"] == "error"
    assert record["error"].startswith("Agent loop failed")
    assert record["error_chain"] == [{"error_type": "ConnectionError", "error": "synthetic offline failure"}]
    assert runner._is_provider_failure(record)
    assert runner._must_stop_after_error(controller.config, record)
    assert not runner._checkpoint_is_stable(controller.config, task, record)
    controller.refine.assert_not_called()
    coordinator.env.close_game.assert_awaited_once()


def test_task_error_in_wrapper_remains_a_task_error():
    task = PlannedTask("synthetic-task", "train", "gameworld", 3, {}, 1, 0)
    try:
        try:
            raise ValueError("synthetic invalid action")
        except ValueError as cause:
            raise RuntimeError("Agent loop failed") from cause
    except RuntimeError as exc:
        record = runner._error_result(task, exc)
    assert not runner._is_provider_failure(record)


def test_nexau_wrapped_connection_error_is_provider_failure():
    assert runner._is_provider_failure({
        "status": "error", "error_type": "RuntimeError",
        "error": "Error in agent execution: Connection error.",
    })
