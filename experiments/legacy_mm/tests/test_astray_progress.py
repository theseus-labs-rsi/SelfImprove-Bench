"""Astray progress follows maze corridors, not straight-line distance to the exit."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GAME_APIS = [ROOT / "gameworld_train_astray/code/games/benchmark/03_astray/game_api.js",
             ROOT / "gameworld/code/games/benchmark/03_astray/game_api.js"]

# 11x11 maze (level 1; true = wall). Route (1,1)->(1,9)->(10,9)->exit (11,9),
# plus a dead end (2,5)->(4,5) that points toward the exit but leads nowhere.
DRIVER = r"""
const fs = require('fs');
const walls = Array.from({length: 11}, () => Array(11).fill(true));
for (let y = 1; y <= 9; y++) walls[1][y] = false;
for (let x = 2; x <= 10; x++) walls[x][9] = false;
for (let x = 2; x <= 4; x++) walls[x][5] = false;
walls.dimension = 11;
global.window = {location: {search: '?level=1'}, maze: walls, mazeDimension: 11, gameState: 'play',
                 ballMesh: {position: {x: 1, y: 1, z: 0}}, wBall: null, ballRadius: 0.25, keyAxis: [0, 0]};
eval(fs.readFileSync(process.argv[2], 'utf8'));
window.gameAPI.init({});
const out = [];
function at(x, y) { window.ballMesh.position.x = x; window.ballMesh.position.y = y; return window.gameAPI.getState(); }
for (const [x, y] of [[1,1],[1,5],[3,5],[5,9]]) {
  const s = at(x, y);
  out.push({cell: [x, y], progress: s.game_state.completion_progress, path: s.metrics.distance_to_goal,
            euclid: s.metrics.euclidean_distance_to_goal, terminal: s.terminal.isTerminal});
}
window.mazeDimension = 13;  // the page's exit check: level up before rebuilding the maze
const after = window.gameAPI.getState();
window.gameState = 'fade out';
const loading = window.gameAPI.getState();
console.log(JSON.stringify({steps: out, after: [after.game_state.completion_progress, after.terminal],
                            loading: [loading.game_state.completion_progress, loading.terminal.outcome]}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("game_api", GAME_APIS, ids=["train", "holdout"])
def test_corridor_progress(game_api, tmp_path):
    driver = tmp_path / "driver.js"
    driver.write_text(DRIVER)
    result = json.loads(subprocess.run(["node", str(driver), str(game_api)], check=True,
                                       capture_output=True, text=True).stdout)
    steps = {tuple(s["cell"]): s for s in result["steps"]}
    # Route length from (1,1) to the exit (11,9) is 18 cells.
    assert steps[(1, 1)]["progress"] == 0 and steps[(1, 1)]["path"] == 18
    assert steps[(1, 5)]["progress"] == pytest.approx(4 / 18)
    # The dead end is closer in a straight line than (1,5) but farther by corridor.
    assert steps[(3, 5)]["euclid"] < steps[(1, 5)]["euclid"]
    assert steps[(3, 5)]["path"] == 16 and steps[(3, 5)]["progress"] < steps[(1, 5)]["progress"]
    assert steps[(5, 9)]["progress"] == pytest.approx(12 / 18)
    assert not any(s["terminal"] for s in result["steps"])
    assert result["after"] == [1, {"isTerminal": True, "outcome": "success", "reason": "exit_reached"}]
    assert result["loading"] == [1, "success"]
