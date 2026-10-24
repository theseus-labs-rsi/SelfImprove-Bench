// Exercise the shipped browser generator and game API without a model or renderer.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../benches/gameworld/assets/core/code/games/benchmark/03_astray');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const bootstrap = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const generator = fs.readFileSync(path.join(root, 'maze.js'), 'utf8');
const api = fs.readFileSync(path.join(root, 'game_api.js'), 'utf8');

function load(level, seed) {
    const dimension = 2 * level + 9;
    const window = {
        location: {search: `?level=${level}&train_seed=${seed}`},
        gameState: 'play',
        mazeDimension: dimension,
        ballMesh: {position: {x: 1, y: 1, z: 0}},
    };
    const context = vm.createContext({window, URLSearchParams});
    vm.runInContext(bootstrap, context);
    vm.runInContext(generator, context);
    window.maze = vm.runInContext(`generateSquareMaze(${dimension})`, context);
    // Match the page's initialization, which opens the boundary toward the exit.
    window.maze[dimension - 1][dimension - 2] = false;
    vm.runInContext(api, context);
    return window;
}

function routeToExit(window) {
    const n = window.mazeDimension;
    const queue = [[1, 1]];
    const previous = new Map([['1,1', null]]);
    for (let head = 0; head < queue.length; head++) {
        const [x, y] = queue[head];
        if (x === n && y === n - 2) {
            const route = [];
            for (let cell = [x, y]; cell; cell = previous.get(cell.join(','))) route.push(cell);
            return route.reverse();
        }
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const nx = x + dx;
            const ny = y + dy;
            const exit = nx === n && ny === n - 2;
            const open = nx >= 0 && ny >= 0 && nx < n && ny < n && window.maze[nx][ny] === false;
            const key = `${nx},${ny}`;
            if (!(exit || open) || previous.has(key)) continue;
            previous.set(key, [x, y]);
            queue.push([nx, ny]);
        }
    }
    throw new Error('Exit is unreachable');
}

for (let difficulty = 1; difficulty <= 5; difficulty++) {
    for (let index = 0; index < 20; index++) {
        const seed = 20260923 + (difficulty - 1) * 20 + index;
        const window = load(difficulty, seed);
        assert.equal(JSON.stringify(window.maze), JSON.stringify(load(difficulty, seed).maze));
        const route = routeToExit(window);
        const initial = window.gameAPI.getState();
        assert.equal(initial.seed, seed);
        assert.equal(initial.metrics.distance_to_goal, route.length - 1);
        assert.equal(initial.game_state.completion_progress, 0);
        const midpoint = Math.floor((route.length - 1) / 2);
        [window.ballMesh.position.x, window.ballMesh.position.y] = route[midpoint];
        const partial = window.gameAPI.getState();
        assert.ok(Math.abs(partial.game_state.completion_progress - midpoint / (route.length - 1)) < 1e-12);
        // The browser reports current progress; Python's evaluator retains the run maximum.
        [window.ballMesh.position.x, window.ballMesh.position.y] = route[0];
        assert.equal(window.gameAPI.getState().game_state.completion_progress, 0);
        window.mazeDimension += 2; // Match the page's immediate transition after the exit.
        const completed = window.gameAPI.getState();
        assert.equal(completed.game_state.completion_progress, 1);
        assert.equal(completed.terminal.outcome, 'success');
        window.gameState = 'fade out';
        assert.equal(window.gameAPI.getState().game_state.completion_progress, 1);
    }
}
const zero = load(3, 0);
assert.equal(zero.gameAPI.getState().seed, 0);
assert.notEqual(JSON.stringify(zero.maze), JSON.stringify(load(3, 42).maze));
assert.equal(JSON.stringify(zero.maze), JSON.stringify(load(3, 0).maze));
assert.equal(load(3, 4294967295).gameAPI.getState().seed, 4294967295);
console.log('Astray: 100 reproducible, reachable maps; BFS progress, completion, and uint32 seeds OK');
