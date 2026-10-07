(function () {
  const GAME_ID = '03_astray';
  // Report the query-selected training map seed; gameAPI.init({seed}) stays unsupported.
  const rawTrainSeed = new URLSearchParams(window.location.search || '').get('train_seed');
  const parsedTrainSeed = rawTrainSeed !== null && /^(0|[1-9][0-9]*)$/.test(rawTrainSeed) ? Number(rawTrainSeed) : NaN;
  const DEFAULT_SEED = Number.isSafeInteger(parsedTrainSeed) && parsedTrainSeed > 0 && parsedTrainSeed <= 0xffffffff
    ? (parsedTrainSeed >>> 0) : 42;

  const capabilities = {
    supports_seed: false,
    supports_level_select: false,
    supports_difficulty: false,
    supports_inplace_reset: false,
    supports_reload_reset: true,
    supports_pause_detection: false,
    supports_menu_detection: false,
    provides_actionable_flag: true
  };

  const session = {
    seed: DEFAULT_SEED,
    requestedLevel: null,
    requestedDifficulty: null,
    episodeStartMs: Date.now(),
    episodeCount: 0,
    progressLevel: null,
    maxDistanceToGoal: null,
    startLevel: null,
    startPathDistance: null,
    completed: false
  };

  const runtimeState = {
    lastResetMethod: null
  };

  function safeNumber(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function clamp01(value) {
    if (!Number.isFinite(value)) return null;
    if (value <= 0) return 0;
    if (value >= 1) return 1;
    return value;
  }

  function intOrNull(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    return Math.trunc(value);
  }

  function normalizeOptions(options) {
    const opts = options || {};
    return {
      seed: intOrNull(opts.seed),
      level:
        opts.level === undefined || opts.level === null
          ? null
          : (typeof opts.level === 'number' && Number.isFinite(opts.level))
            ? Math.trunc(opts.level)
            : String(opts.level),
      difficulty:
        opts.difficulty === undefined || opts.difficulty === null
          ? null
          : String(opts.difficulty)
    };
  }

  function resetDeterministicRandom() {
    if (typeof window !== 'undefined' && typeof window.__resetRandom === 'function') {
      try {
        window.__resetRandom();
      } catch (error) {
        // Best effort only; default seed is fixed by the page bootstrap.
      }
    }
  }

  function beginEpisode(options) {
    const accepted = normalizeOptions(options);
    const notes = [];

    if (accepted.seed !== null) notes.push('seed_not_supported');
    if (accepted.level !== null) notes.push('level_not_supported');
    if (accepted.difficulty !== null) notes.push('difficulty_not_supported');

    resetDeterministicRandom();

    session.seed = DEFAULT_SEED;
    session.requestedLevel = accepted.level;
    session.requestedDifficulty = accepted.difficulty;
    session.episodeStartMs = Date.now();
    session.episodeCount += 1;
    session.progressLevel = null;
    session.maxDistanceToGoal = null;
    session.startLevel = null;
    session.startPathDistance = null;
    session.completed = false;

    return {
      accepted: accepted,
      applied: {
        seed: DEFAULT_SEED,
        level: null,
        difficulty: null
      },
      notes: notes
    };
  }

  function getLevelFromDimension(dimension) {
    if (typeof dimension !== 'number') return null;
    return Math.floor((dimension - 1) / 2 - 4);
  }

  function getGameState() {
    return typeof window !== 'undefined' ? window.gameState : null;
  }

  function mapStatus(state) {
    if (!state) return 'loading';
    if (state === 'play') return 'playing';
    if (state === 'fade in' || state === 'fade out' || state === 'initialize') return 'loading';
    return 'loading';
  }

  function getPlayerSnapshot() {
    const ball = typeof window !== 'undefined' ? window.ballMesh : null;
    if (!ball || !ball.position) return null;

    const body = typeof window !== 'undefined' ? window.wBall : null;
    const velocity =
      body && typeof body.GetLinearVelocity === 'function'
        ? body.GetLinearVelocity()
        : null;

    return {
      x: safeNumber(ball.position.x),
      y: safeNumber(ball.position.y),
      z: safeNumber(ball.position.z),
      vx: safeNumber(velocity && velocity.x),
      vy: safeNumber(velocity && velocity.y),
      state: getGameState() === 'play' ? 'rolling' : null,
      props: {
        radius: safeNumber(typeof window !== 'undefined' ? window.ballRadius : null)
      }
    };
  }

  function getExitSnapshot(mazeDimension) {
    if (mazeDimension === null) return null;
    return {
      x: mazeDimension,
      y: mazeDimension - 2,
      z: 0
    };
  }

  function computeDistanceToGoal(player, exit) {
    if (!player || !exit) return null;
    const dx = exit.x - player.x;
    const dy = exit.y - player.y;
    return safeNumber(Math.sqrt((dx * dx) + (dy * dy)));
  }

  // Shortest open-corridor distance (grid steps) from every cell to the exit.
  // Straight-line distance rewards dead ends that happen to lie near the exit
  // corner; corridor distance only decreases on the way to the exit. The maze
  // is at most a few hundred cells, so one BFS per maze costs well under 1 ms.
  const pathCache = { maze: null, dimension: null, dist: null };

  function corridorDistances(maze, dimension) {
    if (pathCache.maze === maze && pathCache.dimension === dimension) return pathCache.dist;
    const size = dimension + 1; // the exit cell (dimension, dimension - 2) is outside the grid
    const exitX = dimension;
    const exitY = dimension - 2;
    const isOpen = (x, y) => (x === exitX && y === exitY) ||
      (x >= 0 && y >= 0 && x < dimension && y < dimension && maze[x] && maze[x][y] === false);
    const dist = new Array(size * size).fill(-1);
    const queue = [[exitX, exitY]];
    dist[exitX * size + exitY] = 0;
    for (let head = 0; head < queue.length; head++) {
      const [x, y] = queue[head];
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
        if (dist[nx * size + ny] >= 0 || !isOpen(nx, ny)) continue;
        dist[nx * size + ny] = dist[x * size + y] + 1;
        queue.push([nx, ny]);
      }
    }
    pathCache.maze = maze;
    pathCache.dimension = dimension;
    pathCache.dist = dist;
    return dist;
  }

  function computePathDistance(player, mazeDimension) {
    const maze = typeof window !== 'undefined' ? window.maze : null;
    // After the exit the page bumps mazeDimension before it rebuilds the maze.
    if (!player || !maze || mazeDimension === null || maze.dimension !== mazeDimension) return null;
    if (player.x === null || player.y === null) return null;
    // Same cell rounding as the page's own exit check.
    const x = Math.floor(player.x + 0.5);
    const y = Math.floor(player.y + 0.5);
    const size = mazeDimension + 1;
    if (x < 0 || y < 0 || x >= size || y >= size) return null;
    const value = corridorDistances(maze, mazeDimension)[x * size + y];
    return value >= 0 ? value : null;
  }

  // Level the page loads, by the page's own rule (dimension = 2 * level + 9,
  // default 11), so an exit reached before the first observation still counts.
  function initialLevel() {
    let dimension = 11;
    const raw = new URLSearchParams(window.location.search || '').get('level');
    const value = raw ? parseInt(raw, 10) : NaN;
    if (isFinite(value) && value >= 1) dimension = 2 * value + 9;
    return getLevelFromDimension(dimension);
  }

  function updatePathProgress(level, pathDistance) {
    if (session.startLevel === null) session.startLevel = initialLevel();
    // The page advances the level as soon as the ball reaches the exit.
    if (session.startLevel !== null && level !== null && level > session.startLevel) {
      session.completed = true;
    }
    if (session.completed) return 1;
    if (level === null || level !== session.startLevel || pathDistance === null) return null;
    if (session.startPathDistance === null) session.startPathDistance = pathDistance;
    if (session.startPathDistance <= 0) return 1;
    return clamp01((session.startPathDistance - pathDistance) / session.startPathDistance);
  }

  function terminalSnapshot() {
    return session.completed
      ? { isTerminal: true, outcome: 'success', reason: 'exit_reached' }
      : { isTerminal: false, outcome: null, reason: null };
  }

  function buildLoadingState(now) {
    return {
      schemaVersion: '2.0',
      gameId: GAME_ID,
      seed: session.seed,
      timestampMs: now,
      gameTimeMs: safeNumber(now - session.episodeStartMs),
      status: 'loading',
      is_actionable: false,
      // The page fades out and builds the next level right after the exit;
      // a finished episode stays reported as finished during that transition.
      terminal: terminalSnapshot(),
      game_state: {
        score: null,
        level: null,
        player: null,
        environment: null,
        completion_progress: session.completed ? 1 : null
      },
      metrics: {
        primary_score: session.completed ? 1 : null,
        distance_to_goal: session.completed ? 0 : null,
        euclidean_distance_to_goal: null
      },
      debug: {
        state: getGameState(),
        maze_dimension: safeNumber(typeof window !== 'undefined' ? window.mazeDimension : null),
        last_reset_method: runtimeState.lastResetMethod,
        episode_count: session.episodeCount
      }
    };
  }

  window.gameAPI = {
    version: '2.0',
    capabilities: capabilities,

    init: async function init(config) {
      const episode = beginEpisode(config || null);
      return {
        ok: true,
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes.length ? episode.notes : []
      };
    },

    getState: function getState() {
      const now = Date.now();
      const state = getGameState();
      const status = mapStatus(state);
      const mazeDimension = safeNumber(typeof window !== 'undefined' ? window.mazeDimension : null);
      const level = getLevelFromDimension(mazeDimension);
      const player = getPlayerSnapshot();
      const exit = getExitSnapshot(mazeDimension);
      const euclideanDistance = computeDistanceToGoal(player, exit);
      const pathDistance = computePathDistance(player, mazeDimension);
      const completionProgress = updatePathProgress(level, pathDistance);
      const keyAxis =
        typeof window !== 'undefined' && Array.isArray(window.keyAxis)
          ? window.keyAxis.map(safeNumber)
          : null;

      if (status === 'loading') {
        return buildLoadingState(now);
      }

      return {
        schemaVersion: '2.0',
        gameId: GAME_ID,
        seed: session.seed,
        timestampMs: now,
        gameTimeMs: safeNumber(now - session.episodeStartMs),
        status: status,
        is_actionable: status === 'playing' && player !== null,
        terminal: terminalSnapshot(),
        game_state: {
          score: null,
          level: level,
          player: player,
          environment: {
            maze_dimension: mazeDimension,
            exit: exit
          },
          completion_progress: safeNumber(completionProgress)
        },
        metrics: {
          primary_score: safeNumber(completionProgress),
          // Corridor cells to the exit; the original straight-line value is kept for reference.
          distance_to_goal: session.completed ? 0 : pathDistance,
          euclidean_distance_to_goal: euclideanDistance
        },
        debug: {
          state: state,
          maze_dimension: mazeDimension,
          start_path_distance: session.startPathDistance,
          progress_metric: 'corridor_shortest_path',
          key_axis: keyAxis,
          requested_level: session.requestedLevel,
          requested_difficulty: session.requestedDifficulty,
          last_reset_method: runtimeState.lastResetMethod,
          episode_count: session.episodeCount
        }
      };
    },

    reset: async function reset(options) {
      const episode = beginEpisode(options || null);
      if (typeof window !== 'undefined' && window.location && typeof window.location.reload === 'function') {
        runtimeState.lastResetMethod = 'reload';
        window.location.reload();
        return {
          ok: true,
          method: 'reload',
          accepted: episode.accepted,
          applied: episode.applied,
          notes: episode.notes.length ? episode.notes : []
        };
      }

      runtimeState.lastResetMethod = 'unsupported';
      return {
        ok: false,
        method: 'unsupported',
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes.concat(['reload_not_available'])
      };
    }
  };
})();
