(function () {
  const GAME_ID = '03_astray';
  const DEFAULT_SEED = 42;

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
    maxDistanceToGoal: null
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

  function updateDistanceProgress(level, distanceToGoal) {
    if (level !== null && level !== session.progressLevel) {
      session.progressLevel = level;
      session.maxDistanceToGoal = null;
    }

    if (distanceToGoal !== null && session.maxDistanceToGoal === null) {
      session.maxDistanceToGoal = Math.max(distanceToGoal, 0);
    }

    if (distanceToGoal === null || session.maxDistanceToGoal === null) {
      return null;
    }

    if (session.maxDistanceToGoal <= 1e-9) {
      return distanceToGoal <= 1e-6 ? 1 : 0;
    }

    const progress = (session.maxDistanceToGoal - distanceToGoal) / session.maxDistanceToGoal;
    return clamp01(progress);
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
      terminal: {
        isTerminal: false,
        outcome: null,
        reason: null
      },
      game_state: {
        score: null,
        level: null,
        player: null,
        environment: null,
        completion_progress: null
      },
      metrics: {
        primary_score: null,
        distance_to_goal: null
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
      const distanceToGoal = computeDistanceToGoal(player, exit);
      const completionProgress = updateDistanceProgress(level, distanceToGoal);
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
        terminal: {
          isTerminal: false,
          outcome: null,
          reason: null
        },
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
          distance_to_goal: distanceToGoal
        },
        debug: {
          state: state,
          maze_dimension: mazeDimension,
          max_distance_to_goal: safeNumber(session.maxDistanceToGoal),
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
