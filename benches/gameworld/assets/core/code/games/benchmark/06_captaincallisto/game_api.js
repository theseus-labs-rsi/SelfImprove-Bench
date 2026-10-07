(function () {
  "use strict";

  const GAME_ID = "06_captaincallisto";
  const DEFAULT_SEED = 42;
  const MAX_LEVEL = 10;

  const capabilities = {
    supports_seed: true,
    supports_level_select: true,
    supports_difficulty: false,
    supports_inplace_reset: true,
    supports_reload_reset: true,
    supports_pause_detection: false,
    supports_menu_detection: true,
    provides_actionable_flag: true
  };

  const session = {
    seed: DEFAULT_SEED,
    requestedLevel: null,
    requestedDifficulty: null,
    episodeStartMs: Date.now(),
    episodeCount: 0
  };

  const stats = {
    deathCount: 0,
    wasAlive: null
  };

  const runtimeState = {
    lastStatus: null,
    lastRuntimeState: null,
    lastMenuState: null,
    lastLevel: null,
    lastResetMethod: null,
    terminalLatch: {
      isTerminal: false,
      outcome: null,
      reason: null,
      ts: 0
    },
    terminalDisplay: {
      score: null,
      level: null,
      completionProgress: null,
      lives: null
    }
  };

  function finiteOrNull(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function intOrNull(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return Math.trunc(value);
  }

  function clamp01(value) {
    if (!Number.isFinite(value)) return null;
    if (value <= 0) return 0;
    if (value >= 1) return 1;
    return value;
  }

  function normalizeSeed(seed) {
    const numeric = Number(seed);
    if (!Number.isFinite(numeric)) return DEFAULT_SEED;
    return numeric >>> 0;
  }

  function normalizeLevel(level) {
    const numeric = intOrNull(level);
    if (numeric === null || numeric < 1) return null;
    return Math.max(1, Math.min(MAX_LEVEL, numeric));
  }

  function normalizeOptions(options) {
    const opts = options || {};
    return {
      seed: intOrNull(opts.seed),
      level:
        opts.level === undefined || opts.level === null
          ? null
          : normalizeLevel(opts.level),
      difficulty:
        opts.difficulty === undefined || opts.difficulty === null
          ? null
          : String(opts.difficulty)
    };
  }

  function getDebugApi() {
    return typeof window !== "undefined" ? window.__cc_debug__ || null : null;
  }

  function getCurrentSeed() {
    if (typeof window !== "undefined" && typeof window.__getDeterministicSeed === "function") {
      try {
        return normalizeSeed(window.__getDeterministicSeed());
      } catch (error) {
        return session.seed;
      }
    }
    return session.seed;
  }

  function applySeed(seed) {
    const normalized = normalizeSeed(seed);
    if (typeof window !== "undefined" && typeof window.__setDeterministicSeed === "function") {
      try {
        return normalizeSeed(window.__setDeterministicSeed(normalized));
      } catch (error) {
        return normalized;
      }
    }
    return normalized;
  }

  function getPlayer() {
    return typeof window !== "undefined" ? window.T || null : null;
  }

  function getWorldObjects() {
    if (typeof window === "undefined" || !Array.isArray(window.V)) return null;
    return window.V;
  }

  function getExitClass() {
    return typeof window !== "undefined" ? window.$b || null : null;
  }

  function getCoinsCollected() {
    return typeof window !== "undefined" ? intOrNull(window.Xb) : null;
  }

  function getCoinsTotal() {
    return typeof window !== "undefined" ? intOrNull(window.ac) : null;
  }

  function getLevel() {
    return typeof window !== "undefined" ? normalizeLevel(window.ec) : null;
  }

  function getRuntimePhase() {
    return typeof window !== "undefined" ? intOrNull(window.S) : null;
  }

  function getMenuState() {
    return typeof window !== "undefined" ? intOrNull(window.ic) : null;
  }

  function getQueryTargetLevel() {
    return typeof window !== "undefined"
      ? normalizeLevel(window.__captaincallisto_query_level_target)
      : null;
  }

  function getPlayerLives(player) {
    if (!player) return null;
    return intOrNull(player.l);
  }

  function getPlayerPosition(player) {
    if (!player || !player.g || player.g.length < 3) return null;
    const x = finiteOrNull(player.g[0]);
    const y = finiteOrNull(player.g[1]);
    const z = finiteOrNull(player.g[2]);
    if (x === null || y === null || z === null) return null;
    return { x: x, y: y, z: z };
  }

  function getPlayerSnapshot(player, playerLives) {
    const position = getPlayerPosition(player);
    if (!position) return null;
    return {
      x: position.x,
      y: position.y,
      z: position.z,
      state: playerLives !== null && playerLives <= 0 ? "dead" : "alive",
      alive: playerLives === null ? null : playerLives > 0
    };
  }

  function updateDeathCount(player) {
    const lives = getPlayerLives(player);
    if (lives === null) return;
    const isAlive = lives > 0;
    if (stats.wasAlive === null) {
      stats.wasAlive = isAlive;
      return;
    }
    if (stats.wasAlive && !isAlive) {
      stats.deathCount += 1;
    }
    stats.wasAlive = isAlive;
  }

  function getExitSummaries(worldObjects, exitClass) {
    if (!worldObjects || !exitClass) return [];
    const summaries = [];
    for (let index = 0; index < worldObjects.length; index += 1) {
      const obj = worldObjects[index];
      if (!obj || !(obj instanceof exitClass) || !obj.g || obj.g.length < 3) continue;
      const x = finiteOrNull(obj.g[0]);
      const y = finiteOrNull(obj.g[1]);
      const z = finiteOrNull(obj.g[2]);
      if (x === null || y === null || z === null) continue;
      summaries.push({ x: x, y: y, z: z });
    }
    return summaries;
  }

  function getNearestExit(playerPosition, exits) {
    if (!playerPosition || !Array.isArray(exits) || exits.length === 0) return null;
    let nearest = null;
    for (let index = 0; index < exits.length; index += 1) {
      const exit = exits[index];
      const distance = Math.hypot(
        playerPosition.x - exit.x,
        playerPosition.y - exit.y,
        playerPosition.z - exit.z
      );
      if (!Number.isFinite(distance)) continue;
      if (!nearest || distance < nearest.distance) {
        nearest = {
          x: exit.x,
          y: exit.y,
          z: exit.z,
          distance: distance
        };
      }
    }
    return nearest;
  }

  function getEnvironmentSnapshot(exits, nearestExit) {
    return {
      exit_count: exits.length,
      nearest_exit: nearestExit
        ? {
            x: nearestExit.x,
            y: nearestExit.y,
            z: nearestExit.z,
            distance: finiteOrNull(nearestExit.distance)
          }
        : null
    };
  }

  function computeCompletionProgress(coinsCollected, coinsTotal) {
    if (coinsCollected === null || coinsTotal === null || coinsTotal <= 0) {
      return null;
    }
    return clamp01(coinsCollected / coinsTotal);
  }

  function latchTerminal(outcome, reason, display) {
    runtimeState.terminalLatch.isTerminal = true;
    runtimeState.terminalLatch.outcome = outcome;
    runtimeState.terminalLatch.reason = reason;
    runtimeState.terminalLatch.ts = Date.now();
    runtimeState.terminalDisplay.score = display && display.score !== undefined ? display.score : null;
    runtimeState.terminalDisplay.level = display && display.level !== undefined ? display.level : null;
    runtimeState.terminalDisplay.completionProgress =
      display && display.completionProgress !== undefined ? display.completionProgress : null;
    runtimeState.terminalDisplay.lives = display && display.lives !== undefined ? display.lives : null;
  }

  function getLatchedTerminal(now) {
    if (!runtimeState.terminalLatch.isTerminal) return null;
    if (now - runtimeState.terminalLatch.ts > 2500) {
      runtimeState.terminalLatch.isTerminal = false;
      runtimeState.terminalLatch.outcome = null;
      runtimeState.terminalLatch.reason = null;
      runtimeState.terminalLatch.ts = 0;
      runtimeState.terminalDisplay.score = null;
      runtimeState.terminalDisplay.level = null;
      runtimeState.terminalDisplay.completionProgress = null;
      runtimeState.terminalDisplay.lives = null;
      return null;
    }
    return {
      isTerminal: true,
      outcome: runtimeState.terminalLatch.outcome,
      reason: runtimeState.terminalLatch.reason
    };
  }

  function getStatus(player, worldObjects, runtimePhase, terminal) {
    if (terminal && terminal.isTerminal) return "terminal";
    if (!player || !worldObjects) return "loading";
    if (runtimePhase === 4) return "playing";
    if (runtimePhase === 1 || runtimePhase === 2 || runtimePhase === 3) return "ready";
    if (runtimePhase === 0) return "menu";
    return "playing";
  }

  function resetEphemeralState() {
    stats.deathCount = 0;
    stats.wasAlive = null;
    runtimeState.lastStatus = null;
    runtimeState.lastRuntimeState = null;
    runtimeState.lastMenuState = null;
    runtimeState.lastLevel = null;
    runtimeState.terminalLatch.isTerminal = false;
    runtimeState.terminalLatch.outcome = null;
    runtimeState.terminalLatch.reason = null;
    runtimeState.terminalLatch.ts = 0;
    runtimeState.terminalDisplay.score = null;
    runtimeState.terminalDisplay.level = null;
    runtimeState.terminalDisplay.completionProgress = null;
    runtimeState.terminalDisplay.lives = null;
  }

  function beginEpisode(options, appliedSeed, appliedLevel) {
    const accepted = normalizeOptions(options);
    const notes = [];
    const actualSeed = appliedSeed === null ? DEFAULT_SEED : normalizeSeed(appliedSeed);

    if (accepted.difficulty !== null) notes.push("difficulty_not_supported");

    session.seed = actualSeed;
    session.requestedLevel = accepted.level;
    session.requestedDifficulty = accepted.difficulty;
    session.episodeStartMs = Date.now();
    session.episodeCount += 1;
    resetEphemeralState();

    return {
      accepted: accepted,
      applied: {
        seed: actualSeed,
        level: appliedLevel,
        difficulty: null
      },
      notes: notes
    };
  }

  function getFallbackLevel() {
    return getLevel() || getQueryTargetLevel() || 1;
  }

  function prepareEpisode(options) {
    const accepted = normalizeOptions(options);
    const appliedSeed = applySeed(accepted.seed === null ? getCurrentSeed() : accepted.seed);
    const appliedLevel = accepted.level !== null ? accepted.level : getFallbackLevel();
    return {
      episode: beginEpisode(accepted, appliedSeed, appliedLevel),
      appliedLevel: appliedLevel
    };
  }

  function runInplaceReset(level) {
    const debugApi = getDebugApi();
    if (!debugApi || typeof debugApi.startLevel !== "function") return false;

    const result = debugApi.startLevel(level);
    if (!result || result.ok !== true) {
      throw new Error("startLevel_failed");
    }
    return true;
  }

  function buildUnavailableState(now) {
    return {
      schemaVersion: "2.0",
      gameId: GAME_ID,
      seed: session.seed,
      timestampMs: now,
      gameTimeMs: finiteOrNull(now - session.episodeStartMs),
      status: "loading",
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
        coins: null,
        coins_total: null,
        lives: null,
        deaths: finiteOrNull(stats.deathCount),
        distance_to_goal: null
      },
      debug: {
        ready: false,
        last_reset_method: runtimeState.lastResetMethod,
        episode_count: session.episodeCount
      }
    };
  }

  window.gameAPI = {
    version: "2.0",
    capabilities: capabilities,

    init: async function init(config) {
      const prepared = prepareEpisode(config || null);
      const notes = prepared.episode.notes.slice();

      try {
        if (runInplaceReset(prepared.appliedLevel)) {
          runtimeState.lastResetMethod = "inplace";
          return {
            ok: true,
            accepted: prepared.episode.accepted,
            applied: prepared.episode.applied,
            notes: notes
          };
        }
      } catch (error) {
        notes.push("init_inplace_reset_failed");
      }

      runtimeState.lastResetMethod = "reload";
      return {
        ok: true,
        accepted: prepared.episode.accepted,
        applied: prepared.episode.applied,
        notes: notes.concat(["state_prepared_without_immediate_reset"])
      };
    },

    reset: async function reset(options) {
      const prepared = prepareEpisode(options || null);
      const notes = prepared.episode.notes.slice();

      try {
        if (runInplaceReset(prepared.appliedLevel)) {
          runtimeState.lastResetMethod = "inplace";
          return {
            ok: true,
            method: "inplace",
            accepted: prepared.episode.accepted,
            applied: prepared.episode.applied,
            notes: notes
          };
        }
      } catch (error) {
        notes.push("inplace_reset_failed");
      }

      runtimeState.lastResetMethod = "reload";
      if (typeof window !== "undefined" && typeof window.__resetRandom === "function") {
        try {
          window.__resetRandom(session.seed);
        } catch (error) {
          notes.push("reload_seed_reset_failed");
        }
      }
      if (typeof window !== "undefined" && window.location && typeof window.location.reload === "function") {
        window.location.reload();
        return {
          ok: true,
          method: "reload",
          accepted: prepared.episode.accepted,
          applied: prepared.episode.applied,
          notes: notes.concat(["falling_back_to_page_reload"])
        };
      }

      runtimeState.lastResetMethod = "unsupported";
      return {
        ok: false,
        method: "unsupported",
        accepted: prepared.episode.accepted,
        applied: prepared.episode.applied,
        notes: notes.concat(["no_reset_method_available"])
      };
    },

    getState: function getState() {
      const now = Date.now();
      const player = getPlayer();
      const worldObjects = getWorldObjects();
      const exitClass = getExitClass();
      const runtimePhase = getRuntimePhase();
      const menuState = getMenuState();
      const coinsCollected = getCoinsCollected();
      const coinsTotal = getCoinsTotal();
      const level = getLevel();
      const playerLives = getPlayerLives(player);
      const playerSnapshot = getPlayerSnapshot(player, playerLives);

      if (!player || !worldObjects) {
        return buildUnavailableState(now);
      }

      updateDeathCount(player);

      const playerPosition = playerSnapshot
        ? { x: playerSnapshot.x, y: playerSnapshot.y, z: playerSnapshot.z }
        : null;
      const exits = getExitSummaries(worldObjects, exitClass);
      const nearestExit = getNearestExit(playerPosition, exits);
      const completionProgress = computeCompletionProgress(coinsCollected, coinsTotal);

      let terminal = null;
      if (playerLives !== null && playerLives <= 0) {
        terminal = { isTerminal: true, outcome: "fail", reason: "player_died" };
        latchTerminal(terminal.outcome, terminal.reason, {
          score: coinsCollected,
          level: level,
          completionProgress: completionProgress,
          lives: playerLives
        });
      } else if (
        (runtimeState.lastStatus === "ready" || runtimeState.lastStatus === "playing") &&
        runtimePhase === 0
      ) {
        terminal = { isTerminal: true, outcome: "quit", reason: "returned_to_menu" };
        latchTerminal(terminal.outcome, terminal.reason, {
          score: coinsCollected,
          level: level,
          completionProgress: completionProgress,
          lives: playerLives
        });
      } else {
        terminal = getLatchedTerminal(now);
      }

      if (!terminal) {
        terminal = {
          isTerminal: false,
          outcome: null,
          reason: null
        };
      }

      const status = getStatus(player, worldObjects, runtimePhase, terminal);
      const isActionable =
        !terminal.isTerminal &&
        !!playerSnapshot &&
        (status === "ready" || status === "playing") &&
        (playerLives === null || playerLives > 0);

      const reportedScore =
        terminal.isTerminal && runtimeState.terminalDisplay.score !== null
          ? runtimeState.terminalDisplay.score
          : coinsCollected;
      const reportedLevel =
        terminal.isTerminal && runtimeState.terminalDisplay.level !== null
          ? runtimeState.terminalDisplay.level
          : level;
      const reportedCompletionProgress =
        terminal.isTerminal && runtimeState.terminalDisplay.completionProgress !== null
          ? runtimeState.terminalDisplay.completionProgress
          : completionProgress;
      const reportedLives =
        terminal.isTerminal && runtimeState.terminalDisplay.lives !== null
          ? runtimeState.terminalDisplay.lives
          : playerLives;

      const gameState = {
        score: reportedScore,
        level: reportedLevel,
        player: playerSnapshot,
        environment: getEnvironmentSnapshot(exits, nearestExit),
        completion_progress: reportedCompletionProgress
      };

      if (exits.length > 0) {
        gameState.entities = exits.slice(0, 4).map(function mapExit(exit) {
          return {
            type: "exit",
            x: exit.x,
            y: exit.y,
            z: exit.z
          };
        });
      }

      runtimeState.lastStatus = status;
      runtimeState.lastRuntimeState = runtimePhase;
      runtimeState.lastMenuState = menuState;
      runtimeState.lastLevel = reportedLevel;

      return {
        schemaVersion: "2.0",
        gameId: GAME_ID,
        seed: session.seed,
        timestampMs: now,
        gameTimeMs: finiteOrNull(now - session.episodeStartMs),
        status: status,
        is_actionable: isActionable,
        terminal: terminal,
        game_state: gameState,
        metrics: {
          primary_score: reportedScore,
          coins: reportedScore,
          coins_total: coinsTotal,
          lives: reportedLives,
          deaths: finiteOrNull(stats.deathCount),
          distance_to_goal: nearestExit ? finiteOrNull(nearestExit.distance) : null
        },
        debug: {
          runtime_state: runtimePhase,
          menu_state: menuState,
          requested_level: session.requestedLevel,
          query_level_target: getQueryTargetLevel(),
          last_reset_method: runtimeState.lastResetMethod,
          world_object_count: intOrNull(worldObjects.length),
          ready: !!getDebugApi()
        }
      };
    }
  };
})();
