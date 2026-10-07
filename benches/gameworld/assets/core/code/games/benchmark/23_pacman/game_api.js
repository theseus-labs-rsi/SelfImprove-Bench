(function () {
  "use strict";

  var GAME_ID = "23_pacman";
  var DEFAULT_SEED = 42;

  var capabilities = {
    supports_seed: true,
    supports_level_select: false,
    supports_difficulty: false,
    supports_inplace_reset: false,
    supports_reload_reset: true,
    supports_pause_detection: true,
    supports_menu_detection: true,
    provides_actionable_flag: true,
  };

  var session = {
    seed: DEFAULT_SEED,
    difficulty: null,
    episodeStartMs: Date.now(),
  };

  var tracking = {
    deaths: 0,
    lastGameOver: false,
    attempts: 0,
    lastResetMethod: null,
  };

  var autoFlow = {
    timerId: null,
    lastStartAttemptMs: 0,
    gameOverSinceMs: null,
    reloadScheduled: false,
  };

  function intOrNull(value) {
    return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : null;
  }

  function normalizeSeed(value) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) return null;
    return Math.trunc(numeric) >>> 0;
  }

  function normalizeOptions(options) {
    var opts = options || {};
    return {
      seed: normalizeSeed(opts.seed),
      level:
        opts.level === undefined || opts.level === null
          ? null
          : typeof opts.level === "number" && Number.isFinite(opts.level)
            ? Math.trunc(opts.level)
            : String(opts.level),
      difficulty:
        opts.difficulty === undefined || opts.difficulty === null
          ? null
          : String(opts.difficulty),
    };
  }

  function getCurrentSeed() {
    if (typeof window !== "undefined" && typeof window.__getDeterministicSeed === "function") {
      var seed = normalizeSeed(window.__getDeterministicSeed());
      return seed === null ? DEFAULT_SEED : seed;
    }
    return DEFAULT_SEED;
  }

  function applySeed(seed) {
    if (typeof window === "undefined") return null;
    if (typeof window.__setDeterministicSeed === "function") {
      return normalizeSeed(window.__setDeterministicSeed(seed));
    }
    if (typeof window.__resetRandom === "function") {
      return normalizeSeed(window.__resetRandom(seed));
    }
    return null;
  }

  function beginEpisode(options) {
    var normalized = normalizeOptions(options);
    var notes = [];
    var appliedSeed = session.seed;

    if (normalized.level !== null) notes.push("level_not_supported");
    if (normalized.difficulty !== null) notes.push("difficulty_not_supported");

    if (normalized.seed !== null) {
      appliedSeed = applySeed(normalized.seed);
      if (appliedSeed === null) {
        notes.push("seed_not_supported");
        appliedSeed = getCurrentSeed();
      }
    } else {
      appliedSeed = getCurrentSeed();
    }

    session.episodeStartMs = Date.now();
    session.seed = appliedSeed === null ? getCurrentSeed() : appliedSeed;
    session.difficulty = null;
    tracking.deaths = 0;
    tracking.lastGameOver = false;
    tracking.attempts += 1;

    return {
      accepted: normalized,
      applied: {
        seed: session.seed,
        level: null,
        difficulty: null,
      },
      notes: notes,
    };
  }

  function query(selector) {
    try {
      return document.querySelector(selector);
    } catch (_e) {
      return null;
    }
  }

  function queryAll(selector) {
    try {
      return document.querySelectorAll(selector);
    } catch (_e) {
      return [];
    }
  }

  function readNumberFromText(el) {
    if (!el) return null;
    var text = el.textContent;
    if (!text) return null;
    var cleaned = text.replace(/[^0-9\-]/g, "");
    if (!cleaned) return null;
    var value = parseInt(cleaned, 10);
    return Number.isFinite(value) ? value : null;
  }

  function isVisible(el) {
    if (!el) return false;
    var style = null;
    try {
      style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    } catch (_e) {
      style = null;
    }
    if (style) {
      if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
        return false;
      }
    }
    if (el.offsetParent === null && el.getClientRects && el.getClientRects().length === 0) {
      return false;
    }
    return true;
  }

  function countVisible(selector) {
    var list = queryAll(selector);
    var count = 0;
    for (var i = 0; i < list.length; i += 1) {
      if (isVisible(list[i])) count += 1;
    }
    return count > 0 ? count : null;
  }

  function getMazeIndex(playground) {
    if (!playground || typeof playground.className !== "string") return null;
    var match = playground.className.match(/\bmaze-(\d+)\b/i);
    if (!match) return null;
    var index = parseInt(match[1], 10);
    return Number.isFinite(index) ? index : null;
  }

  function detectStatus() {
    var playground = query(".js-pacman-playground");
    if (!playground || !isVisible(playground)) {
      return {
        status: "loading",
        terminal: { isTerminal: false, outcome: null, reason: null },
        flags: {
          playground_visible: false,
        },
      };
    }

    var flags = {
      playground_visible: true,
      game_over: isVisible(query(".game-over")),
      splash: isVisible(query(".splash")),
      paused: isVisible(query(".paused")),
      ready: isVisible(query(".start-ready")),
      start_p1: isVisible(query(".start-p1")),
      start_button: isVisible(query(".splash .start")),
    };

    if (flags.game_over) {
      return {
        status: "terminal",
        terminal: {
          isTerminal: true,
          outcome: "fail",
          reason: "game_over",
        },
        flags: flags,
      };
    }

    if (flags.splash || flags.start_p1 || flags.start_button) {
      return {
        status: "menu",
        terminal: { isTerminal: false, outcome: null, reason: null },
        flags: flags,
      };
    }

    if (flags.paused) {
      return {
        status: "paused",
        terminal: { isTerminal: false, outcome: null, reason: null },
        flags: flags,
      };
    }

    if (flags.ready) {
      return {
        status: "ready",
        terminal: { isTerminal: false, outcome: null, reason: null },
        flags: flags,
      };
    }

    return {
      status: "playing",
      terminal: { isTerminal: false, outcome: null, reason: null },
      flags: flags,
    };
  }

  function updateDeaths(flags) {
    var gameOverNow = !!(flags && flags.game_over);
    if (gameOverNow && !tracking.lastGameOver) tracking.deaths += 1;
    tracking.lastGameOver = gameOverNow;
    return tracking.deaths;
  }

  function readLivesGuess() {
    var visiblePacmans = countVisible(".js-pacman-playground .pacman");
    if (visiblePacmans === null) return null;
    return visiblePacmans > 0 ? Math.max(0, visiblePacmans - 1) : 0;
  }

  function readLevelGuess(playground) {
    var bonusCount = countVisible(".js-pacman-playground .bonus");
    if (bonusCount !== null) return bonusCount;
    return getMazeIndex(playground);
  }

  function clickStartButton() {
    var btn = query(".splash .start");
    if (btn && typeof btn.click === "function" && isVisible(btn)) {
      btn.click();
      return true;
    }
    return false;
  }

  function dispatchSyntheticPress(key, code) {
    var target = document.body || document.documentElement;
    if (!target || typeof target.dispatchEvent !== "function") return;

    try {
      target.dispatchEvent(new KeyboardEvent("keydown", {
        key: key,
        code: code,
        bubbles: true,
        cancelable: true,
      }));
      target.dispatchEvent(new KeyboardEvent("keyup", {
        key: key,
        code: code,
        bubbles: true,
        cancelable: true,
      }));
    } catch (_e) {}
  }

  function dispatchStartKeys() {
    dispatchSyntheticPress("Enter", "Enter");
    dispatchSyntheticPress(" ", "Space");
    dispatchSyntheticPress("ArrowUp", "ArrowUp");
  }

  function triggerStart() {
    var clicked = clickStartButton();
    if (!clicked) dispatchStartKeys();
    return clicked;
  }

  async function resetGame(options) {
    var episode = beginEpisode(options || {});

    if (window.location && typeof window.location.reload === "function") {
      tracking.lastResetMethod = "reload";
      window.location.reload();
      return {
        ok: true,
        method: "reload",
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes,
      };
    }

    tracking.lastResetMethod = "unsupported";
    return {
      ok: false,
      method: "unsupported",
      accepted: episode.accepted,
      applied: episode.applied,
      notes: episode.notes.concat(["no_reset_method_available"]),
    };
  }

  function autoFlowTick() {
    var now = Date.now();
    var statusInfo = detectStatus();
    var flags = statusInfo.flags || {};
    var isStartable = statusInfo.status === "menu" || statusInfo.status === "ready" || !!flags.start_button;
    var isGameOver = statusInfo.status === "terminal" || !!flags.game_over;

    if (isStartable) {
      if (now - autoFlow.lastStartAttemptMs >= 350) {
        triggerStart();
        autoFlow.lastStartAttemptMs = now;
      }
      autoFlow.gameOverSinceMs = null;
      autoFlow.reloadScheduled = false;
      return;
    }

    if (isGameOver) {
      if (autoFlow.gameOverSinceMs === null) autoFlow.gameOverSinceMs = now;
      if (now - autoFlow.lastStartAttemptMs >= 350) {
        triggerStart();
        autoFlow.lastStartAttemptMs = now;
      }

      if (!autoFlow.reloadScheduled && now - autoFlow.gameOverSinceMs >= 1800) {
        autoFlow.reloadScheduled = true;
        resetGame({});
      }
      return;
    }

    autoFlow.gameOverSinceMs = null;
    autoFlow.reloadScheduled = false;
  }

  function startAutoFlow() {
    if (autoFlow.timerId !== null) return;
    autoFlow.timerId = window.setInterval(autoFlowTick, 150);
    autoFlowTick();
  }

  function installAutoFlow() {
    if (document.readyState === "loading") {
      window.addEventListener("DOMContentLoaded", startAutoFlow, { once: true });
      window.addEventListener("load", startAutoFlow, { once: true });
      return;
    }
    startAutoFlow();
  }

  function buildEnvironment(mazeIndex) {
    if (mazeIndex === null) return null;
    return {
      maze_index: mazeIndex,
    };
  }

  function buildDebug(playground, flags, highScore) {
    return {
      playground_class: playground ? playground.className : null,
      playground_visible: !!(flags && flags.playground_visible),
      splash_visible: !!(flags && flags.splash),
      paused_visible: !!(flags && flags.paused),
      ready_visible: !!(flags && flags.ready),
      start_button_visible: !!(flags && flags.start_button),
      high_score: highScore,
      requested_difficulty: session.difficulty,
      last_reset_method: tracking.lastResetMethod,
    };
  }

  installAutoFlow();

  window.gameAPI = {
    version: "2.0",
    capabilities: capabilities,

    init: async function init(config) {
      startAutoFlow();
      var episode = beginEpisode(config || {});
      tracking.lastResetMethod = "init";
      return {
        ok: true,
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes,
      };
    },

    getState: function getState() {
      var now = Date.now();
      var playground = query(".js-pacman-playground");
      var statusInfo = detectStatus();

      var score = readNumberFromText(query(".p1-score span"));
      if (score === null) score = readNumberFromText(query(".score"));
      var highScore = readNumberFromText(query(".high-score span"));
      var lives = readLivesGuess();
      var mazeIndex = getMazeIndex(playground);
      var level = readLevelGuess(playground);
      var deaths = updateDeaths(statusInfo.flags);

      var gameTimeMs =
        statusInfo.status === "loading"
          ? null
          : intOrNull(now - session.episodeStartMs);

      return {
        schemaVersion: "2.0",
        gameId: GAME_ID,
        seed: session.seed,
        timestampMs: now,
        gameTimeMs: gameTimeMs,
        status: statusInfo.status,
        is_actionable: statusInfo.status === "ready" || statusInfo.status === "playing",
        terminal: statusInfo.terminal,
        game_state: {
          score: score,
          level: level,
          player: null,
          environment: buildEnvironment(mazeIndex),
          maze_index: mazeIndex,
        },
        metrics: {
          primary_score: score,
          high_score: highScore,
          lives: lives,
          deaths: deaths,
          attempts: intOrNull(tracking.attempts),
          maze_index: mazeIndex,
        },
        debug: buildDebug(playground, statusInfo.flags, highScore),
      };
    },

    reset: resetGame,

    start: function start() {
      return triggerStart();
    },

    restart: function restart() {
      return resetGame(null);
    },
  };
})();
