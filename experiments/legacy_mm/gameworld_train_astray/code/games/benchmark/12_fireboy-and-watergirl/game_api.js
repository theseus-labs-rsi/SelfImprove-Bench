(function () {
  "use strict";

  var GAME_ID = "12_fireboy-and-watergirl";
  var GAME_KEY = "__fbwg_game";
  var MAX_SCAN = 2000;
  var TERMINAL_LATCH_MS = 4000;
  var DEFAULT_SEED = 42;

  var capabilities = {
    supports_seed: false,
    supports_level_select: true,
    supports_difficulty: false,
    supports_inplace_reset: false,
    supports_reload_reset: true,
    supports_pause_detection: true,
    supports_menu_detection: true,
    provides_actionable_flag: true
  };

  var session = {
    seed: DEFAULT_SEED,
    requestedLevel: null,
    difficulty: null,
    episodeStartMs: Date.now(),
    episodeCount: 0
  };
  var runtime = {
    resetCount: 0,
    gameplayStartMs: null,
    diamondBaseByLevel: {},
    lastKnownLevel: null,
    lastResetMethod: null
  };
  var terminalLatch = {
    isTerminal: false,
    outcome: null,
    reason: null,
    ts: 0
  };

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function finiteInt(value) {
    var num = finiteNumber(value);
    return num === null ? null : Math.trunc(num);
  }

  function assignIfPresent(target, key, value) {
    if (value !== null && value !== undefined) target[key] = value;
  }

  function normalizeSeed(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return value >>> 0;
  }

  function clamp01(value) {
    var num = finiteNumber(value);
    if (num === null) return null;
    if (num < 0) return 0;
    if (num > 1) return 1;
    return num;
  }

  function normalizeOptions(options) {
    var opts = options || {};
    return {
      seed: normalizeSeed(opts.seed),
      level:
        opts.level === undefined || opts.level === null
          ? null
          : Number.isFinite(opts.level)
            ? Math.trunc(opts.level)
            : String(opts.level),
      difficulty:
        Object.prototype.hasOwnProperty.call(opts, "difficulty") && opts.difficulty != null
          ? String(opts.difficulty)
          : null
    };
  }

  function beginEpisode(options, countAsReset) {
    var accepted = normalizeOptions(options);
    var notes = [];

    if (accepted.seed !== null) notes.push("seed_not_supported");
    if (accepted.difficulty !== null) notes.push("difficulty_not_supported");

    session.episodeStartMs = Date.now();
    session.seed = DEFAULT_SEED;
    session.requestedLevel = accepted.level;
    session.difficulty = accepted.difficulty;
    session.episodeCount += 1;

    if (countAsReset !== false) runtime.resetCount += 1;
    runtime.gameplayStartMs = null;
    runtime.diamondBaseByLevel = {};
    clearTerminal();

    return {
      accepted: accepted,
      applied: {
        seed: session.seed,
        level: session.requestedLevel,
        difficulty: null
      },
      notes: notes
    };
  }

  function clearTerminal() {
    terminalLatch.isTerminal = false;
    terminalLatch.outcome = null;
    terminalLatch.reason = null;
    terminalLatch.ts = 0;
  }

  function latchTerminal(outcome, reason) {
    terminalLatch.isTerminal = true;
    terminalLatch.outcome = outcome;
    terminalLatch.reason = reason;
    terminalLatch.ts = Date.now();
  }

  function getLatchedTerminal(now) {
    if (!terminalLatch.isTerminal) return null;
    if (now - terminalLatch.ts > TERMINAL_LATCH_MS) {
      clearTerminal();
      return null;
    }
    return {
      isTerminal: true,
      outcome: terminalLatch.outcome,
      reason: terminalLatch.reason
    };
  }

  function getQueryParam(name) {
    try {
      if (typeof window === "undefined" || !window.location || !window.location.search) return null;
      return new URLSearchParams(window.location.search).get(name);
    } catch (_err) {
      return null;
    }
  }

  function parseLevel(value) {
    if (value == null || value === "") return null;
    var parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    var level = Math.trunc(parsed);
    return level >= 1 ? level : null;
  }

  function getRequestedLevelFromUrl() {
    return parseLevel(getQueryParam("level"));
  }

  function resolveTargetLevel(level) {
    var targetLevel = parseLevel(level);
    if (targetLevel === null) targetLevel = parseLevel(session.requestedLevel);
    if (targetLevel === null) targetLevel = runtime.lastKnownLevel;
    if (targetLevel === null) targetLevel = getRequestedLevelFromUrl();
    return targetLevel;
  }

  function isDirectStartEnabled() {
    if (getRequestedLevelFromUrl() !== null) return true;
    var direct = getQueryParam("direct");
    if (direct === "1" || direct === "true") return true;
    var autostart = getQueryParam("autostart");
    return autostart === "1" || autostart === "true";
  }

  function disablePlayOverlayIfDirect() {
    if (!isDirectStartEnabled()) return;
    try {
      window.playBtn = false;
    } catch (_err) {}
    try {
      if (window._azerionIntegration) window._azerionIntegration.playBtn = false;
    } catch (_err) {}
  }

  function patchPhaserGame() {
    if (typeof window === "undefined") return false;
    if (!window.Phaser || !window.Phaser.Game) return false;
    var OriginalGame = window.Phaser.Game;
    if (OriginalGame.__fbwgWrapped) return true;

    function WrappedGame(config) {
      var game = new OriginalGame(config);
      window[GAME_KEY] = game;
      return game;
    }

    WrappedGame.prototype = OriginalGame.prototype;
    Object.setPrototypeOf(WrappedGame, OriginalGame);
    WrappedGame.__fbwgWrapped = true;
    window.Phaser.Game = WrappedGame;
    return true;
  }

  function ensurePhaserPatched() {
    if (patchPhaserGame()) return;
    var timer = setInterval(function () {
      if (patchPhaserGame()) clearInterval(timer);
    }, 50);
    setTimeout(function () {
      clearInterval(timer);
    }, 30000);
  }

  function getGame() {
    if (typeof window === "undefined") return null;
    if (window[GAME_KEY]) return window[GAME_KEY];
    if (window.Phaser && Array.isArray(window.Phaser.GAMES) && window.Phaser.GAMES.length) {
      return window.Phaser.GAMES[0] || null;
    }
    return null;
  }

  function getCurrentState(game) {
    if (!game || !game.state) return null;
    if (typeof game.state.getCurrentState === "function") {
      return game.state.getCurrentState();
    }
    if (game.state.current && game.state.states) {
      return game.state.states[game.state.current] || null;
    }
    return null;
  }

  function getActiveStateName(game) {
    if (!game || !game.state) return null;
    return game.state.current || null;
  }

  function isSprite(obj) {
    return (
      obj &&
      typeof obj === "object" &&
      typeof obj.x === "number" &&
      typeof obj.y === "number" &&
      (typeof obj.key === "string" || typeof obj.frameName === "string" || obj.body)
    );
  }

  function collectWorldSprites(game) {
    var sprites = [];
    if (!game || !game.world) return sprites;
    var stack = Array.isArray(game.world.children) ? game.world.children.slice() : [];
    var seen = new Set();
    while (stack.length && sprites.length < 200 && seen.size < MAX_SCAN) {
      var obj = stack.pop();
      if (!obj || seen.has(obj)) continue;
      seen.add(obj);
      if (isSprite(obj)) sprites.push(obj);
      if (obj.children && obj.children.length) {
        for (var i = obj.children.length - 1; i >= 0; i -= 1) stack.push(obj.children[i]);
      }
    }
    return sprites;
  }

  function findStateObject(state, pattern) {
    if (!state) return null;
    var keys = Object.keys(state);
    for (var i = 0; i < keys.length; i += 1) {
      var key = keys[i];
      if (!pattern.test(key)) continue;
      var value = state[key];
      if (value && typeof value === "object") return value;
    }
    return null;
  }

  function getNestedSprite(obj) {
    if (!obj || typeof obj !== "object") return null;
    if (isSprite(obj.sprite)) return obj.sprite;
    if (isSprite(obj.bodySprite)) return obj.bodySprite;
    if (isSprite(obj)) return obj;
    return null;
  }

  function getEntityBody(obj) {
    if (!obj || typeof obj !== "object") return null;
    if (obj.body && obj.body.velocity) return obj.body;
    var sprite = getNestedSprite(obj);
    if (sprite && sprite.body && sprite.body.velocity) return sprite.body;
    return null;
  }

  function getEntityPosition(obj) {
    var sprite = getNestedSprite(obj);
    if (sprite) {
      return {
        x: finiteNumber(sprite.x),
        y: finiteNumber(sprite.y)
      };
    }
    if (!obj || typeof obj !== "object") return null;
    if (typeof obj.x === "number" && typeof obj.y === "number") {
      return {
        x: finiteNumber(obj.x),
        y: finiteNumber(obj.y)
      };
    }
    return null;
  }

  function pickCharacter(state, sprites, pattern) {
    var direct = findStateObject(state, pattern);
    if (direct && (isSprite(direct) || getNestedSprite(direct))) return direct;
    for (var i = 0; i < sprites.length; i += 1) {
      var sprite = sprites[i];
      var frame = (sprite.frameName || "").toLowerCase();
      var key = (sprite.key || "").toLowerCase();
      var name = (sprite.name || "").toLowerCase();
      if (pattern.test(frame) || pattern.test(key) || pattern.test(name)) return sprite;
    }
    return null;
  }

  function getCharacterInfo(entity) {
    if (!entity) return null;
    var sprite = getNestedSprite(entity) || entity;
    var body = getEntityBody(entity);
    var alive = null;
    if (typeof entity.isDead === "boolean") alive = !entity.isDead;
    else if (typeof entity.alive === "boolean") alive = entity.alive;
    else if (typeof sprite.alive === "boolean") alive = sprite.alive;
    return {
      x: finiteNumber(sprite.x),
      y: finiteNumber(sprite.y),
      vx: body && body.velocity ? finiteNumber(body.velocity.x) : null,
      vy: body && body.velocity ? finiteNumber(body.velocity.y) : null,
      alive: alive,
      key: sprite.key || entity.key || null
    };
  }

  function findExits(state, sprites) {
    var exits = { fireboy: null, watergirl: null };

    var fireboyExit =
      (state && state.door1) || findStateObject(state, /fireboy.*exit|fire.*door|red.*exit|red.*door/i);
    var watergirlExit =
      (state && state.door2) || findStateObject(state, /watergirl.*exit|water.*door|blue.*exit|blue.*door/i);
    var fireboyPos = getEntityPosition(fireboyExit);
    var watergirlPos = getEntityPosition(watergirlExit);
    if (fireboyPos) exits.fireboy = fireboyPos;
    if (watergirlPos) exits.watergirl = watergirlPos;

    if (!exits.fireboy || !exits.watergirl) {
      for (var i = 0; i < sprites.length; i += 1) {
        var sprite = sprites[i];
        var combined =
          (sprite.frameName || "").toLowerCase() +
          " " +
          (sprite.key || "").toLowerCase() +
          " " +
          (sprite.name || "").toLowerCase();
        if (!exits.fireboy && /fireboy.*exit|fire.*door|red.*exit|red.*door/i.test(combined)) {
          exits.fireboy = { x: finiteNumber(sprite.x), y: finiteNumber(sprite.y) };
        }
        if (!exits.watergirl && /watergirl.*exit|water.*door|blue.*exit|blue.*door/i.test(combined)) {
          exits.watergirl = { x: finiteNumber(sprite.x), y: finiteNumber(sprite.y) };
        }
        if (exits.fireboy && exits.watergirl) break;
      }
    }

    return exits;
  }

  function normalizeDiamondCounts(counts) {
    var input = counts || {};
    return {
      fireboy: Math.max(0, finiteInt(input.fireboy) || 0),
      watergirl: Math.max(0, finiteInt(input.watergirl) || 0),
      common: Math.max(0, finiteInt(input.common) || 0)
    };
  }

  function subtractDiamondCounts(left, right) {
    var a = normalizeDiamondCounts(left);
    var b = normalizeDiamondCounts(right);
    return {
      fireboy: Math.max(0, a.fireboy - b.fireboy),
      watergirl: Math.max(0, a.watergirl - b.watergirl),
      common: Math.max(0, a.common - b.common)
    };
  }

  function buildDiamondState(totalCounts, remainingCounts, collectedCounts) {
    var total = normalizeDiamondCounts(totalCounts);
    var remaining = normalizeDiamondCounts(remainingCounts);
    var collected = normalizeDiamondCounts(collectedCounts);
    return {
      fireboy: {
        total: finiteInt(total.fireboy),
        remaining: finiteInt(Math.min(total.fireboy, remaining.fireboy)),
        collected: finiteInt(Math.min(total.fireboy, collected.fireboy))
      },
      watergirl: {
        total: finiteInt(total.watergirl),
        remaining: finiteInt(Math.min(total.watergirl, remaining.watergirl)),
        collected: finiteInt(Math.min(total.watergirl, collected.watergirl))
      },
      common: {
        total: finiteInt(total.common),
        remaining: finiteInt(Math.min(total.common, remaining.common)),
        collected: finiteInt(Math.min(total.common, collected.common))
      }
    };
  }

  function getMapDiamondTotals(state) {
    if (!state || !state.mapData || !state.mapData.objects || !Array.isArray(state.mapData.objects.Chars)) {
      return null;
    }
    var chars = state.mapData.objects.Chars;
    var totals = { fireboy: 0, watergirl: 0, common: 0 };
    var found = false;
    for (var i = 0; i < chars.length; i += 1) {
      var obj = chars[i];
      if (!obj || obj.type !== "diamond") continue;
      found = true;
      var props = obj.properties || {};
      var charType = typeof props.char === "string" ? props.char.toLowerCase() : "";
      if (charType === "fb") totals.fireboy += 1;
      else if (charType === "wg") totals.watergirl += 1;
      else totals.common += 1;
    }
    return found ? totals : null;
  }

  function getCollectedDiamondCounts(state) {
    if (!state) return null;
    var fireboy = state.pers1 && state.pers1.data ? finiteInt(state.pers1.data.diamonds) : null;
    var watergirl = state.pers2 && state.pers2.data ? finiteInt(state.pers2.data.diamonds) : null;
    var commonFireboy =
      state.pers1 && state.pers1.data ? finiteInt(state.pers1.data.silverDiamond) : null;
    var commonWatergirl =
      state.pers2 && state.pers2.data ? finiteInt(state.pers2.data.silverDiamond) : null;
    if (fireboy === null && watergirl === null && commonFireboy === null && commonWatergirl === null) {
      return null;
    }
    return {
      fireboy: Math.max(0, fireboy || 0),
      watergirl: Math.max(0, watergirl || 0),
      common: Math.max(0, (commonFireboy || 0) + (commonWatergirl || 0))
    };
  }

  function isDiamondContainer(obj) {
    if (!obj || typeof obj !== "object") return false;
    if (typeof obj.x !== "number" || typeof obj.y !== "number") return false;
    if (typeof obj.data !== "object" || typeof obj.data.char !== "string") return false;
    if (!obj.children || !obj.children.length) return false;
    for (var i = 0; i < obj.children.length; i += 1) {
      var child = obj.children[i];
      var frame = (child && child.frameName) || "";
      if (/^diamond_/i.test(frame)) return true;
    }
    return false;
  }

  function collectLiveDiamondCounts(game) {
    var counts = { fireboy: 0, watergirl: 0, common: 0 };
    if (!game || !game.world) return null;
    var stack = Array.isArray(game.world.children) ? game.world.children.slice() : [];
    var seen = new Set();
    var found = false;
    while (stack.length && seen.size < MAX_SCAN) {
      var obj = stack.pop();
      if (!obj || seen.has(obj)) continue;
      seen.add(obj);
      if (isDiamondContainer(obj)) {
        var exists = typeof obj.exists === "boolean" ? obj.exists : true;
        var alive = typeof obj.alive === "boolean" ? obj.alive : true;
        if (exists && alive) {
          found = true;
          var charType = String(obj.data.char || "").toLowerCase();
          if (charType === "fb") counts.fireboy += 1;
          else if (charType === "wg") counts.watergirl += 1;
          else counts.common += 1;
        }
      }
      if (obj.children && obj.children.length) {
        for (var i = obj.children.length - 1; i >= 0; i -= 1) stack.push(obj.children[i]);
      }
    }
    return found ? counts : null;
  }

  function calculateDistance(x1, y1, x2, y2) {
    if (x1 === null || y1 === null || x2 === null || y2 === null) return null;
    return Math.hypot(x2 - x1, y2 - y1);
  }

  function getDistanceToExit(character, exit) {
    if (!character || !exit) return null;
    return calculateDistance(character.x, character.y, exit.x, exit.y);
  }

  function collectGemSprites(state) {
    var gems = [];
    if (!state) return gems;
    var keys = Object.keys(state);
    for (var i = 0; i < keys.length; i += 1) {
      var key = keys[i];
      if (!/gem|diamond/i.test(key)) continue;
      var group = state[key];
      if (!group || !group.children || !group.children.length) continue;
      for (var j = 0; j < group.children.length; j += 1) {
        var child = group.children[j];
        if (!child) continue;
        var exists = typeof child.exists === "boolean" ? child.exists : true;
        var alive = typeof child.alive === "boolean" ? child.alive : true;
        if (!exists || !alive || typeof child.x !== "number" || typeof child.y !== "number") continue;
        var frame = (child.frameName || "").toLowerCase();
        var spriteKey = (child.key || "").toLowerCase();
        var type = "common";
        if (frame.indexOf("red") >= 0 || spriteKey.indexOf("red") >= 0) type = "fireboy";
        else if (frame.indexOf("blue") >= 0 || spriteKey.indexOf("blue") >= 0) type = "watergirl";
        gems.push({ type: type });
      }
    }
    return gems;
  }

  function getRemainingDiamonds(state, game) {
    var liveCounts = collectLiveDiamondCounts(game);
    if (liveCounts) return liveCounts;
    var remaining = { fireboy: 0, watergirl: 0, common: 0 };
    var gems = collectGemSprites(state);
    for (var i = 0; i < gems.length; i += 1) {
      var type = gems[i].type;
      if (type === "fireboy") remaining.fireboy += 1;
      else if (type === "watergirl") remaining.watergirl += 1;
      else remaining.common += 1;
    }
    return remaining;
  }

  function trackDiamonds(level, state, game) {
    var totalsFromMap = getMapDiamondTotals(state);
    var collectedFromState = getCollectedDiamondCounts(state);
    var remainingFromScene = getRemainingDiamonds(state, game);
    if (totalsFromMap) {
      var collectedResolved = collectedFromState;
      if (!collectedResolved && remainingFromScene) {
        collectedResolved = subtractDiamondCounts(totalsFromMap, remainingFromScene);
      }
      if (!collectedResolved) collectedResolved = { fireboy: 0, watergirl: 0, common: 0 };
      var remainingResolved = remainingFromScene || subtractDiamondCounts(totalsFromMap, collectedResolved);
      return buildDiamondState(totalsFromMap, remainingResolved, collectedResolved);
    }

    var remaining = remainingFromScene || { fireboy: 0, watergirl: 0, common: 0 };
    var levelKey = level == null ? "unknown" : String(level);
    var base = runtime.diamondBaseByLevel[levelKey];
    if (!base) {
      base = {
        fireboy: remaining.fireboy,
        watergirl: remaining.watergirl,
        common: remaining.common
      };
      runtime.diamondBaseByLevel[levelKey] = base;
    } else {
      if (remaining.fireboy > base.fireboy) base.fireboy = remaining.fireboy;
      if (remaining.watergirl > base.watergirl) base.watergirl = remaining.watergirl;
      if (remaining.common > base.common) base.common = remaining.common;
    }

    var fireboyCollected = Math.max(0, base.fireboy - remaining.fireboy);
    var watergirlCollected = Math.max(0, base.watergirl - remaining.watergirl);
    var commonCollected = Math.max(0, base.common - remaining.common);

    return {
      fireboy: {
        total: finiteInt(base.fireboy),
        remaining: finiteInt(remaining.fireboy),
        collected: finiteInt(fireboyCollected)
      },
      watergirl: {
        total: finiteInt(base.watergirl),
        remaining: finiteInt(remaining.watergirl),
        collected: finiteInt(watergirlCollected)
      },
      common: {
        total: finiteInt(base.common),
        remaining: finiteInt(remaining.common),
        collected: finiteInt(commonCollected)
      }
    };
  }

  function extractLevel(state) {
    if (!state) return null;
    var directKeys = ["level", "levelIndex", "currentLevel", "levelId", "levelNum"];
    for (var i = 0; i < directKeys.length; i += 1) {
      var key = directKeys[i];
      if (Number.isFinite(state[key])) return Math.trunc(state[key]);
    }
    var keys = Object.keys(state);
    for (var j = 0; j < keys.length; j += 1) {
      var k = keys[j];
      if (!/level/i.test(k)) continue;
      var value = state[k];
      if (Number.isFinite(value)) return Math.trunc(value);
      if (value && typeof value === "object") {
        if (Number.isFinite(value.index)) return Math.trunc(value.index);
        if (Number.isFinite(value.id)) return Math.trunc(value.id);
        if (Number.isFinite(value.number)) return Math.trunc(value.number);
      }
    }
    return null;
  }

  function getTotalDiamondInfo(diamonds) {
    return {
      total:
        (diamonds.fireboy.total || 0) + (diamonds.watergirl.total || 0) + (diamonds.common.total || 0),
      collected:
        (diamonds.fireboy.collected || 0) +
        (diamonds.watergirl.collected || 0) +
        (diamonds.common.collected || 0)
    };
  }

  function calculateLevelProgress(diamonds, fireboy, watergirl, exits) {
    var totals = getTotalDiamondInfo(diamonds);
    var diamondPart = totals.total > 0 ? (totals.collected / totals.total) * 0.7 : 0;

    var exitPart = 0;
    var fireboyDist = fireboy && exits.fireboy ? getDistanceToExit(fireboy, exits.fireboy) : null;
    if (fireboyDist !== null) exitPart += (1 - Math.min(fireboyDist / 1000, 1)) * 0.15;
    var watergirlDist = watergirl && exits.watergirl ? getDistanceToExit(watergirl, exits.watergirl) : null;
    if (watergirlDist !== null) exitPart += (1 - Math.min(watergirlDist / 1000, 1)) * 0.15;

    if (totals.total === 0 && fireboyDist === null && watergirlDist === null) return null;
    return clamp01(diamondPart + exitPart);
  }

  function detectOutcome(stateName, state) {
    if (!stateName) return null;
    var normalized = String(stateName).toLowerCase();
    if (normalized === "endgame") {
      var successFlag =
        state &&
        state.levelState &&
        Object.prototype.hasOwnProperty.call(state.levelState, "success")
          ? state.levelState.success
          : null;
      if (successFlag === true) {
        return { outcome: "success", reason: "level_completed" };
      }
      if (successFlag === false) {
        return { outcome: "fail", reason: "level_failed" };
      }
      return { outcome: "fail", reason: "level_ended" };
    }
    if (/complete|finish|win|won|victory|clear/i.test(stateName)) {
      return { outcome: "success", reason: "level_completed" };
    }
    if (/fail|lose|lost|dead|gameover/i.test(stateName)) {
      return { outcome: "fail", reason: "level_failed" };
    }
    return null;
  }

  function getStatus(stateName, state) {
    if (!stateName) return "loading";
    var normalized = String(stateName).toLowerCase();
    if (/boot|preload|load/i.test(normalized)) return "loading";
    if (/menu|select|map|intro|levelmenu|buy/i.test(normalized)) return "menu";
    if (/pause/i.test(normalized)) return "paused";
    if (normalized === "level") {
      return state && state.levelStarted === true ? "playing" : "loading";
    }
    if (/play|game|level/i.test(normalized)) return "playing";
    return "menu";
  }

  function getGameTimeMs(game, now) {
    if (game && game.time && typeof game.time.totalElapsedSeconds === "function") {
      var totalSec = finiteNumber(game.time.totalElapsedSeconds());
      if (totalSec !== null) return Math.max(0, Math.round(totalSec * 1000));
    }
    if (runtime.gameplayStartMs !== null) return finiteNumber(Math.max(0, now - runtime.gameplayStartMs));
    return null;
  }

  function buildEntities(fireboy, watergirl, exits) {
    var entities = [];
    if (fireboy) {
      var fireboyEntity = { type: "fireboy" };
      assignIfPresent(fireboyEntity, "x", fireboy.x);
      assignIfPresent(fireboyEntity, "y", fireboy.y);
      assignIfPresent(fireboyEntity, "vx", fireboy.vx);
      assignIfPresent(fireboyEntity, "vy", fireboy.vy);
      if (fireboy.alive !== null) {
        fireboyEntity.props = { alive: fireboy.alive };
      }
      entities.push(fireboyEntity);
    }
    if (watergirl) {
      var watergirlEntity = { type: "watergirl" };
      assignIfPresent(watergirlEntity, "x", watergirl.x);
      assignIfPresent(watergirlEntity, "y", watergirl.y);
      assignIfPresent(watergirlEntity, "vx", watergirl.vx);
      assignIfPresent(watergirlEntity, "vy", watergirl.vy);
      if (watergirl.alive !== null) {
        watergirlEntity.props = { alive: watergirl.alive };
      }
      entities.push(watergirlEntity);
    }
    if (exits.fireboy) {
      var fireboyExit = { type: "fireboy_exit" };
      assignIfPresent(fireboyExit, "x", exits.fireboy.x);
      assignIfPresent(fireboyExit, "y", exits.fireboy.y);
      entities.push(fireboyExit);
    }
    if (exits.watergirl) {
      var watergirlExit = { type: "watergirl_exit" };
      assignIfPresent(watergirlExit, "x", exits.watergirl.x);
      assignIfPresent(watergirlExit, "y", exits.watergirl.y);
      entities.push(watergirlExit);
    }
    return entities.length ? entities : null;
  }

  function buildEnvironment(exits) {
    return {
      objective: "both_characters_reach_their_matching_exit",
      exits: {
        fireboy: exits.fireboy || null,
        watergirl: exits.watergirl || null
      }
    };
  }

  function buildLoadingState(now) {
    return {
      schemaVersion: "2.0",
      gameId: GAME_ID,
      seed: session.seed,
      timestampMs: now,
      gameTimeMs: null,
      status: "loading",
      is_actionable: false,
      terminal: {
        isTerminal: false,
        outcome: null,
        reason: null
      },
      game_state: {
        score: null,
        level: resolveTargetLevel(null),
        player: null,
        environment: null,
        completion_progress: null
      },
      metrics: {
        primary_score: null,
        coins: null,
        coins_total: null,
        distance_to_goal: null,
        attempts: finiteInt(runtime.resetCount),
        fireboy_distance_to_goal: null,
        watergirl_distance_to_goal: null
      },
      debug: {
        state_name: null,
        direct_start_state:
          typeof window !== "undefined" ? window.__fbwgDirectStartState || null : null,
        last_reset_method: runtime.lastResetMethod
      }
    };
  }

  function sleep(ms) {
    return new Promise(function (resolve) {
      setTimeout(resolve, ms);
    });
  }

  async function waitFor(getter, timeoutMs, intervalMs) {
    var start = Date.now();
    while (Date.now() - start < timeoutMs) {
      try {
        var value = getter();
        if (value) return value;
      } catch (_err) {}
      await sleep(intervalMs);
    }
    return null;
  }

  function getSplashLoader() {
    if (typeof window === "undefined" || !window.h5branding) return null;
    var SplashLoader = window.h5branding.SplashLoader;
    if (!SplashLoader || typeof SplashLoader.getInstance !== "function") return null;
    try {
      return SplashLoader.getInstance();
    } catch (_err) {
      return null;
    }
  }

  function isSplashReadyForDirectStart() {
    var splash = getSplashLoader();
    if (!splash) return false;
    return splash.loaded === true && Number(splash.progress) >= 100;
  }

  function triggerSplashPlayButton() {
    if (typeof window !== "undefined" && window.__fbwgDirectStartPlayTriggered) return false;
    var splash = getSplashLoader();
    if (!splash || typeof splash.onPlayButtonClick !== "function") return false;
    try {
      if (typeof window !== "undefined") window.__fbwgDirectStartPlayTriggered = true;
      splash.onPlayButtonClick();
      return true;
    } catch (_err) {
      if (typeof window !== "undefined") window.__fbwgDirectStartPlayTriggered = false;
      return false;
    }
  }

  async function advancePastLoad(game) {
    var deadline = Date.now() + 30000;
    var splashReadySince = 0;
    while (Date.now() < deadline) {
      var stateName = getActiveStateName(game) || "";
      if (stateName && !/^load$/i.test(stateName)) return true;
      if (isSplashReadyForDirectStart()) {
        if (!splashReadySince) splashReadySince = Date.now();
        if (Date.now() - splashReadySince >= 1500) {
          triggerSplashPlayButton();
        }
      } else {
        splashReadySince = 0;
      }
      await sleep(250);
    }
    var finalStateName = getActiveStateName(game) || "";
    return !!finalStateName && !/^load$/i.test(finalStateName);
  }

  async function directStart() {
    if (!isDirectStartEnabled()) return;
    if (typeof window === "undefined") return;
    if (/^running/.test(window.__fbwgDirectStartState || "") || window.__fbwgDirectStartState === "done") return;
    window.__fbwgDirectStartState = "running_wait_game";
    window.__fbwgDirectStartPlayTriggered = false;

    var game = await waitFor(getGame, 30000, 100);
    if (!game || !game.state) {
      window.__fbwgDirectStartState = "failed_no_game";
      return;
    }

    window.__fbwgDirectStartState = "running_wait_load";
    await advancePastLoad(game);

    window.__fbwgDirectStartState = "running_wait_menu";
    var menuState = await waitFor(function () {
      var state = getCurrentState(game);
      return state && typeof state.skipToLevel === "function" ? state : null;
    }, 30000, 100);
    if (!menuState) {
      window.__fbwgDirectStartState = "failed_no_menu_state";
      return;
    }

    var temple = null;
    try {
      var resp = await fetch("data/forest/temple.json", { cache: "no-store" });
      temple = await resp.json();
    } catch (_err) {
      window.__fbwgDirectStartState = "failed_fetch_temple";
      return;
    }

    var requestedLevel = getRequestedLevelFromUrl();
    var levelData = null;
    if (temple && Array.isArray(temple.levels)) {
      if (requestedLevel !== null) {
        levelData =
          temple.levels.find(function (l) {
            return l && l.id === requestedLevel;
          }) || null;
      }
      if (!levelData) {
        levelData =
          temple.levels.find(function (l) {
            return l && (l.initial === true || l.initial === 1);
          }) || null;
      }
      if (!levelData) levelData = temple.levels[0] || null;
    }

    if (!levelData) {
      window.__fbwgDirectStartState = "failed_no_level_data";
      return;
    }

    try {
      game.currentTemple = temple;
    } catch (_err) {}

    try {
      window.__fbwgDirectStartState = "running_start_level";
      menuState.skipToLevel(levelData);
      var switched = await waitFor(function () {
        var active = (getActiveStateName(game) || "").toLowerCase();
        return active && !/boot|load|menu|select|intro|levelmenu|buy/.test(active);
      }, 6000, 100);
      window.__fbwgDirectStartState = switched ? "done" : "failed_still_menu";
    } catch (_err) {
      window.__fbwgDirectStartState = "failed_skip_to_level";
    }
  }

  function startDirectStartWatchdog() {
    if (!isDirectStartEnabled()) return;
    if (typeof window === "undefined") return;
    if (window.__fbwgDirectStartWatchdogStarted) return;
    window.__fbwgDirectStartWatchdogStarted = true;

    var maxTicks = 300;
    var ticks = 0;
    var timer = setInterval(function () {
      ticks += 1;
      disablePlayOverlayIfDirect();

      var game = getGame();
      var stateName = (getActiveStateName(game) || "").toLowerCase();
      var directState = window.__fbwgDirectStartState || "";
      var isNotPlaying = !stateName || /boot|load|menu|select|intro|levelmenu|buy/.test(stateName);
      if (!/^running/.test(directState) && isNotPlaying) {
        directStart();
      }

      if (ticks >= maxTicks || (directState === "done" && !isNotPlaying)) {
        clearInterval(timer);
      }
    }, 200);
  }

  disablePlayOverlayIfDirect();
  ensurePhaserPatched();
  directStart();
  startDirectStartWatchdog();

  window.gameAPI = {
    version: "2.0",
    capabilities: capabilities,
    init: async function init(config) {
      var episode = beginEpisode(config, false);
      var initLevel = resolveTargetLevel(episode.accepted.level);
      session.requestedLevel = initLevel;
      episode.applied.level = initLevel;
      runtime.lastResetMethod = null;
      return {
        ok: true,
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes.length ? episode.notes : []
      };
    },
    reset: async function reset(options) {
      var episode = beginEpisode(options || null, true);
      var targetLevel = resolveTargetLevel(episode.accepted.level);
      session.requestedLevel = targetLevel;
      episode.applied.level = targetLevel;

      runtime.lastResetMethod = "reload";

      if (typeof window !== "undefined" && typeof window.__resetRandom === "function") {
        try {
          window.__resetRandom();
        } catch (_err) {}
      }

      var result = {
        ok: true,
        method: "reload",
        accepted: episode.accepted,
        applied: episode.applied,
        notes: episode.notes.length ? episode.notes : []
      };

      if (typeof window !== "undefined" && window.location) {
        try {
          var url = new URL(window.location.href);
          url.searchParams.set("direct", "1");
          if (targetLevel !== null) {
            url.searchParams.set("level", String(targetLevel));
          } else {
            url.searchParams.delete("level");
          }
          url.searchParams.delete("autostart");
          window.location.href = url.toString();
          return result;
        } catch (_err) {
          result.notes.push("reload_navigation_failed");
        }
        if (typeof window.location.reload === "function") {
          window.location.reload();
          return result;
        }
      }

      runtime.lastResetMethod = "unsupported";
      result.ok = false;
      result.method = "unsupported";
      result.notes.push("window_location_unavailable");
      return result;
    },
    getState: function getState() {
      var now = Date.now();
      var game = getGame();
      var stateName = getActiveStateName(game);
      var state = getCurrentState(game);
      if (!game || !state) {
        return buildLoadingState(now);
      }
      var sprites = collectWorldSprites(game);

      var fireboyEntity = (state && state.pers1) || pickCharacter(state, sprites, /fireboy|fire_/i);
      var watergirlEntity = (state && state.pers2) || pickCharacter(state, sprites, /watergirl|water_/i);
      var fireboy = getCharacterInfo(fireboyEntity);
      var watergirl = getCharacterInfo(watergirlEntity);

      var level = extractLevel(state);
      if (level !== null) runtime.lastKnownLevel = level;
      var diamonds = trackDiamonds(level, state, game);
      var exits = findExits(state, sprites);
      var fireboyDistanceToExit = getDistanceToExit(fireboy, exits.fireboy);
      var watergirlDistanceToExit = getDistanceToExit(watergirl, exits.watergirl);
      var levelProgress = calculateLevelProgress(diamonds, fireboy, watergirl, exits);

      var detected = detectOutcome(stateName, state);
      if (detected) latchTerminal(detected.outcome, detected.reason);
      var terminal = getLatchedTerminal(now) || {
        isTerminal: false,
        outcome: null,
        reason: null
      };

      var status = terminal.isTerminal ? "terminal" : getStatus(stateName, state);
      if (status === "playing" && runtime.gameplayStartMs === null) runtime.gameplayStartMs = now;

      var totals = getTotalDiamondInfo(diamonds);
      var distanceToGoal = null;
      if (fireboyDistanceToExit !== null && watergirlDistanceToExit !== null) {
        distanceToGoal = (fireboyDistanceToExit + watergirlDistanceToExit) / 2;
      } else if (fireboyDistanceToExit !== null) {
        distanceToGoal = fireboyDistanceToExit;
      } else if (watergirlDistanceToExit !== null) {
        distanceToGoal = watergirlDistanceToExit;
      }

      var environment = buildEnvironment(exits);
      var entities = buildEntities(fireboy, watergirl, exits);
      var gameState = {
        score: finiteInt(totals.collected),
        level: level,
        completion_progress: levelProgress,
        player: fireboy,
        environment: environment,
        watergirl: watergirl,
        diamonds: diamonds,
        fireboy_distance_to_goal: finiteNumber(fireboyDistanceToExit),
        watergirl_distance_to_goal: finiteNumber(watergirlDistanceToExit)
      };

      if (entities !== null) {
        gameState.entities = entities;
      }

      return {
        schemaVersion: "2.0",
        gameId: GAME_ID,
        seed: session.seed,
        timestampMs: now,
        gameTimeMs: getGameTimeMs(game, now),
        status: status,
        is_actionable: status === "playing" || status === "ready",
        terminal: terminal,
        game_state: gameState,
        metrics: {
          primary_score: finiteInt(totals.collected),
          coins: finiteInt(totals.collected),
          coins_total: finiteInt(totals.total),
          distance_to_goal: finiteNumber(distanceToGoal),
          attempts: finiteInt(Math.max(1, runtime.resetCount)),
          fireboy_distance_to_goal: finiteNumber(fireboyDistanceToExit),
          watergirl_distance_to_goal: finiteNumber(watergirlDistanceToExit)
        },
        debug: {
          state_name: stateName || null,
          direct_start_state:
            typeof window !== "undefined" ? window.__fbwgDirectStartState || null : null,
          last_reset_method: runtime.lastResetMethod
        }
      };
    }
  };
})();
