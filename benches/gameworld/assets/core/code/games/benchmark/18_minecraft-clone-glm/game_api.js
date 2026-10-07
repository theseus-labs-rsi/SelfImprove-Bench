(function () {
  "use strict";

  var GAME_ID = "18_minecraft-clone-glm";
  var DEFAULT_SEED = 42;
  var TERMINAL_LATCH_MS = 3500;
  var capabilities = {
    supports_seed: true,
    supports_level_select: false,
    supports_difficulty: false,
    supports_inplace_reset: true,
    supports_reload_reset: true,
    supports_pause_detection: true,
    supports_menu_detection: true,
    provides_actionable_flag: true,
  };

  var session = {
    seed: DEFAULT_SEED,
    requestedLevel: null,
    requestedDifficulty: null,
    episodeStartMs: Date.now(),
    episodeCount: 0,
  };

  var runtime = {
    resetCount: 0,
    gameplayStartMs: null,
    baselineItemCounts: null,
    lastResetMethod: null,
  };

  var terminalLatch = {
    isTerminal: false,
    outcome: null,
    reason: null,
    ts: 0,
  };

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function finiteInt(value) {
    var num = finiteNumber(value);
    return num === null ? null : Math.trunc(num);
  }

  function boolOrNull(value) {
    return typeof value === "boolean" ? value : null;
  }

  function cloneDict(input) {
    var out = {};
    if (!input || typeof input !== "object") return out;
    var keys = Object.keys(input);
    for (var i = 0; i < keys.length; i += 1) {
      var key = keys[i];
      out[key] = input[key];
    }
    return out;
  }

  function normalizeSeed(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return value >>> 0;
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

  function getLatchedTerminal(nowMs) {
    if (!terminalLatch.isTerminal) return null;
    if (nowMs - terminalLatch.ts > TERMINAL_LATCH_MS) {
      clearTerminal();
      return null;
    }
    return {
      isTerminal: true,
      outcome: terminalLatch.outcome,
      reason: terminalLatch.reason,
    };
  }

  function isElementVisibleById(id) {
    if (typeof document === "undefined") return null;
    var el = document.getElementById(id);
    if (!el) return null;
    var style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    if (!style) return null;
    return style.display !== "none" && style.visibility !== "hidden";
  }

  function getGameInstance() {
    if (typeof game !== "undefined" && game && typeof game === "object") return game;
    return null;
  }

  function getPlayerInstance() {
    if (typeof player !== "undefined" && player && typeof player === "object") return player;
    return null;
  }

  function getInventoryState() {
    if (typeof inventory !== "undefined" && inventory && typeof inventory === "object") return inventory;
    return null;
  }

  function getUIState() {
    if (typeof uiState !== "undefined" && uiState && typeof uiState === "object") return uiState;
    return null;
  }

  function getWorldSeed() {
    if (typeof worldSeed !== "undefined" && Number.isFinite(worldSeed)) {
      return Math.trunc(worldSeed);
    }
    return session.seed;
  }

  function resolveEpisodeSeed(normalizedOptions) {
    var opts = normalizedOptions || {};
    if (Number.isFinite(opts.seed)) return Math.trunc(opts.seed);
    if (Number.isFinite(session.seed)) return Math.trunc(session.seed);
    if (Number.isFinite(getWorldSeed())) return Math.trunc(getWorldSeed());
    return Math.trunc(Date.now());
  }

  function beginEpisode(options, countAsReset) {
    var accepted = normalizeOptions(options);
    var resolvedSeed = resolveEpisodeSeed(accepted);
    var notes = [];

    if (accepted.level !== null) notes.push("level_not_supported");
    if (accepted.difficulty !== null) notes.push("difficulty_not_supported");

    session.episodeStartMs = Date.now();
    session.seed = Number.isFinite(resolvedSeed) ? Math.trunc(resolvedSeed) : null;
    session.requestedLevel = accepted.level;
    session.requestedDifficulty = accepted.difficulty;

    session.episodeCount += 1;
    if (countAsReset !== false) runtime.resetCount += 1;
    runtime.gameplayStartMs = null;
    runtime.baselineItemCounts = null;
    clearTerminal();

    return {
      accepted: accepted,
      applied: {
        seed: session.seed,
        level: null,
        difficulty: null,
      },
      notes: notes,
    };
  }

  function buildBlockLookup() {
    var byId = {};
    var keys = [];
    if (typeof BlockType !== "undefined" && BlockType && typeof BlockType === "object") {
      var rawKeys = Object.keys(BlockType);
      for (var i = 0; i < rawKeys.length; i += 1) {
        var rawKey = rawKeys[i];
        var rawId = BlockType[rawKey];
        if (!Number.isFinite(rawId)) continue;
        var id = Math.trunc(rawId);
        var key = String(rawKey).toLowerCase();
        byId[id] = key;
        if (key !== "air") keys.push(key);
      }
    }
    return {
      byId: byId,
      keys: keys,
    };
  }

  function getItemKeyFromType(typeId, lookup) {
    if (!Number.isFinite(typeId)) return null;
    var id = Math.trunc(typeId);
    if (lookup && lookup.byId && lookup.byId[id]) return lookup.byId[id];
    return "type_" + String(id);
  }

  function buildEmptyCounts(lookup) {
    var counts = {};
    if (!lookup || !Array.isArray(lookup.keys)) return counts;
    for (var i = 0; i < lookup.keys.length; i += 1) {
      counts[lookup.keys[i]] = 0;
    }
    return counts;
  }

  function addSlotCounts(slotArray, counts, lookup) {
    if (!Array.isArray(slotArray)) return;
    for (var i = 0; i < slotArray.length; i += 1) {
      var slot = slotArray[i];
      if (!slot || typeof slot !== "object") continue;
      var key = getItemKeyFromType(slot.type, lookup);
      if (!key) continue;
      var slotCount = finiteInt(slot.count);
      if (slotCount === null || slotCount <= 0) continue;
      if (!Object.prototype.hasOwnProperty.call(counts, key)) counts[key] = 0;
      counts[key] += slotCount;
    }
  }

  function countOccupiedSlots(slotArray) {
    if (!Array.isArray(slotArray)) return null;
    var used = 0;
    for (var i = 0; i < slotArray.length; i += 1) {
      if (slotArray[i] && typeof slotArray[i] === "object") used += 1;
    }
    return used;
  }

  function summarizeStack(stack, lookup) {
    if (!stack || typeof stack !== "object") return null;
    var key = getItemKeyFromType(stack.type, lookup);
    var count = finiteInt(stack.count);
    return {
      type: key,
      count: count,
    };
  }

  function snapshotInventory(lookup) {
    var inv = getInventoryState();
    var counts = buildEmptyCounts(lookup);

    if (!inv) {
      return {
        counts: counts,
        selectedSlot: null,
        selectedItemKey: null,
        selectedItemCount: null,
        selectedStack: null,
        hotbarOccupied: null,
        inventoryOccupied: null,
      };
    }

    var hotbar = Array.isArray(inv.hotbar) ? inv.hotbar : [];
    var slots = Array.isArray(inv.slots) ? inv.slots : [];
    addSlotCounts(hotbar, counts, lookup);
    addSlotCounts(slots, counts, lookup);

    var selectedSlot = finiteInt(inv.selectedSlot);
    var selectedStack = null;
    if (selectedSlot !== null && selectedSlot >= 0 && selectedSlot < hotbar.length) {
      selectedStack = hotbar[selectedSlot];
    }
    var selectedTypeId =
      selectedStack && Number.isFinite(selectedStack.type) ? Math.trunc(selectedStack.type) : null;
    var selectedSummary = summarizeStack(selectedStack, lookup);

    return {
      counts: counts,
      selectedSlot: selectedSlot,
      selectedTypeId: selectedTypeId,
      selectedItemKey: selectedSummary ? selectedSummary.type : null,
      selectedItemCount: selectedSummary ? selectedSummary.count : null,
      selectedStack: selectedSummary,
      hotbarOccupied: countOccupiedSlots(hotbar),
      inventoryOccupied: countOccupiedSlots(slots),
    };
  }

  function sumCounts(counts) {
    if (!counts || typeof counts !== "object") return null;
    var keys = Object.keys(counts);
    var total = 0;
    for (var i = 0; i < keys.length; i += 1) {
      var val = finiteInt(counts[keys[i]]);
      if (val !== null && val > 0) total += val;
    }
    return total;
  }

  function countUniqueItems(counts) {
    if (!counts || typeof counts !== "object") return null;
    var keys = Object.keys(counts);
    var total = 0;
    for (var i = 0; i < keys.length; i += 1) {
      var val = finiteInt(counts[keys[i]]);
      if (val !== null && val > 0) total += 1;
    }
    return total;
  }

  function ensureBaseline(currentCounts) {
    if (!runtime.baselineItemCounts) {
      runtime.baselineItemCounts = cloneDict(currentCounts);
    }
  }

  function computeCountDelta(currentCounts) {
    var baseline = runtime.baselineItemCounts || {};
    var delta = {};
    var keyMap = {};
    var key;
    var i;

    var currentKeys = Object.keys(currentCounts || {});
    for (i = 0; i < currentKeys.length; i += 1) keyMap[currentKeys[i]] = true;
    var baselineKeys = Object.keys(baseline);
    for (i = 0; i < baselineKeys.length; i += 1) keyMap[baselineKeys[i]] = true;

    var allKeys = Object.keys(keyMap);
    var netTotal = 0;
    var positiveTotal = 0;
    for (i = 0; i < allKeys.length; i += 1) {
      key = allKeys[i];
      var cur = finiteInt(currentCounts[key]);
      var base = finiteInt(baseline[key]);
      if (cur === null) cur = 0;
      if (base === null) base = 0;
      var diff = cur - base;
      delta[key] = diff;
      netTotal += diff;
      if (diff > 0) positiveTotal += diff;
    }

    return {
      delta: delta,
      netTotal: netTotal,
      positiveTotal: positiveTotal,
    };
  }

  function getDayTimeValue() {
    if (typeof getDayTime === "function") {
      var fromFn = finiteNumber(getDayTime());
      if (fromFn !== null) return fromFn;
    }
    if (typeof lightingState !== "undefined" && lightingState && typeof lightingState === "object") {
      return finiteNumber(lightingState.time);
    }
    return null;
  }

  function getDayCountValue() {
    if (typeof lightingState !== "undefined" && lightingState && typeof lightingState === "object") {
      return finiteInt(lightingState.dayCount);
    }
    return null;
  }

  function getLoadedChunksValue() {
    if (typeof chunkManager === "undefined" || !chunkManager || !chunkManager.chunks) return null;
    return finiteInt(chunkManager.chunks.size);
  }

  function getBreakProgressValue() {
    if (typeof getBreakProgress !== "function") return null;
    var progress = finiteNumber(getBreakProgress());
    if (progress === null) return null;
    if (progress < 0) return 0;
    if (progress > 1) return 1;
    return progress;
  }

  function getTargetedBlockSummary(lookup) {
    if (typeof targetedBlock === "undefined" || !targetedBlock || !targetedBlock.position) return null;
    var pos = targetedBlock.position;
    var blockType = targetedBlock.block;
    return {
      x: finiteInt(pos.x),
      y: finiteInt(pos.y),
      z: finiteInt(pos.z),
      face: targetFace || null,
      block_type: getItemKeyFromType(blockType, lookup),
    };
  }

  function getBlockTypeAt(x, y, z, lookup) {
    if (
      typeof chunkManager === "undefined" ||
      !chunkManager ||
      typeof chunkManager.getBlock !== "function"
    ) {
      return null;
    }
    var blockType = chunkManager.getBlock(Math.floor(x), Math.floor(y), Math.floor(z));
    return getItemKeyFromType(blockType, lookup);
  }

  function getToolInfoFromTypeId(typeId, lookup) {
    if (!Number.isFinite(typeId)) {
      return {
        item_key: null,
        tool_type: null,
        tool_speed: null,
        is_tool: false,
      };
    }
    var id = Math.trunc(typeId);
    var props = typeof BlockProperties !== "undefined" && BlockProperties ? BlockProperties[id] : null;
    var toolType = props && typeof props.toolType === "string" ? props.toolType : null;
    var toolSpeed = props && Number.isFinite(props.toolSpeed) ? props.toolSpeed : null;
    return {
      item_key: getItemKeyFromType(id, lookup),
      tool_type: toolType,
      tool_speed: toolSpeed,
      is_tool: toolType !== null,
    };
  }

  function getBiomeNameAt(x, z) {
    if (typeof getBiome !== "function" || typeof Biome === "undefined" || !Biome) return null;
    var biomeId = getBiome(Math.floor(x), Math.floor(z));
    var biomeKeys = Object.keys(Biome);
    for (var i = 0; i < biomeKeys.length; i += 1) {
      var key = biomeKeys[i];
      if (Biome[key] === biomeId) return String(key).toLowerCase();
    }
    return Number.isFinite(biomeId) ? String(Math.trunc(biomeId)) : null;
  }

  function getSurfaceHeightAt(x, z) {
    if (typeof getTerrainHeight !== "function" || typeof getBiome !== "function") return null;
    var wx = Math.floor(x);
    var wz = Math.floor(z);
    return finiteInt(getTerrainHeight(wx, wz, getBiome(wx, wz)));
  }

  function buildPlayerMiningContext(playerInst, lookup) {
    if (!playerInst || !playerInst.position) return null;
    var pos = playerInst.position;
    var px = finiteNumber(pos.x);
    var py = finiteNumber(pos.y);
    var pz = finiteNumber(pos.z);
    if (px === null || py === null || pz === null) return null;

    var surfaceY = getSurfaceHeightAt(px, pz);
    var depthBelowSurface =
      surfaceY !== null ? finiteNumber(Math.max(0, surfaceY - py)) : null;

    return {
      block_x: finiteInt(px),
      block_y: finiteInt(py),
      block_z: finiteInt(pz),
      biome: getBiomeNameAt(px, pz),
      feet_block: getBlockTypeAt(px, py, pz, lookup),
      head_block: getBlockTypeAt(px, py + PLAYER_EYE_HEIGHT, pz, lookup),
      block_below: getBlockTypeAt(px, py - 1, pz, lookup),
      block_above: getBlockTypeAt(px, py + PLAYER_HEIGHT, pz, lookup),
      surface_y: surfaceY,
      depth_below_surface: depthBelowSurface,
      is_underground: depthBelowSurface !== null ? depthBelowSurface >= 2 : null,
    };
  }

  function createEmptyResourceSummary() {
    return {
      count: 0,
      nearest_distance: null,
      nearest_x: null,
      nearest_y: null,
      nearest_z: null,
      nearest_dx: null,
      nearest_dy: null,
      nearest_dz: null,
    };
  }

  // Scan a compact volume around the player so mining tasks can reason about nearby stone/ores.
  function buildNearbyResourceScan(playerInst, lookup) {
    if (
      !playerInst ||
      !playerInst.position ||
      typeof chunkManager === "undefined" ||
      !chunkManager ||
      typeof chunkManager.getBlock !== "function"
    ) {
      return null;
    }

    var px = finiteInt(playerInst.position.x);
    var py = finiteInt(playerInst.position.y);
    var pz = finiteInt(playerInst.position.z);
    if (px === null || py === null || pz === null) return null;

    var radiusHorizontal = 6;
    var radiusVertical = 8;
    var tracked = {
      grass: createEmptyResourceSummary(),
      dirt: createEmptyResourceSummary(),
      stone: createEmptyResourceSummary(),
      wood_log: createEmptyResourceSummary(),
      leaves: createEmptyResourceSummary(),
      coal_ore: createEmptyResourceSummary(),
      iron_ore: createEmptyResourceSummary(),
    };

    for (var dx = -radiusHorizontal; dx <= radiusHorizontal; dx += 1) {
      for (var dz = -radiusHorizontal; dz <= radiusHorizontal; dz += 1) {
        for (var dy = -radiusVertical; dy <= radiusVertical; dy += 1) {
          var wx = px + dx;
          var wy = py + dy;
          var wz = pz + dz;
          var blockKey = getBlockTypeAt(wx, wy, wz, lookup);
          if (!blockKey || !Object.prototype.hasOwnProperty.call(tracked, blockKey)) continue;

          var bucket = tracked[blockKey];
          bucket.count += 1;

          var distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (bucket.nearest_distance === null || distance < bucket.nearest_distance) {
            bucket.nearest_distance = finiteNumber(distance);
            bucket.nearest_x = wx;
            bucket.nearest_y = wy;
            bucket.nearest_z = wz;
            bucket.nearest_dx = dx;
            bucket.nearest_dy = dy;
            bucket.nearest_dz = dz;
          }
        }
      }
    }

    return {
      radius_horizontal: radiusHorizontal,
      radius_vertical: radiusVertical,
      grass: tracked.grass,
      dirt: tracked.dirt,
      stone: tracked.stone,
      wood_log: tracked.wood_log,
      leaves: tracked.leaves,
      coal_ore: tracked.coal_ore,
      iron_ore: tracked.iron_ore,
    };
  }

  function buildTargetedBlockState(lookup, inventorySnapshot) {
    var targetedSummary = getTargetedBlockSummary(lookup);
    var selectedTool = getToolInfoFromTypeId(
      inventorySnapshot ? inventorySnapshot.selectedTypeId : null,
      lookup
    );

    if (!targetedSummary) {
      return {
        exists: false,
        x: null,
        y: null,
        z: null,
        face: null,
        distance: null,
        block_type: null,
        hardness: null,
        harvest_tool: null,
        selected_tool_type: selectedTool.tool_type,
        selected_tool_matches: false,
      };
    }

    var targetTypeId =
      typeof targetedBlock !== "undefined" && targetedBlock && Number.isFinite(targetedBlock.block)
        ? Math.trunc(targetedBlock.block)
        : null;
    var targetProps =
      targetTypeId !== null && typeof BlockProperties !== "undefined" && BlockProperties
        ? BlockProperties[targetTypeId]
        : null;
    var harvestTool =
      targetProps && typeof targetProps.harvestTool === "string" ? targetProps.harvestTool : null;

    return {
      exists: true,
      x: targetedSummary.x,
      y: targetedSummary.y,
      z: targetedSummary.z,
      face: targetedSummary.face,
      distance:
        typeof targetedBlock !== "undefined" && targetedBlock
          ? finiteNumber(targetedBlock.distance)
          : null,
      block_type: targetedSummary.block_type,
      hardness: targetProps ? finiteNumber(targetProps.hardness) : null,
      harvest_tool: harvestTool,
      selected_tool_type: selectedTool.tool_type,
      selected_tool_matches:
        harvestTool !== null && selectedTool.tool_type !== null
          ? harvestTool === selectedTool.tool_type
          : false,
    };
  }

  function buildPlayerState(playerInst) {
    if (!playerInst) return null;

    var pos = playerInst.position || {};
    var vel = playerInst.velocity || {};

    return {
      x: finiteNumber(pos.x),
      y: finiteNumber(pos.y),
      z: finiteNumber(pos.z),
      vx: finiteNumber(vel.x),
      vy: finiteNumber(vel.y),
      vz: finiteNumber(vel.z),
      health: finiteNumber(playerInst.health),
      hunger: finiteNumber(playerInst.hunger),
      is_grounded: boolOrNull(playerInst.isGrounded),
      is_sprinting: boolOrNull(playerInst.isSprinting),
      is_in_water: boolOrNull(playerInst.isInWater),
      is_underwater: boolOrNull(playerInst.isUnderwater),
      is_locked: boolOrNull(playerInst.isLocked),
      yaw: finiteNumber(playerInst.yaw),
      pitch: finiteNumber(playerInst.pitch),
    };
  }

  function maybeUpdateTerminal(playerInst) {
    if (!playerInst) return;
    var hp = finiteNumber(playerInst.health);
    if (hp !== null && hp <= 0) {
      latchTerminal("fail", "player_died");
    }
  }

  function computeStatus(nowMs, terminal) {
    var gameInst = getGameInstance();
    var playerInst = getPlayerInstance();
    var ui = getUIState();
    var mainMenuVisible = isElementVisibleById("main-menu");
    var pauseMenuVisible = isElementVisibleById("pause-menu");
    var inventoryVisible = isElementVisibleById("inventory-screen");
    var clickToPlayVisible = isElementVisibleById("click-to-play");

    if (terminal && terminal.isTerminal) {
      return {
        status: "terminal",
        ui: {
          is_main_menu_visible: mainMenuVisible,
          is_pause_menu_visible: pauseMenuVisible,
          is_inventory_visible: inventoryVisible,
          is_click_to_play_visible: clickToPlayVisible,
          is_loading: ui ? boolOrNull(ui.isLoading) : null,
        },
      };
    }

    if (gameInst && gameInst.isRunning === true && runtime.gameplayStartMs === null) {
      runtime.gameplayStartMs = nowMs;
    }

    if (!gameInst) {
      return {
        status: mainMenuVisible ? "ready" : "loading",
        ui: {
          is_main_menu_visible: mainMenuVisible,
          is_pause_menu_visible: pauseMenuVisible,
          is_inventory_visible: inventoryVisible,
          is_click_to_play_visible: clickToPlayVisible,
          is_loading: ui ? boolOrNull(ui.isLoading) : null,
        },
      };
    }

    if (gameInst.isRunning !== true) {
      return {
        status: mainMenuVisible ? "ready" : "loading",
        ui: {
          is_main_menu_visible: mainMenuVisible,
          is_pause_menu_visible: pauseMenuVisible,
          is_inventory_visible: inventoryVisible,
          is_click_to_play_visible: clickToPlayVisible,
          is_loading: ui ? boolOrNull(ui.isLoading) : null,
        },
      };
    }

    if (
      gameInst.isPaused === true ||
      (ui && ui.isPauseMenuOpen === true) ||
      (ui && ui.isInventoryOpen === true) ||
      pauseMenuVisible === true ||
      inventoryVisible === true
    ) {
      return {
        status: "paused",
        ui: {
          is_main_menu_visible: mainMenuVisible,
          is_pause_menu_visible: pauseMenuVisible,
          is_inventory_visible: inventoryVisible,
          is_click_to_play_visible: clickToPlayVisible,
          is_loading: ui ? boolOrNull(ui.isLoading) : null,
        },
      };
    }

    if (playerInst && playerInst.isLocked === true) {
      return {
        status: "playing",
        ui: {
          is_main_menu_visible: mainMenuVisible,
          is_pause_menu_visible: pauseMenuVisible,
          is_inventory_visible: inventoryVisible,
          is_click_to_play_visible: clickToPlayVisible,
          is_loading: ui ? boolOrNull(ui.isLoading) : null,
        },
      };
    }

    return {
      status: "ready",
      ui: {
        is_main_menu_visible: mainMenuVisible,
        is_pause_menu_visible: pauseMenuVisible,
        is_inventory_visible: inventoryVisible,
        is_click_to_play_visible: clickToPlayVisible,
        is_loading: ui ? boolOrNull(ui.isLoading) : null,
      },
    };
  }

  function buildEnvironment(playerMiningContext, nearbyResources, targeted, breakProgress) {
    return {
      world_seed: finiteInt(getWorldSeed()),
      day_time: getDayTimeValue(),
      day_count: getDayCountValue(),
      loaded_chunks: getLoadedChunksValue(),
      player_context: playerMiningContext,
      nearby_resources: nearbyResources,
      targeted_block: targeted,
      break_progress: breakProgress,
    };
  }

  async function resetEpisode(options) {
    var setup = beginEpisode(options || {});
    var seed = setup.applied.seed;

    if (typeof window !== "undefined" && typeof window.__resetRandom === "function") {
      try {
        window.__resetRandom();
      } catch (_err) {}
    }

    if (typeof startGame === "function") {
      try {
        startGame(seed, null);
        runtime.lastResetMethod = "soft_restart";
        return {
          ok: true,
          method: "soft_restart",
          accepted: setup.accepted,
          applied: setup.applied,
          notes: setup.notes,
        };
      } catch (_err2) {}
    }

    if (typeof startNewGame === "function") {
      try {
        startNewGame();
        runtime.lastResetMethod = "soft_restart";
        return {
          ok: true,
          method: "soft_restart",
          accepted: setup.accepted,
          applied: {
            seed: finiteInt(getWorldSeed()),
            level: null,
            difficulty: null,
          },
          notes:
            setup.accepted.seed !== null
              ? setup.notes.concat(["seed_not_guaranteed_with_startNewGame"])
              : setup.notes,
        };
      } catch (_err3) {}
    }

    if (
      typeof window !== "undefined" &&
      window.location &&
      typeof window.location.reload === "function"
    ) {
      runtime.lastResetMethod = "reload";
      window.location.reload();
      return {
        ok: true,
        method: "reload",
        accepted: setup.accepted,
        applied: setup.applied,
        notes: setup.notes,
      };
    }

    runtime.lastResetMethod = "unsupported";
    return {
      ok: false,
      method: "unsupported",
      accepted: setup.accepted,
      applied: setup.applied,
      notes: setup.notes.concat(["reset_not_available"]),
    };
  }

  window.gameAPI = {
    version: "2.0",
    capabilities: capabilities,

    init: async function init(config) {
      var setup = beginEpisode(config || {}, false);
      runtime.lastResetMethod = "init";
      return {
        ok: true,
        accepted: setup.accepted,
        applied: setup.applied,
        notes: setup.notes,
      };
    },

    reset: async function reset(options) {
      return resetEpisode(options || {});
    },

    getState: function getState() {
      var nowMs = Date.now();
      var lookup = buildBlockLookup();
      var playerInst = getPlayerInstance();

      maybeUpdateTerminal(playerInst);
      var terminal = getLatchedTerminal(nowMs) || {
        isTerminal: false,
        outcome: null,
        reason: null,
      };

      var statusInfo = computeStatus(nowMs, terminal);
      var status = statusInfo.status;

      var gameTimeMs = null;
      if (runtime.gameplayStartMs !== null) {
        gameTimeMs = finiteNumber(Math.max(0, nowMs - runtime.gameplayStartMs));
      }

      var inventorySnapshot = snapshotInventory(lookup);
      ensureBaseline(inventorySnapshot.counts);
      var gains = computeCountDelta(inventorySnapshot.counts);
      var totalItems = sumCounts(inventorySnapshot.counts);
      var uniqueItems = countUniqueItems(inventorySnapshot.counts);
      var selectedTool = getToolInfoFromTypeId(inventorySnapshot.selectedTypeId, lookup);

      var playerState = buildPlayerState(playerInst);
      var playerMiningContext = buildPlayerMiningContext(playerInst, lookup);
      var nearbyResources = buildNearbyResourceScan(playerInst, lookup);
      var targeted = buildTargetedBlockState(lookup, inventorySnapshot);
      var breakProgress = getBreakProgressValue();
      var environment = buildEnvironment(
        playerMiningContext,
        nearbyResources,
        targeted,
        breakProgress
      );

      var entities = [];
      if (inventorySnapshot.selectedItemKey) {
        entities.push({
          type: "held_item",
          state: inventorySnapshot.selectedItemKey,
          props: {
            slot: inventorySnapshot.selectedSlot,
            count: inventorySnapshot.selectedItemCount,
          },
        });
      }
      if (targeted.exists) {
        entities.push({
          type: "target_block",
          x: targeted.x,
          y: targeted.y,
          z: targeted.z,
          state: targeted.block_type,
          props: {
            face: targeted.face,
            distance: targeted.distance,
            harvest_tool: targeted.harvest_tool,
          },
        });
      }
      if (!entities.length) entities = null;

      var isActionable = status === "ready" || status === "playing";

      return {
        schemaVersion: "2.0",
        gameId: GAME_ID,
        seed: finiteInt(getWorldSeed()),
        timestampMs: nowMs,
        gameTimeMs: gameTimeMs,
        status: status,
        is_actionable: isActionable,
        terminal: terminal,
        game_state: {
          score: totalItems,
          level: null,
          completion_progress: null,
          player: playerState,
          environment: environment,
          entities: entities,
          inventory: {
            selected_slot: inventorySnapshot.selectedSlot,
            selected_item: inventorySnapshot.selectedItemKey,
            selected_item_count: inventorySnapshot.selectedItemCount,
            selected_tool_type: selectedTool.tool_type,
            selected_tool_speed: selectedTool.tool_speed,
            item_counts: inventorySnapshot.counts,
            item_gains: gains.delta,
          },
        },
        metrics: {
          primary_score: totalItems,
          attempts: finiteInt(runtime.resetCount + 1),
          hp: playerState ? playerState.health : null,
          items_total: totalItems,
          items_unique: uniqueItems,
          hotbar_occupied_slots: inventorySnapshot.hotbarOccupied,
          inventory_occupied_slots: inventorySnapshot.inventoryOccupied,
          held_selected_item_count: inventorySnapshot.selectedItemCount,
          held_selected_item_type: inventorySnapshot.selectedItemKey,
          selected_tool_type: selectedTool.tool_type,
          selected_tool_speed: selectedTool.tool_speed,
          collected_net_total: gains.netTotal,
          collected_positive_total: gains.positiveTotal,
          day_time: getDayTimeValue(),
          day_count: getDayCountValue(),
          loaded_chunks: getLoadedChunksValue(),
          break_progress: breakProgress,
          targeted_block_exists: targeted.exists,
          targeted_block_type: targeted.block_type,
          targeted_block_distance: targeted.distance,
          targeted_block_harvest_tool: targeted.harvest_tool,
          selected_tool_matches_target: targeted.selected_tool_matches,
          surface_y: playerMiningContext ? playerMiningContext.surface_y : null,
          depth_below_surface: playerMiningContext ? playerMiningContext.depth_below_surface : null,
          is_underground: playerMiningContext ? playerMiningContext.is_underground : null,
          biome: playerMiningContext ? playerMiningContext.biome : null,
          nearby_stone_count:
            nearbyResources && nearbyResources.stone ? nearbyResources.stone.count : null,
          nearest_stone_distance:
            nearbyResources && nearbyResources.stone
              ? nearbyResources.stone.nearest_distance
              : null,
          nearby_wood_log_count:
            nearbyResources && nearbyResources.wood_log ? nearbyResources.wood_log.count : null,
          nearest_wood_log_distance:
            nearbyResources && nearbyResources.wood_log
              ? nearbyResources.wood_log.nearest_distance
              : null,
          nearby_leaves_count:
            nearbyResources && nearbyResources.leaves ? nearbyResources.leaves.count : null,
          nearest_leaves_distance:
            nearbyResources && nearbyResources.leaves
              ? nearbyResources.leaves.nearest_distance
              : null,
          nearby_iron_ore_count:
            nearbyResources && nearbyResources.iron_ore ? nearbyResources.iron_ore.count : null,
          nearest_iron_ore_distance:
            nearbyResources && nearbyResources.iron_ore
              ? nearbyResources.iron_ore.nearest_distance
              : null,
        },
        debug: {
          ui: statusInfo.ui,
          selected_stack: inventorySnapshot.selectedStack,
          baseline_item_counts: runtime.baselineItemCounts,
          last_reset_method: runtime.lastResetMethod,
        },
      };
    },

    start: function start() {
      return this.reset({});
    },
  };
})();
