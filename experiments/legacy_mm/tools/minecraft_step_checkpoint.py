"""Minecraft (18_minecraft-clone-glm) browser checkpoint for step-level resume.

Never changes the GameWorld checkout: instrumented copies of game_api.js and
index.html are served to the browser for the episode (evaluator runtime state
and the game's seeded RNG stream become readable/writable). Game rules and
score formulas are unchanged.

What a checkpoint holds, and why each part is needed for an identical restore:
- every loaded chunk's block array (content-addressed, run-length encoded).
  Terrain generation writes trees across chunk borders, so the world depends on
  chunk-load order; regenerating from the seed is NOT guaranteed identical.
- the block modification map, player (position/velocity/view/stats), inventory,
  in-progress mining, time of day, mobs, evaluator runtime (score baseline and
  clock) and the RNG stream positions.
After restore, the same capture is taken again and must match the saved one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _sibling(name: str, file: str):
    # GameWorld ships its own `tools` package; load siblings by path.
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _HERE / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_astray = _sibling('_csd_astray_browser_checkpoint', 'astray_step_checkpoint.py')
_bridge = _sibling('_csd_gameworld_browser_checkpoint', 'gameworld_browser_checkpoint.py')
_state = sys.modules['_csd_astray_step_state']

GAME_ID = '18_minecraft-clone-glm'
# Agent memory helpers are game-independent.
snapshot_memory = _astray.snapshot_memory
restore_memory = _astray.restore_memory

_KNOWN_API = frozenset({'6e1254211b86f418d155bbb77b5529d14e3fde261929b075272dace603d2aaa1'})
_KNOWN_INDEX = frozenset({'d160e783bf312108b8afa146f8551e02f4583eeb84806309d7282d325991af3d'})


def instrument(game_dir: Path) -> dict[str, bytes]:
    """Return the patched files to serve for this game, or refuse."""
    api = (game_dir / 'game_api.js').read_bytes()
    index = (game_dir / 'index.html').read_bytes()
    if hashlib.sha256(api).hexdigest() not in _KNOWN_API:
        raise _state.UnsafeResume('Minecraft game_api.js changed; refusing checkpoint instrumentation')
    if hashlib.sha256(index).hexdigest() not in _KNOWN_INDEX:
        raise _state.UnsafeResume('Minecraft index.html changed; refusing checkpoint instrumentation')
    return {'game_api.js': _bridge.instrument_api(api),
            'index.html': _bridge.instrument_index_random(index)}


# Two independent 32-bit FNV-1a variants per chunk: cheap change detection in
# the page. Stored blobs are addressed by the SHA-256 of their encoding.
_HASH_CHUNKS = """() => {
  const out = {};
  for (const [key, chunk] of chunkManager.chunks) {
    const b = chunk.blocks;
    let h1 = 0x811c9dc5, h2 = 0x01000193 ^ b.length;
    for (let i = 0; i < b.length; i++) {
      h1 = Math.imul(h1 ^ b[i], 0x01000193);
      h2 = Math.imul(h2 ^ b[i], 0x5bd1e995) ^ (h2 >>> 15);
    }
    out[key] = (h1 >>> 0).toString(16).padStart(8, '0') + (h2 >>> 0).toString(16).padStart(8, '0')
      + ':' + (chunk.isGenerated ? 1 : 0);
  }
  return out;
}"""

_EXPORT_CHUNKS = """(keys) => {
  const out = {};
  for (const key of keys) {
    const chunk = chunkManager.chunks.get(key);
    if (!chunk) throw new Error('Chunk vanished during checkpoint: ' + key);
    const b = chunk.blocks, runs = [];
    let value = b[0], count = 0;
    for (let i = 0; i < b.length; i++) {
      if (b[i] === value) { count++; continue; }
      runs.push(value, count); value = b[i]; count = 1;
    }
    runs.push(value, count);
    out[key] = runs;
  }
  return out;
}"""

_CAPTURE_STATE = """() => {
  if (!game || !game.isRunning || !player || !chunkManager || !mobManager)
    throw new Error('Minecraft not running');
  for (const id of ['main-menu', 'pause-menu', 'inventory-screen']) {
    const el = document.getElementById(id);
    if (el && el.offsetParent !== null && getComputedStyle(el).display !== 'none')
      throw new Error('Minecraft UI open during checkpoint: ' + id);
  }
  const v3 = (v) => ({x: v.x, y: v.y, z: v.z});
  const clone = (x) => JSON.parse(JSON.stringify(x));
  return {
    world_seed: worldSeed,
    player: {
      position: v3(player.position), velocity: v3(player.velocity),
      pitch: player.pitch, yaw: player.yaw,
      isGrounded: player.isGrounded, isSprinting: player.isSprinting,
      isInWater: player.isInWater, isUnderwater: player.isUnderwater,
      health: player.health, maxHealth: player.maxHealth,
      hunger: player.hunger, maxHunger: player.maxHunger,
      keys: clone(player.keys), isLocked: player.isLocked,
      footstepTimer: player.footstepTimer, lastFootstepBlock: player.lastFootstepBlock,
      handSwingTime: player.handSwingTime, isSwinging: player.isSwinging
    },
    inventory: clone({hotbar: inventory.hotbar, slots: inventory.slots,
                      crafting: inventory.crafting, craftingResult: inventory.craftingResult,
                      selectedSlot: inventory.selectedSlot}),
    dragged: clone({item: draggedItem, from: draggedFrom}),
    mining: {breakProgress: breakProgress, breakingBlock: breakingBlock,
             isLeftMouseDown: isLeftMouseDown, isRightMouseDown: isRightMouseDown,
             leftMouseDidMine: leftMouseDidMine,
             leftMouseDownAgeMs: isLeftMouseDown ? Date.now() - leftMouseDownStartedAtMs : null},
    lighting: {time: lightingState.time, dayCount: lightingState.dayCount},
    mobs: mobManager.mobs.map((m) => ({
      type: m.type, position: v3(m.position), velocity: v3(m.velocity),
      rotation: m.rotation, health: m.health, maxHealth: m.maxHealth,
      isGrounded: m.isGrounded, targetPlayer: m.targetPlayer,
      wanderAngle: m.wanderAngle, wanderTimer: m.wanderTimer,
      attackCooldown: m.attackCooldown, animationTime: m.animationTime, isDead: m.isDead})),
    mob_spawn_cooldown: mobManager.spawnCooldown,
    modifications: Array.from(chunkManager.modifications.entries()),
    runtime: window.__gwCheckpointRuntime.capture(),
    rng: window.__gwGameRandomState.capture(),
    api: window.gameAPI.getState()
  };
}"""

_RESTORE = """(saved) => {
  const s = saved.state;
  if (!game || !game.isRunning || worldSeed !== s.world_seed)
    throw new Error('Minecraft page not running with the checkpoint world seed');
  // 1. Exact chunk set and block contents.
  for (const [key, chunk] of Array.from(chunkManager.chunks)) {
    if (saved.chunks[key]) continue;
    for (const mesh of [chunk.mesh, chunk.waterMesh, chunk.spriteMesh]) {
      if (mesh) { chunkManager.scene.remove(mesh); mesh.geometry.dispose(); }
    }
    chunkManager.chunks.delete(key);
  }
  for (const [key, info] of Object.entries(saved.chunks)) {
    const [cx, cz] = key.split(',').map(Number);
    const chunk = chunkManager.createChunk(cx, cz);
    const runs = saved.blobs[info.sha256];
    let i = 0;
    for (let r = 0; r < runs.length; r += 2) {
      chunk.blocks.fill(runs[r], i, i + runs[r + 1]);
      i += runs[r + 1];
    }
    if (i !== chunk.blocks.length) throw new Error('Chunk blob length mismatch: ' + key);
    chunk.isGenerated = info.generated;
    chunk.needsUpdate = true;
  }
  chunkManager.modifications = new Map(s.modifications);
  // 2. Player, inventory, mining.
  const p = s.player;
  player.position.set(p.position.x, p.position.y, p.position.z);
  player.velocity.set(p.velocity.x, p.velocity.y, p.velocity.z);
  for (const k of ['pitch', 'yaw', 'isGrounded', 'isSprinting', 'isInWater', 'isUnderwater',
                   'health', 'maxHealth', 'hunger', 'maxHunger', 'isLocked', 'footstepTimer',
                   'lastFootstepBlock', 'handSwingTime', 'isSwinging']) player[k] = p[k];
  Object.assign(player.keys, p.keys);
  inventory.hotbar = s.inventory.hotbar; inventory.slots = s.inventory.slots;
  inventory.crafting = s.inventory.crafting; inventory.craftingResult = s.inventory.craftingResult;
  inventory.selectedSlot = s.inventory.selectedSlot;
  draggedItem = s.dragged.item; draggedFrom = s.dragged.from;
  updateHotbarDisplay();
  breakProgress = s.mining.breakProgress; breakingBlock = s.mining.breakingBlock;
  isLeftMouseDown = s.mining.isLeftMouseDown; isRightMouseDown = s.mining.isRightMouseDown;
  leftMouseDidMine = s.mining.leftMouseDidMine;
  if (s.mining.leftMouseDownAgeMs !== null) leftMouseDownStartedAtMs = Date.now() - s.mining.leftMouseDownAgeMs;
  // 3. Time of day.
  lightingState.time = s.lighting.time; lightingState.dayCount = s.lighting.dayCount;
  // 4. Mobs (constructors draw from the RNG; the RNG is restored last).
  mobManager.clear();
  for (const m of s.mobs) {
    const mob = new Mob(m.type, m.position.x, m.position.y, m.position.z);
    for (const k of ['rotation', 'health', 'maxHealth', 'isGrounded', 'targetPlayer', 'wanderAngle',
                     'wanderTimer', 'attackCooldown', 'animationTime', 'isDead']) mob[k] = m[k];
    mob.velocity.set(m.velocity.x, m.velocity.y, m.velocity.z);
    mobManager.mobs.push(mob);
    mobManager.scene.add(mob.mesh);
  }
  mobManager.spawnCooldown = s.mob_spawn_cooldown;
  // 5. Meshes for the restored blocks, camera, then evaluator and RNG.
  for (const chunk of chunkManager.chunks.values()) chunkManager.buildChunkMesh(chunk);
  player.updateCamera();
  window.__gwCheckpointRuntime.restore(s.runtime);
  window.__gwGameRandomState.restore(s.rng);
  game.render();
  return true;
}"""


def _encode(value) -> bytes:
    return json.dumps(value, separators=(',', ':')).encode()


async def capture(env, blob_dir: Path) -> dict:
    """Snapshot the paused page. New chunk contents are written to blob_dir."""
    page = env.game_manager.page
    state = await page.evaluate(_CAPTURE_STATE)
    hashes = await page.evaluate(_HASH_CHUNKS)
    blob_dir.mkdir(parents=True, exist_ok=True)
    index_path = blob_dir / 'index.json'
    index = json.loads(index_path.read_bytes()) if index_path.is_file() else {}
    missing = sorted({key for key, value in hashes.items() if value.split(':')[0] not in index})
    if missing:
        exported = await page.evaluate(_EXPORT_CHUNKS, missing)
        for key in missing:
            data = _encode(exported[key])
            digest = hashlib.sha256(data).hexdigest()
            path = blob_dir / f'{digest}.json'
            if not path.exists():
                _state.atomic_write(path, data)
            index[hashes[key].split(':')[0]] = digest
        _state.atomic_write(index_path, _encode(index))
    chunks = {key: {'fnv': value.split(':')[0], 'sha256': index[value.split(':')[0]],
                    'generated': value.split(':')[1] == '1'} for key, value in sorted(hashes.items())}
    # The modification map also lists every generated tree block (~1 MB) and
    # changes only when a block is mined or placed: store it once by content.
    data = _encode(state.pop('modifications'))
    state['modifications_sha256'] = hashlib.sha256(data).hexdigest()
    path = blob_dir / f"{state['modifications_sha256']}.json"
    if not path.exists():
        _state.atomic_write(path, data)
    world_digest = hashlib.sha256(_encode({'chunks': {k: v['sha256'] for k, v in chunks.items()},
                                           'modifications': state['modifications_sha256']})).hexdigest()
    return {'version': 1, 'game': GAME_ID, 'seed': state['world_seed'], 'state': state,
            'chunks': chunks, 'maze_sha256': world_digest, 'level': None, 'api': state['api']}


def _comparable(world: dict) -> dict:
    s = world['state']
    return {'chunks': {k: v['sha256'] for k, v in world['chunks'].items()},
            'modifications': s['modifications_sha256'], 'player': s['player'],
            'inventory': s['inventory'], 'mining': {k: v for k, v in s['mining'].items()
                                                    if k != 'leftMouseDownAgeMs'},
            'lighting': s['lighting'], 'mobs': s['mobs'], 'cooldown': s['mob_spawn_cooldown'],
            'rng': s['rng'], 'baseline': s['runtime'].get('baselineItemCounts'),
            'terminal': {k: v for k, v in s['runtime']['terminalLatch'].items() if k != 'ageMs'},
            'item_gains': ((s['api'].get('game_state') or {}).get('inventory') or {}).get('item_gains')}


async def restore(env, saved: dict, blob_dir: Path, *, leave_paused: bool = False) -> dict:
    """Load a checkpoint into the freshly started page, then verify it exactly."""
    await env.pause_game()
    try:
        blobs = {}
        for info in saved['chunks'].values():
            if info['sha256'] not in blobs:
                data = (blob_dir / f"{info['sha256']}.json").read_bytes()
                if hashlib.sha256(data).hexdigest() != info['sha256']:
                    raise _state.UnsafeResume('Chunk blob changed on disk')
                blobs[info['sha256']] = json.loads(data)
        if 'modifications' in saved['state']:  # early checkpoints stored the map inline
            mods = saved['state']['modifications']
            saved = {**saved, 'state': {**saved['state'],
                     'modifications_sha256': hashlib.sha256(_encode(mods)).hexdigest()}}
        else:
            data = (blob_dir / f"{saved['state']['modifications_sha256']}.json").read_bytes()
            if hashlib.sha256(data).hexdigest() != saved['state']['modifications_sha256']:
                raise _state.UnsafeResume('Modification blob changed on disk')
            mods = json.loads(data)
        state = {**saved['state'], 'modifications': mods}
        await env.game_manager.page.evaluate(_RESTORE, {**saved, 'state': state, 'blobs': blobs})
        actual = await capture(env, blob_dir)
        expected, got = _comparable(saved), _comparable(actual)
        for key in expected:
            if expected[key] != got[key]:
                raise _state.UnsafeResume(f'Minecraft {key} differs after restore')
        return {'map_sha256': actual['maze_sha256'], 'position': actual['state']['player']['position']}
    finally:
        if not leave_paused:
            await env.resume_game()
