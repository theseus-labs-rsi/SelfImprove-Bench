"""Astray browser/agent checkpoint utilities; does not modify official GameWorld.

Supports compatible Astray pages with the original or parameterized map seed.
Browser-internal RNG state is not captured: future mazes after level transition
may diverge from an uninterrupted run, although level-1 layouts are deterministic.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path

# A maze is an array with an extra .dimension property (JSON.stringify(maze)
# does not include that property). Store both a digest and the actual walls.
CAPTURE = """() => {
  if (!window.maze || !window.wBall || !window.ballMesh || !window.camera || !window.light ||
      window.gameState !== 'play') throw Error('Astray not ready/play');
  const p = wBall.GetPosition(), v = wBall.GetLinearVelocity();
  return {maze: Array.from(maze, row => Array.from(row, cell => !!cell)),
    dimension: maze.dimension, state: gameState, key_axis: Array.from(keyAxis),
    body: {x:p.x,y:p.y,vx:v.x,vy:v.y,angle:wBall.GetAngle(), angular_velocity:wBall.GetAngularVelocity()},
    mesh: {x:ballMesh.position.x,y:ballMesh.position.y,z:ballMesh.position.z},
    camera: {x:camera.position.x,y:camera.position.y,z:camera.position.z},
    light: {x:light.position.x,y:light.position.y,z:light.position.z,intensity:light.intensity},
    mesh_matrix: Array.from(ballMesh.matrix.elements),
    rotation: {x:ballMesh.rotation.x,y:ballMesh.rotation.y,z:ballMesh.rotation.z},
    api: window.gameAPI.getState(),
    evaluator_runtime: window.__gwAstrayCheckpointRuntime
      ? window.__gwAstrayCheckpointRuntime.capture() : null};
}"""

RESTORE = """(saved) => {
  if (gameState !== 'play' || !wBall || !ballMesh || !maze || !camera)
    throw Error('Astray not actionable');
  if (saved.dimension !== maze.dimension) throw Error('maze dimension mismatch');
  if (saved.maze !== null) {
    const current = Array.from(maze, row => Array.from(row, c => !!c));
    if (JSON.stringify(current) !== JSON.stringify(saved.maze)) throw Error('maze walls mismatch');
  }
  const b = saved.body;
  wBall.SetPosition(new b2Vec2(b.x, b.y));
  wBall.SetLinearVelocity(new b2Vec2(b.vx, b.vy));
  wBall.SetAngle(b.angle || 0);
  wBall.SetAngularVelocity(b.angular_velocity || 0);
  wBall.SetAwake(true);
  ballMesh.position.set(saved.mesh.x, saved.mesh.y, saved.mesh.z);
  if (saved.rotation) ballMesh.rotation.set(saved.rotation.x, saved.rotation.y, saved.rotation.z);
  if (saved.camera) camera.position.set(saved.camera.x,saved.camera.y,saved.camera.z);
  else camera.position.set(saved.mesh.x,saved.mesh.y,5);
  if (saved.light) { light.position.set(saved.light.x, saved.light.y, saved.light.z);
    light.intensity = saved.light.intensity; }
  keyAxis = saved.key_axis.slice();
  if (saved.evaluator_runtime) {
    if (!window.__gwAstrayCheckpointRuntime) throw Error('Missing evaluator clock bridge');
    window.__gwAstrayCheckpointRuntime.restore(saved.evaluator_runtime);
  }
  if (window.renderer && window.scene) renderer.render(scene,camera);
  return window.gameAPI.getState();
}"""


def maze_hash(maze):
    return hashlib.sha256(json.dumps(maze, separators=(',', ':')).encode()).hexdigest()


def from_legacy_checkpoint(checkpoint: dict) -> dict:
    state = checkpoint['game_state']
    if state['gameId'] != '03_astray' or state['status'] != 'playing':
        raise ValueError('Legacy checkpoint is not a live Astray play state')
    player = state['game_state']['player']
    # Legacy checkpoint omitted Box2D angle, maze layout, camera, and RNG state.
    return {'version': 1, 'step': checkpoint['interaction_id'], 'seed': state['seed'],
        'level': state['game_state']['level'], 'dimension': state['debug']['maze_dimension'],
        'maze': None, 'maze_sha256': None, 'state': 'play',
        'key_axis': state['debug']['key_axis'],
        'body': {k: player[k] for k in ('x','y','vx','vy')},
        'mesh': {k: player[k] for k in ('x','y','z')},
        'camera': None, 'rotation': None,
        'api': state, 'metrics': checkpoint['task_evaluation']['metrics']}


async def capture(env):
    page = env.game_manager.page
    snapshot = await page.evaluate(CAPTURE)
    snapshot['maze_sha256'] = maze_hash(snapshot['maze'])
    snapshot['seed'] = snapshot['api']['seed']
    # Version-2 browser snapshots need the same level guard as legacy sources.
    # Without this, restore() raises KeyError('level') after restarting the page.
    snapshot['level'] = snapshot['api']['game_state']['level']
    return snapshot


async def restore(env, saved, *, leave_paused=False):
    await env.pause_game()
    try:
        before = await capture(env)
        if saved['seed'] != before['seed'] or saved['dimension'] != before['dimension']:
            raise RuntimeError('Seed or maze dimension changed; refusing resume')
        if saved.get('maze_sha256') and saved['maze_sha256'] != before['maze_sha256']:
            raise RuntimeError('Maze hash changed; refusing resume')
        # The corridor-progress denominator is fixed at the first observation of
        # the episode; it must be identical before the evaluator state is restored.
        old_distance = saved['api']['debug'].get('start_path_distance')
        new_distance = before['api']['debug'].get('start_path_distance')
        if old_distance != new_distance:
            raise RuntimeError(f'Progress denominator mismatch: {new_distance} != {old_distance}')
        await env.game_manager.page.evaluate(RESTORE, saved)
        actual = await capture(env)
        for key in ('x','y','vx','vy'):
            if abs(actual['body'][key] - saved['body'][key]) > 1e-4:
                raise RuntimeError(f'Physical {key} mismatch after restore')
        for key in ('x','y'):
            if abs(actual['api']['game_state']['player'][key] - saved['mesh'][key]) > 1e-4:
                raise RuntimeError(f'API player {key} mismatch after restore')
        if actual['api']['game_state']['level'] != saved['level']:
            raise RuntimeError('Level mismatch after restore')
        return {'map_sha256': before['maze_sha256'], 'map_verified_against_source': bool(saved.get('maze_sha256')),
                'position': actual['body'], 'api_progress': actual['api']['game_state']['completion_progress'],
                'legacy_map_unknown': saved.get('maze_sha256') is None}
    finally:
        if not leave_paused:
            await env.resume_game()


def memory_from_interactions(client, run_dir: Path, step: int):
    """Reconstruct last N model rounds from GameWorld's own interaction log."""
    from agents.harness.memory import record_memory_round
    store = client.memory_store
    if store is None:
        return 0
    records = [json.loads(line) for line in (run_dir/'agent_0/interactions.jsonl').read_text().splitlines() if line.strip()]
    records = [r for r in records if int(r['interaction_id']) <= step]
    if not records or int(records[-1]['interaction_id']) != step:
        raise RuntimeError('Source interactions do not end at checkpoint step')
    last = records[-store.capacity:]
    for record in last:
        output = record['output']
        if output.get('error') or not isinstance(output.get('parsed_action'), dict):
            raise RuntimeError('Source interaction is incomplete')
        path = run_dir/'agent_0'/record['input']['screenshot']
        if not path.is_file():
            raise RuntimeError(f'Source model-memory screenshot missing: {path}')
        record_memory_round(store, user_prompt='Game screen:\n', screenshot_path=path,
            action=json.dumps(output['parsed_action'], ensure_ascii=False, sort_keys=True),
            reasoning=output.get('diagnostic_reasoning') or
            (output['parsed_action'].get('arguments') or {}).get('reasoning'))
    return len(last)


def snapshot_memory(client):
    if client.memory_store is None:
        return []
    return [[asdict(entry) for entry in round_] for round_ in client.memory_store._memory_rounds]


def restore_memory(client, rounds):
    from agents.harness.memory import MemoryEntry
    if client.memory_store is None:
        if rounds:
            raise RuntimeError('Source memory provided but model memory is disabled')
        return
    for round_ in rounds:
        entries = [MemoryEntry(**entry) for entry in round_]
        for entry in entries:
            if entry.type == 'image' and (not entry.image_path or not Path(entry.image_path).is_file()):
                raise RuntimeError('Stored memory screenshot unavailable')
        client.memory_store.add_memory_round(entries)


def durable_memory(client, directory: Path):
    """Copy the rolling screenshots out of the browser's temporary directory."""
    rounds = snapshot_memory(client)
    directory.mkdir(parents=True, exist_ok=True)
    copied = {}
    for round_ in rounds:
        for entry in round_:
            if entry['type'] != 'image':
                continue
            source = Path(entry['image_path'])
            if not source.is_file():
                raise RuntimeError(f'Memory screenshot disappeared: {source}')
            if str(source) not in copied:
                dest = directory / f'memory_{len(copied):03d}.png'
                shutil.copy2(source, dest)
                copied[str(source)] = str(dest.resolve())
            entry['image_path'] = copied[str(source)]
    return rounds
