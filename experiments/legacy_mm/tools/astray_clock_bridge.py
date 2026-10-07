"""Fail-closed Astray gameAPI bridge for step checkpoints.

Never changes the GameWorld checkout: a patched copy of game_api.js is served to
the browser for the episode. The bridge exposes the evaluator's private session
state (start level, corridor start distance, completion flag, episode clock) so
a restored page reports the same progress as the interrupted one. Game rules
and score formulas are unchanged.
"""
from __future__ import annotations

import hashlib


_KNOWN_SOURCE_DIGESTS = frozenset({
    # Corridor-progress game_api.js (CHANGES.md item 14): training overlay with
    # ?train_seed= maps, and the official holdout copy.
    "3a0a0ecfaa2fc152c5dfc4ec4940fbe79ddea4d0570ae4b06c9a38942fe62c80",
    "25c96e48fb859e2fa82a0ce71e835ce13fb8466f249ea7ca16ac90d5d8ce585e",
})

_BRIDGE = r'''
  window.__gwAstrayCheckpointRuntime = Object.freeze({
    capture: function() {
      if (!window.gameAPI || window.gameAPI.getState().status !== 'playing')
        throw new Error('Astray not actionable for runtime checkpoint');
      const elapsedMs = Date.now() - session.episodeStartMs;
      if (!Number.isFinite(elapsedMs) || elapsedMs < 0)
        throw new Error('Invalid Astray evaluator clock');
      return {
        schema: 2, seed: session.seed, episodeCount: session.episodeCount,
        requestedLevel: session.requestedLevel,
        requestedDifficulty: session.requestedDifficulty,
        progressLevel: session.progressLevel,
        maxDistanceToGoal: session.maxDistanceToGoal,
        startLevel: session.startLevel,
        startPathDistance: session.startPathDistance,
        completed: session.completed,
        lastResetMethod: runtimeState.lastResetMethod,
        elapsedMs: elapsedMs
      };
    },
    restore: function(saved) {
      if (!saved || saved.schema !== 2 || !window.gameAPI ||
          window.gameAPI.getState().status !== 'playing' ||
          saved.seed !== session.seed ||
          saved.episodeCount !== session.episodeCount ||
          saved.requestedLevel !== session.requestedLevel ||
          saved.requestedDifficulty !== session.requestedDifficulty ||
          !Number.isFinite(saved.elapsedMs) || saved.elapsedMs < 0 ||
          saved.elapsedMs > 86400000 ||
          (saved.startLevel !== null && !Number.isInteger(saved.startLevel)) ||
          (saved.startPathDistance !== null &&
           (!Number.isFinite(saved.startPathDistance) || saved.startPathDistance < 0)) ||
          typeof saved.completed !== 'boolean' ||
          saved.lastResetMethod !== runtimeState.lastResetMethod)
        throw new Error('Incompatible Astray evaluator checkpoint');
      // The next getState() recomputes progress from the stored start distance.
      session.progressLevel = saved.progressLevel;
      session.maxDistanceToGoal = saved.maxDistanceToGoal;
      session.startLevel = saved.startLevel;
      session.startPathDistance = saved.startPathDistance;
      session.completed = saved.completed;
      session.episodeStartMs = Date.now() - saved.elapsedMs;
    }
  });
'''


def instrument_astray_api(source: bytes) -> bytes:
    """Inject into a *copy* of the exact expected closure, or refuse."""
    ending = b'\n})();\n'
    guards = (
        b"const GAME_ID = '03_astray';",
        b'episodeStartMs: Date.now(),',
        b'session.episodeStartMs = Date.now();',
        b'window.gameAPI = {',
        b'startPathDistance: null,',
        b'completed: false',
    )
    if (hashlib.sha256(source).hexdigest() not in _KNOWN_SOURCE_DIGESTS
            or source.count(ending) != 1
            or any(source.count(guard) != 1 for guard in guards)
            or b'__gwAstrayCheckpointRuntime' in source):
        raise RuntimeError('Astray game_api.js changed; refusing unsafe clock instrumentation')
    return source.replace(ending, _BRIDGE.encode() + ending)
