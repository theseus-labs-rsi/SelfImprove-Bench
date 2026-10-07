"""Minecraft GameWorld recovery research, never a paid-run restore adapter.

Stage a *separate* instrumented copy of game_api.js. The upstream game files
and game rules remain untouched. The added bridge exposes the existing private
score baseline/clock so browser restart can be tested without model calls.
This is only one of the prerequisites for genuine step-level continuation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import importlib.util as _ilu
import sys as _sys

# Load the sibling module by path: GameWorld ships its own `tools` package.
_name = '_csd_astray_step_state'  # same instance as benches.gameworld.astray_step_resume
if _name not in _sys.modules:
    _spec = _ilu.spec_from_file_location(_name, Path(__file__).with_name('gameworld_step_state.py'))
    _module = _ilu.module_from_spec(_spec)
    _sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
UnsafeResume, _sync_dir = _sys.modules[_name].UnsafeResume, _sys.modules[_name]._sync_dir


# Inject at the end of the existing IIFE so that its private `runtime` object
# is accessible. Deliberately no reset/start/score code is changed.
_RUNTIME_BRIDGE = r'''
  // GameWorld recovery instrumentation: no game rules/score formulas changed.
  window.__gwCheckpointRuntime = Object.freeze({
    schema: 1,
    capture: function () {
      var now = Date.now();
      return {
        schema: 1,
        resetCount: runtime.resetCount,
        elapsedGameplayMs: runtime.gameplayStartMs === null
          ? null : now - runtime.gameplayStartMs,
        baselineItemCounts: runtime.baselineItemCounts === null
          ? null : cloneDict(runtime.baselineItemCounts),
        lastResetMethod: runtime.lastResetMethod,
        session: {
          seed: session.seed,
          requestedLevel: session.requestedLevel,
          requestedDifficulty: session.requestedDifficulty,
          elapsedEpisodeMs: now - session.episodeStartMs,
          episodeCount: session.episodeCount
        },
        terminalLatch: {
          isTerminal: terminalLatch.isTerminal,
          outcome: terminalLatch.outcome,
          reason: terminalLatch.reason,
          ageMs: now - terminalLatch.ts
        }
      };
    },
    restore: function (s) {
      // Refuse malformed or incomplete inputs before changing private state.
      if (!s || s.schema !== 1 || !Number.isSafeInteger(s.resetCount)
          || s.resetCount < 0 || !s.session || !s.terminalLatch
          || !Number.isFinite(s.session.elapsedEpisodeMs)
          || !Number.isSafeInteger(s.session.episodeCount)
          || !Number.isFinite(s.terminalLatch.ageMs)
          || (s.elapsedGameplayMs !== null
              && (!Number.isFinite(s.elapsedGameplayMs) || s.elapsedGameplayMs < 0))
          || (s.baselineItemCounts !== null
              && (!s.baselineItemCounts || Array.isArray(s.baselineItemCounts)
                  || typeof s.baselineItemCounts !== 'object'))
          || !game || !game.isRunning) {
        throw new Error('Unsafe/incomplete GameWorld runtime recovery record');
      }
      var now = Date.now();
      runtime.resetCount = s.resetCount;
      runtime.gameplayStartMs = s.elapsedGameplayMs === null
        ? null : now - s.elapsedGameplayMs;
      runtime.baselineItemCounts = s.baselineItemCounts === null
        ? null : cloneDict(s.baselineItemCounts);
      runtime.lastResetMethod = s.lastResetMethod;
      session.seed = s.session.seed;
      session.requestedLevel = s.session.requestedLevel;
      session.requestedDifficulty = s.session.requestedDifficulty;
      session.episodeStartMs = now - s.session.elapsedEpisodeMs;
      session.episodeCount = s.session.episodeCount;
      terminalLatch.isTerminal = s.terminalLatch.isTerminal;
      terminalLatch.outcome = s.terminalLatch.outcome;
      terminalLatch.reason = s.terminalLatch.reason;
      terminalLatch.ts = now - s.terminalLatch.ageMs;
    }
  });
'''


def instrument_api(original: bytes) -> bytes:
    """Refuse changed upstream/duplicate instrumentation, never edit in place."""
    marker = b'\n})();'
    if (original.count(marker) != 1 or b'__gwCheckpointRuntime' in original
            or b'window.gameAPI = {' not in original
            or b'baselineItemCounts: null' not in original):
        raise UnsafeResume('Upstream game_api.js differs; manual source audit required')
    return original.replace(marker, _RUNTIME_BRIDGE.encode('utf-8') + marker)


def stage_api(original: Path, destination: Path) -> dict:
    """Create a standalone file to use ONLY in a throwaway no-model overlay."""
    if destination.exists() or original.resolve() == destination.resolve():
        raise UnsafeResume('Refusing to overwrite upstream or existing recovery source')
    source = original.read_bytes()
    generated = instrument_api(source)
    # Exclusive creation: atomic_write normally replaces existing files; we
    # fail closed against prior staging even if another process raced us.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as output:
        output.write(generated)
        output.flush()
        import os
        os.fsync(output.fileno())
    _sync_dir(destination.parent)
    return {'upstream_sha256': hashlib.sha256(source).hexdigest(),
            'instrumented_sha256': hashlib.sha256(generated).hexdigest(),
            'instrumented_path': str(destination)}


def instrument_random_source(original: str) -> str:
    """Test-only bridge for the upstream seeded PRNG's *current* stream position.

    Keep the mulberry32 algorithm unchanged. Refuse an unfamiliar upstream RNG
    rather than mistakenly resuming with a fresh seed at step 98.
    """
    changes = (
        ('        return function() {\n            seed |= 0;',
         '        const next = function() {\n            seed |= 0;'),
        ('            return ((t ^ t >>> 14) >>> 0) / 4294967296;\n        };',
         '            return ((t ^ t >>> 14) >>> 0) / 4294967296;\n'
         '        };\n'
         '        next.__gwState = function() { return seed | 0; };\n'
         '        next.__gwRestore = function(value) { seed = value | 0; };\n'
         '        return next;'),
        ('    const seededRandom = mulberry32(SEED);',
         '    let seededRandom = mulberry32(SEED);'),
        ('    window.__resetRandomSeed = function() {\n        Math.random = mulberry32(SEED);',
         '    window.__resetRandomSeed = function() {\n'
         '        seededRandom = mulberry32(SEED);\n'
         '        Math.random = seededRandom;'),
        ('    window.__restoreOriginalRandom = function() {',
         '    window.__gwRandomState = Object.freeze({\n'
         '        capture: function() {\n'
         '            if (Math.random !== seededRandom) throw new Error("RNG stream replaced");\n'
         '            return seededRandom.__gwState();\n'
         '        },\n'
         '        restore: function(value) {\n'
         '            if (!Number.isSafeInteger(value) || value < -2147483648\n'
         '                || value > 2147483647 || Math.random !== seededRandom)\n'
         '                throw new Error("Invalid or replaced RNG stream");\n'
         '            seededRandom.__gwRestore(value);\n'
         '        }\n'
         '    });\n'
         '    window.__restoreOriginalRandom = function() {'),
    )
    if '__gwRandomState' in original:
        raise UnsafeResume('RNG already instrumented')
    text = original
    for old, new in changes:
        if text.count(old) != 1:
            raise UnsafeResume('Upstream deterministic_random.js changed; cannot restore RNG safely')
        text = text.replace(old, new)
    return text


def instrument_index_random(original: bytes) -> bytes:
    """Expose existing Minecraft index.html PRNG + crypto stream positions.

    The GitHub game seeds and replaces Math.random inside its index.html. The
    browser-level deterministic_random.js is overwritten on page load, so its
    RNG is *not* the game's RNG. Stage a temporary HTML copy only.
    """
    text = original.decode('utf-8')
    if '__gwGameRandomState' in text or text.count('<!-- [DETERMINISTIC_RANDOM] -->') != 2:
        raise UnsafeResume('Unrecognized or instrumented Minecraft index.html')
    changes = (
        ('        return function() {\n            s |= 0;',
         '        var next = function() {\n            s |= 0;'),
        ('            return ((t ^ t >>> 14) >>> 0) / 4294967296;\n        };',
         '            return ((t ^ t >>> 14) >>> 0) / 4294967296;\n'
         '        };\n'
         '        next.__gwState = function() { return s | 0; };\n'
         '        next.__gwRestore = function(value) { s = value | 0; };\n'
         '        return next;'),
        ('    var originalCrypto = window.crypto',
         '    var rngWrapper = Math.random;\n    var originalCrypto = window.crypto'),
        ('        Math.random = function() { return rng(); };\n    };',
         '        Math.random = function() { return rng(); };\n'
         '        rngWrapper = Math.random;\n    };'),
        ("    console.log('[DeterministicRandom] Seeded with:', SEED);", '''    window.__gwGameRandomState = Object.freeze({
        capture: function() {
            if (Math.random !== rngWrapper) throw new Error('Game RNG replaced');
            return {random: rng.__gwState(), crypto: rngCrypto.__gwState()};
        },
        restore: function(state) {
            if (!state || Math.random !== rngWrapper) throw new Error('Game RNG replaced');
            for (var value of [state.random, state.crypto]) {
                if (!Number.isSafeInteger(value) || value < -2147483648 || value > 2147483647)
                    throw new Error('Invalid Game RNG checkpoint');
            }
            rng.__gwRestore(state.random);
            rngCrypto.__gwRestore(state.crypto);
        }
    });
    console.log('[DeterministicRandom] Seeded with:', SEED);'''),
    )
    for old, new in changes:
        if text.count(old) != 1:
            raise UnsafeResume('GitHub Minecraft PRNG has changed; manual audit required')
        text = text.replace(old, new)
    return text.encode('utf-8')
