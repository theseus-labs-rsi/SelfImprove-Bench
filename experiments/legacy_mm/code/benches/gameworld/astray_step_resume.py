"""Opt-in, fail-closed GameWorld step transactions for the *real* Coordinator.

Supports the single-agent 03_astray and 18_minecraft-clone-glm benchmarks. Each
committed step stores the browser world, evaluator, agent context and the paid
model response; a restarted episode restores the last step, verifies it against
the saved state, and continues. A receipted model response is never requested
again. Real-time physics between frames is not bitwise deterministic, so the
continuation matches an uninterrupted run only up to that same jitter.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

# Step-resume helpers live in this version's top-level tools/ (not GameWorld's tools package).
_PROJECT = Path(__file__).resolve().parents[3]


def _local_module(name: str, file: Path):
    # GameWorld has its own `tools` package, which must NOT be shadowed.
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


_state = _local_module('_csd_astray_step_state', _PROJECT / 'tools/gameworld_step_state.py')
_world = _local_module('_csd_astray_browser_checkpoint', _PROJECT / 'tools/astray_step_checkpoint.py')
_mc = _local_module('_csd_minecraft_step_checkpoint', _PROJECT / 'tools/minecraft_step_checkpoint.py')
ASTRAY, MINECRAFT = '03_astray', '18_minecraft-clone-glm'
SUPPORTED_GAMES = (ASTRAY, MINECRAFT)
StepStore, UnsafeResume = _state.StepStore, _state.UnsafeResume


def _json(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_DIAG_FIELDS = ('_recent_action_ids', '_diagnostic_recovery_ids', '_diagnosis_pending',
                '_active_recovery_remaining', '_active_blocked_ids', '_latest_state_by_agent',
                '_mc_guard_counts', '_mc_mining_streaks')


class AstrayTransaction:
    def __init__(self, *, run_root: Path, task: dict, gameworld_root: Path, framework: str):
        self.game = task['game_id']
        if self.game not in SUPPORTED_GAMES or framework not in ('ace', 'hyperagents', 'ahe'):
            raise UnsafeResume('Step recovery supports three-framework Astray and Minecraft only')
        yaml = gameworld_root / 'code/catalog/tasks' / self.game / f"{task['game_task_id']}.yaml"
        game_dir = gameworld_root / 'code/games/benchmark' / self.game
        game_api = game_dir / 'game_api.js'
        if not yaml.is_file() or not game_api.is_file():
            raise UnsafeResume('Game or task files missing; cannot bind recovery identity')
        # Execution-only switches must not change the recovery identity.
        identity = {k: v for k, v in task.items() if k not in ('resumable_steps',)}
        source = {'task': identity, 'task_sha256': _digest(yaml),
                  'api_sha256': _digest(game_api), 'framework': framework}
        if self.game == MINECRAFT:
            source['index_sha256'] = _digest(game_dir / 'index.html')
        source_digest = hashlib.sha256(_state.encode(source)).hexdigest()
        self.root = run_root / 'astray_recovery'  # historical name, used for both games
        # Content-addressed Minecraft chunk blobs; kept outside the step store root.
        self.blob_dir = run_root / 'step_chunks'
        self.store = StepStore(self.root, run_root.name, source_digest,
                               integrity_audit_interval=task.get("step_integrity_audit_interval", 1))
        self.framework = framework
        if self.game == ASTRAY:
            bridge = _local_module('_csd_astray_clock_bridge', _PROJECT / "tools/astray_clock_bridge.py")
            self._routes = {'game_api.js': bridge.instrument_astray_api(game_api.read_bytes())}
        else:
            self._routes = _mc.instrument(game_dir)
        self.started_at = time.monotonic()
        self._latest_receipt = None

    def _asset(self, path: Path, dest: Path) -> dict:
        if not path.is_file():
            raise UnsafeResume(f'Checkpoint screenshot missing: {path}')
        # Content-addressed: a screenshot reused as memory in later steps is stored once.
        dest = dest.parent / f'{_digest(path)}{path.suffix}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _digest(dest) != _digest(path):
                raise UnsafeResume('Conflicting immutable screenshot evidence')
        else:
            # Same filesystem: copy to private temp, fsync/rename via atomic_write.
            _state.atomic_write(dest, path.read_bytes())
        return {'path': str(dest.resolve()), 'sha256': _digest(dest)}

    def _snapshot_client(self, client, number: int) -> dict:
        memory = _world.snapshot_memory(client)
        for row_idx, row in enumerate(memory):
            for col_idx, entry in enumerate(row):
                if entry['type'] != 'image':
                    continue
                evidence = self._asset(Path(entry['image_path']),
                    self.root / 'images' / f'{number:04d}_memory_{row_idx}_{col_idx}.png')
                entry['image_path'] = evidence['path']
                entry['sha256'] = evidence['sha256']
        state = {'schema': 1, 'memory': memory,
                 'runtime_advisory': client._runtime_advisory,
                 'runtime_reasoning_required': client._runtime_reasoning_required,
                 'worker': None}
        if self.framework == 'ahe':
            state['worker'] = self._pack_worker(client._ahe_agent.snapshot_session())
        return state

    def _pack_worker(self, snapshot: dict) -> dict:
        """Store each NexAU message once (by content); the history grows every step."""
        packed = {k: v for k, v in snapshot.items() if k != 'history'}
        refs = []
        for message in snapshot['history']:
            data = _state.encode(message)
            sha = hashlib.sha256(data).hexdigest()
            path = self.blob_dir / 'messages' / f'{sha}.json'
            if not path.exists():
                _state.atomic_write(path, data)
            refs.append(sha)
        packed['history_refs'] = refs
        return packed

    def _unpack_worker(self, packed: dict) -> dict:
        if 'history' in packed:
            return packed
        history = []
        for sha in packed['history_refs']:
            data = (self.blob_dir / 'messages' / f'{sha}.json').read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise UnsafeResume('Stored AHE message changed on disk')
            history.append(json.loads(data))
        return {**{k: v for k, v in packed.items() if k != 'history_refs'}, 'history': history}

    def _restore_client(self, client, state: dict):
        if state.get('schema') != 1 or not isinstance(state.get('memory'), list):
            raise UnsafeResume('Incomplete framework model history')
        if self.framework == 'ahe':
            client._ahe_agent.restore_session(self._unpack_worker(state['worker']))
        memory = []
        for row in state['memory']:
            entries = []
            for item in row:
                entry = dict(item)
                sha = entry.pop('sha256', None)
                if entry['type'] == 'image' and (_digest(Path(entry['image_path'])) != sha):
                    raise UnsafeResume('Model memory image changed on disk')
                entries.append(entry)
            memory.append(entries)
        if client.memory_store is not None:
            client.memory_store._memory_rounds.clear()
        _world.restore_memory(client, memory)
        client._runtime_advisory = state['runtime_advisory']
        client._runtime_reasoning_required = state['runtime_reasoning_required']

    def _snapshot_agent(self, coordinator, agent, number: int):
        diagnostics = {name: _json(getattr(coordinator, name)) for name in _DIAG_FIELDS}
        diagnostics['_diagnosis_pending'] = sorted(coordinator._diagnosis_pending)
        return {'schema': 1, 'step_index': agent.step_index,
                'eval_metrics': _json(agent.eval_metrics),
                'diagnostics': diagnostics,
                'client': self._snapshot_client(agent.client, number)}

    def _restore_agent(self, coordinator, agent, state: dict):
        if state.get('schema') != 1 or not isinstance(state.get('eval_metrics'), dict):
            raise UnsafeResume('Evaluator metrics missing from checkpoint')
        self._restore_client(agent.client, state['client'])
        agent.step_index = state['step_index']
        agent.eval_metrics = state['eval_metrics']
        coordinator.env._frame_counter = agent.step_index
        for logger in coordinator.agent_loggers.values():
            logger.interaction_count = agent.step_index
        for name in _DIAG_FIELDS:
            value = state['diagnostics'][name]
            setattr(coordinator, name, set(value) if name == '_diagnosis_pending' else value)

    @staticmethod
    def _observation(world, agent):
        api = world['api']
        progress = (api['game_state']['completion_progress'] if world.get('game', ASTRAY) == ASTRAY
                    else float((api.get('metrics') or {}).get('primary_score') or 0))
        return {'schema': 1, 'map_sha256': world['maze_sha256'],
                'level': world['level'], 'seed': world['seed'],
                'game_progress': progress,
                'eval_metrics': _json(agent.eval_metrics),
                'step_index': agent.step_index,
                'game_time_ms': api.get('gameTimeMs')}

    async def _capture(self, env):
        if self.game == ASTRAY:
            return await _world.capture(env)
        return await _mc.capture(env, self.blob_dir)

    async def _restore(self, env, saved, *, leave_paused):
        if self.game == ASTRAY:
            return await _world.restore(env, saved, leave_paused=leave_paused)
        return await _mc.restore(env, saved, self.blob_dir, leave_paused=leave_paused)

    async def _save(self, coordinator, agent, *, already_paused=False):
        if not already_paused:
            await coordinator.env.pause_game()
        try:
            world = await self._capture(coordinator.env)
            if world['api']['gameId'] != self.game:
                raise UnsafeResume('Unexpected browser game during step recovery')
            return world
        finally:
            if not already_paused:
                await coordinator.env.resume_game()

    async def initialize(self, coordinator):
        agent, = coordinator.agents
        await coordinator.env.start()
        if not (self.root / 'genesis.json').exists():
            if self.root.exists() and any(p.name != '.episode.lock' for p in self.root.iterdir()):
                raise UnsafeResume('Incomplete genesis; do not infer or overwrite')
            world = await self._save(coordinator, agent)
            self.store.save_genesis(world=world,
                observed_state=self._observation(world, agent),
                agent_state=self._snapshot_agent(coordinator, agent, 0),
                ledger={'schema': 1, 'total_tokens': 0, 'wall_seconds': 0.0})
            return
        genesis = self.store.load_genesis()
        latest = self.store.recover_finished_commit()
        boundary = latest or genesis
        saved = self.store._read(self.store.steps / f'{latest["step"]:04d}.world.json') if latest else genesis['world']
        if saved.get('terminal'):
            raise UnsafeResume('Episode is already terminal; never restart it')
        # Keep the newly restored world frozen through all verification.  The
        # previous _save() unpaused the page between restore and score check.
        await self._restore(coordinator.env, saved, leave_paused=True)
        try:
            pending = self.store.pending_model_result()
            restore_state = boundary['agent_state']
            if pending and self.framework == 'ahe':
                restore_state = {**restore_state, 'client': pending['agent_after_model']['client']}
            self._restore_agent(coordinator, agent, restore_state)
            self._verify_or_quarantine_logs(coordinator, agent.step_index, pending is not None)
            observed = await self._save(coordinator, agent, already_paused=True)
            expected = boundary['observed_state']
            actual = self._observation(observed, agent)
            # Check the immutable saved browser coordinates, not a moving page.
            for key in ('map_sha256', 'level', 'seed', 'step_index', 'eval_metrics'):
                if actual[key] != expected[key]:
                    raise UnsafeResume(f'Restored {key} does not match step checkpoint')
            if abs(actual['game_progress'] - expected['game_progress']) > 1e-5:
                raise UnsafeResume(f'Restored official score differs from checkpoint: expected={expected["game_progress"]!r}, actual={actual["game_progress"]!r}')
            if (self.root / 'inflight.json').exists() and not self.store.pending_model_result():
                raise UnsafeResume('Unknown paid call: audit required, do NOT repeat the inference')
        finally:
            await coordinator.env.resume_game()

    def _verify_or_quarantine_logs(self, coordinator, completed: int, has_receipt: bool):
        logger = coordinator.agent_loggers[coordinator.agents[0].agent_id]
        self._verify_log_path(logger.interactions_path, completed, has_receipt)

    def _verify_log_path(self, path: Path, completed: int, has_receipt: bool):
        lines = path.read_bytes().splitlines() if path.is_file() else []
        if len(lines) < completed:
            raise UnsafeResume('GameWorld interactions missing committed history')
        try:
            records = [json.loads(line) for line in lines]
        except (ValueError, UnicodeDecodeError) as exc:
            raise UnsafeResume('Corrupted GameWorld interaction log') from exc
        if any(item.get('interaction_id') != index for index, item in enumerate(records[:completed], 1)):
            raise UnsafeResume('Logged steps differ from committed contiguous prefix')
        # A contiguous ID alone is insufficient: reject damaged screenshot
        # evidence or a log whose usage differs from the immutable receipt.
        for index, item in enumerate(records[:completed], 1):
            receipt = self.store._read(self.store.steps / f'{index:04d}.receipt.json')['result']
            input_data = item.get('input') or {}
            output_data = item.get('output') or {}
            screenshot = input_data.get('screenshot')
            if not isinstance(screenshot, str) or not screenshot:
                raise UnsafeResume(f'Missing logged screenshot for step {index}')
            logged_path = Path(screenshot)
            if not logged_path.is_absolute():
                logged_path = self.root.parent / 'agent_0' / logged_path
            try:
                evidence_matches = _digest(logged_path) == receipt['trace']['screenshot_sha256']
            except (OSError, KeyError, TypeError) as exc:
                raise UnsafeResume(f'Lost screenshot evidence at step {index}') from exc
            if not evidence_matches or output_data.get('usage') != receipt['usage']:
                raise UnsafeResume(f'GameWorld log/receipt mismatch at step {index}')
        if len(lines) > completed:
            if not has_receipt or len(lines) > completed + 1:
                raise UnsafeResume('Uncommitted logs with no paid-response receipt')
            orphan = records[-1]
            if orphan.get('interaction_id') != completed + 1:
                raise UnsafeResume('Unexpected orphan interaction number')
            target = self.root / 'quarantine' / f'{completed + 1:04d}_orphan_interaction.jsonl'
            if target.exists():
                raise UnsafeResume('Orphan interaction was already quarantined')
            _state.atomic_write(target, lines[-1] + b'\n')
            _state.atomic_write(path, b''.join(line + b'\n' for line in lines[:completed]))

    def completed_summary(self, max_steps: int) -> bool:
        """Skip a finished browser episode after an outer-run interruption."""
        if not (self.root / 'genesis.json').is_file():
            return False
        self.store.load_genesis()
        latest = self.store.recover_finished_commit()
        if latest is None:
            return False
        terminal = latest['step'] >= max_steps or latest['agent_state']['eval_metrics'].get('finalized') is True
        if not terminal:
            return False
        if (self.root / 'inflight.json').exists() or (self.root / 'model_result.json').exists():
            raise UnsafeResume('Terminal episode has an unexpected pending paid request')
        self._verify_log_path(self.root.parent / 'agent_0/interactions.jsonl', latest['step'], False)
        summary = self.store._read(self.root.parent / 'agent_0/evaluation/summary.json')
        if (summary.get('finalized') is not True or summary.get('interaction_id') != latest['step']
                or summary.get('progress') != latest['agent_state']['eval_metrics'].get('progress')):
            raise UnsafeResume('Finished episode evaluation differs from committed game step')
        return True

    def prepare_retry(self) -> dict | None:
        """Before (re)starting: a request interrupted without a response receipt
        may have been billed. Record it as unknown cost and allow the step's
        inference once more; a receipted response is always replayed instead."""
        if not (self.root / 'genesis.json').is_file():
            return None
        if (self.root / 'inflight.json').exists() and not (self.root / 'model_result.json').exists():
            latest = self.store.load()
            return self.store.authorize_retry_after_unknown(
                step=latest['step'] + 1 if latest else 1,
                operator_reason='automatic resume after a transport/browser failure; '
                                'the interrupted request may have been billed')
        return None

    def adopt_completed_traces(self, agent) -> None:
        """A finished episode is not replayed; AHE evolution still needs its step traces."""
        latest = self.store.load()
        worker = (latest or {}).get('agent_state', {}).get('client', {}).get('worker')
        if self.framework == 'ahe':
            if not isinstance(worker, dict):
                raise UnsafeResume('Finished AHE episode lacks its worker snapshot')
            agent.adopt_trace_paths(worker.get('trace_paths') or [])

    def _receipt_trace(self, client, screenshot, number):
        trace = client._last_interaction
        if not isinstance(trace, dict):
            raise UnsafeResume('Model returned without GameWorld interaction record')
        screenshot_record = self._asset(Path(screenshot), self.root / 'images' / f'{number:04d}_input.png')
        value = _json(trace)
        value['screenshot_path'] = screenshot_record['path']
        value['screenshot_sha256'] = screenshot_record['sha256']
        return value

    async def infer(self, coordinator, agent, original):
        client = agent.client
        original_get_action = client.get_action
        def guarded(screenshot):
            pending = self.store.pending_model_result()
            if pending is not None:
                if self.framework != 'ahe':
                    self._restore_client(client, pending['agent_after_model']['client'])
                else:
                    # AHE's worker was restored to the receipted post-call
                    # history at initialization (restore_session is one-shot).
                    actual = client._ahe_agent.snapshot_session()
                    if actual['history'] != self._unpack_worker(
                            pending['agent_after_model']['client']['worker'])['history']:
                        raise UnsafeResume('AHE paid-response history differs')
                trace = dict(pending['trace'])
                sha = trace.pop('screenshot_sha256', None)
                if _digest(Path(trace['screenshot_path'])) != sha:
                    raise UnsafeResume('Paid-response screenshot changed')
                trace['screenshot_path'] = Path(trace['screenshot_path'])
                from agents.harness.memory import MemoryEntry
                trace['memory_entries'] = [MemoryEntry(**item) for item in trace['memory_entries']]
                # The coordinator logs this persisted response without another model call.
                client._last_interaction = trace
                self._latest_receipt = pending
                return pending['action']['value']
            self.store.require_safe_to_infer()
            number = agent.step_index + 1
            self.store.mark_model_call(number, uuid4().hex)
            action = original_get_action(screenshot)
            trace = self._receipt_trace(client, screenshot, number)
            usage = trace.get('usage')
            if not isinstance(usage, dict):
                usage = {'total_tokens': None, 'coverage': 'unknown'}
            result = {'action': {'value': _json(action)}, 'usage': usage,
                      'trace': trace, 'at': datetime.now(timezone.utc).isoformat(),
                      'agent_after_model': {'schema': 1, 'client': self._snapshot_client(client, number)}}
            self.store.record_model_result(number, result)
            self._latest_receipt = result
            return action
        client.get_action = guarded
        try:
            return await original(coordinator, agent)
        finally:
            client.get_action = original_get_action

    async def commit(self, coordinator, agent, original, logger):
        before = agent.step_index
        await original(coordinator, agent, logger)
        if agent.step_index != before + 1:
            raise UnsafeResume('Coordinator step did not advance exactly once')
        number = agent.step_index
        receipt = self.store.pending_model_result()
        if receipt is None or self._latest_receipt is None:
            raise UnsafeResume('Completed game step has no paid-response receipt')
        try:
            world = await self._save(coordinator, agent)
            observed = self._observation(world, agent)
        except Exception:
            if not coordinator._stop_event.is_set():
                raise
            # Successful task transitions may leave play mode. It is terminal
            # and must never be treated as a resumable live browser snapshot.
            prior = self.store.load()
            world = {'terminal': True, 'last_world': prior['world_sha256'] if prior else 'genesis',
                     'last_step': number}
            observed = {'terminal': True, 'step_index': number,
                        'eval_metrics': _json(agent.eval_metrics)}
        old = self.store.load()
        ledger = old['ledger'] if old else self.store.load_genesis()['ledger']
        used = receipt['usage'].get('total_tokens')
        known = (isinstance(used, int) and not isinstance(used, bool) and used > 0
                 and isinstance(ledger.get('total_tokens'), int))
        ledger = {'schema': 1, 'total_tokens': ledger['total_tokens'] + used if known else None,
                  'wall_seconds': ledger['wall_seconds'] + (time.monotonic() - self.started_at),
                  'last_model_usage': receipt['usage'], 'last_request_at': receipt['at']}
        self.started_at = time.monotonic()
        agent_state = self._snapshot_agent(coordinator, agent, number)
        self.store.commit(step=number, world=world, observed_state=observed,
                          agent_state=agent_state, ledger=ledger)


@contextmanager
def coordinator_resume(transaction: AstrayTransaction):
    """Install hooks only for this one episode; don't alter other experiments."""
    from runtime.coordinator import Coordinator
    from env.browser_manager import BrowserGameManager
    original_navigate = BrowserGameManager._navigate_to_game
    original_run, original_step, original_infer, original_control = (
        Coordinator.run, Coordinator._run_agent_step, Coordinator._get_raw_action,
        Coordinator._handle_eval_controls)

    async def navigate(manager):
        # Serve the verified original/overlay API with an in-memory, score-neutral
        # evaluator-clock snapshot bridge. Never alter the source checkout.
        from urllib.parse import urlsplit
        if manager.context is None or manager.page is None:
            raise UnsafeResume('Browser page unavailable for checkpoint bridge')
        if not urlsplit(manager.config.game_url).path.endswith('/index.html'):
            raise UnsafeResume('Unexpected game page for checkpoint bridge')
        def server(name, body):
            # Playwright may pass (route) or (route, request): accept both.
            async def serve(route, *_):
                if not urlsplit(route.request.url).path.endswith('/' + name):
                    raise UnsafeResume('Unexpected URL for checkpoint bridge')
                content_type = 'text/html' if name.endswith('.html') else 'text/javascript'
                await route.fulfill(status=200, body=body, content_type=content_type)
            return serve
        for name, body in transaction._routes.items():
            serve = server(name, body)
            await manager.context.route('**/' + name, serve)
            await manager.context.route('**/' + name + '?*', serve)
        await original_navigate(manager)
        probe = ("() => !!(window.__gwAstrayCheckpointRuntime && window.gameAPI)"
                 if transaction.game == ASTRAY else
                 "() => !!(window.__gwCheckpointRuntime && window.__gwGameRandomState && window.gameAPI)")
        if not await manager.page.evaluate(probe):
            raise UnsafeResume('Checkpoint bridge was not installed')

    async def run(coordinator, *, start_env=True):
        if not start_env or len(coordinator.agents) != 1:
            raise UnsafeResume('Step resume requires one fresh agent/browser')
        try:
            await transaction.initialize(coordinator)
            return await original_run(coordinator, start_env=False)
        finally:
            if coordinator.env.game_manager or coordinator.env.game_launcher:
                await coordinator.env.close_game()

    async def step(coordinator, agent, logger):
        coordinator._astray_pending_stop = False
        try:
            await transaction.commit(coordinator, agent, original_step, logger)
        finally:
            if coordinator._astray_pending_stop:
                coordinator._stop_event.set()

    async def control(coordinator, agent, result):
        if result and result.should_reset:
            reset_ok = await coordinator.env.reset_game()
            if reset_ok and coordinator.evaluator:
                agent.eval_metrics = coordinator.evaluator.reset_metrics(agent.eval_metrics)
            if not reset_ok:
                coordinator._astray_pending_stop = True
        if result and result.should_stop:
            # Setting stop_event inside the original step races the durable
            # snapshot: original Coordinator.run cancels the agent task.
            coordinator._astray_pending_stop = True

    async def infer(coordinator, agent):
        return await transaction.infer(coordinator, agent, original_infer)

    transaction.root.mkdir(parents=True, exist_ok=True)
    with (transaction.root / '.episode.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UnsafeResume('Astray episode already has an active owner') from exc
        BrowserGameManager._navigate_to_game = navigate
        Coordinator.run, Coordinator._run_agent_step, Coordinator._get_raw_action, Coordinator._handle_eval_controls = run, step, infer, control
        try:
            yield
        finally:
            BrowserGameManager._navigate_to_game = original_navigate
            Coordinator.run, Coordinator._run_agent_step, Coordinator._get_raw_action, Coordinator._handle_eval_controls = (
                original_run, original_step, original_infer, original_control)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
