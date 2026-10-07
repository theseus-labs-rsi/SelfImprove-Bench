#!/usr/bin/env python3
"""Offline step-resume acceptance on the real GameWorld stack (no model/API calls).

Real framework client, real Coordinator, evaluator, actions and browser game; the
model is replaced by a scripted backend whose n-th call returns SCRIPT[n]. Run a
leg with --fault-after K (crash after the durable commit of step K) or
--fault-after-receipt K (crash after the paid-response receipt of step K, before
its game action), then rerun the same --root: it must resume from the saved
step without re-requesting any receipted response.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

from benches.gameworld.astray_step_resume import AstrayTransaction, coordinator_resume
from benches.gameworld.frameworks import client_import_spec

SCRIPTS = {
    '03_astray': ['move_down', 'move_down', 'move_right', 'move_right', 'move_up', 'move_left',
                  'move_down', 'move_right'],
    # Face the ground and mine repeatedly: exercises blocks, inventory and score restore.
    '18_minecraft-clone-glm': ['look_down', 'look_down', 'look_down', 'mine_target', 'mine_target',
                               'mine_target', 'mine_target', 'move_forward', 'mine_target',
                               'mine_target', 'look_left', 'mine_target'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', required=True, choices=('ace', 'hyperagents', 'ahe'))
    parser.add_argument('--game', required=True, choices=tuple(SCRIPTS))
    parser.add_argument('--game-task', required=True)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--gameworld', required=True, type=Path)
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--fault-after', type=int, default=0)
    parser.add_argument('--fault-after-receipt', type=int, default=0)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    code = args.gameworld.resolve() / 'code'
    sys.path.insert(0, str(code))
    from agents.mm_agents.base.base_client import BaseClientConfig
    from main import main as gameworld_main
    import importlib
    module, cls = client_import_spec(args.framework)
    client_class = getattr(importlib.import_module(module), cls)
    calls_path = args.root / 'offline_mock_calls.jsonl'
    script = SCRIPTS[args.game]

    class ScriptedBackend:
        def __init__(self):
            self.history = []
            self._attempt_index = 0
            self.trace_paths = []

        def act(self, prompt, image_paths):
            n = sum(1 for _ in calls_path.open()) if calls_path.exists() else 0
            action = script[n % len(script)]
            with calls_path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps({'call': n + 1, 'action': action, 'images': len(image_paths or [])}) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            self._attempt_index += 1
            self.history.append({'call': n + 1})
            usage = SimpleNamespace(input_tokens=30, cached_input_tokens=0,
                                    output_tokens=12, reasoning_tokens=0, total_tokens=42)
            return SimpleNamespace(text=json.dumps({'tool_name': action, 'arguments': {'reasoning': 'scripted'}}),
                                   reasoning='scripted', usage=usage, metadata={})

        def snapshot_session(self):
            return {'schema': 1, 'history': list(self.history), 'attempt_index': self._attempt_index,
                    'last_usage_snapshot': None, 'unreconciled_failed_attempts': False,
                    'trace_paths': []}

        def restore_session(self, saved):
            if self._attempt_index:
                raise RuntimeError('Scripted worker restored after inference')
            self.history = list(saved['history'])
            self._attempt_index = saved['attempt_index']

        def adopt_trace_paths(self, paths):
            self.trace_paths = list(paths)

    task = {'task_id': 'offline_' + args.game_task, 'game_id': args.game,
            'game_task_id': args.game_task, 'framework': args.framework,
            'gameworld_root': str(args.gameworld.resolve()), 'max_steps': args.max_steps,
            'model_spec': 'azure-gpt-5.6-sol', 'reasoning_effort': 'low'}
    trans = AstrayTransaction(run_root=args.root, task=task, gameworld_root=args.gameworld.resolve(),
                              framework=args.framework)
    if args.fault_after_receipt:
        original_receipt = type(trans.store).record_model_result
        def fault_receipt(store, number, result):
            saved = original_receipt(store, number, result)
            if number == args.fault_after_receipt:
                raise RuntimeError('OFFLINE_INJECTED_AFTER_PAID_RECEIPT')
            return saved
        type(trans.store).record_model_result = fault_receipt
    original_commit = trans.commit
    async def fault_after(coordinator, agent, original_step, logger):
        await original_commit(coordinator, agent, original_step, logger)
        if agent.step_index == args.fault_after:
            raise RuntimeError('OFFLINE_INJECTED_AFTER_DURABLE_COMMIT')
    if args.fault_after:
        trans.commit = fault_after

    def factory(cfg, agent_ids):
        return [client_class(BaseClientConfig(model='offline-mock', log_root=str(args.root),
                                              memory_rounds=2), ScriptedBackend(),
                             semantic_controls_specs=cfg.semantic_controls_specs[i])
                for i, _ in enumerate(agent_ids)]

    if trans.completed_summary(args.max_steps):
        print('ALREADY COMPLETE: episode reused without replay', flush=True)
    else:
        trans.prepare_retry()
        with coordinator_resume(trans):
            try:
                asyncio.run(gameworld_main(config_preset=f'{args.game}+{args.game_task}+azure-gpt-5.6-sol',
                    headless=True, port=args.port, log_root=str(args.root), rerun_policy='never',
                    max_steps_override=args.max_steps, client_factory=factory))
            except Exception as exc:
                text = str(exc.__cause__ or exc)
                if args.fault_after and 'OFFLINE_INJECTED_AFTER_DURABLE_COMMIT' in text:
                    print('EXPECTED OFFLINE INTERRUPTION after step', args.fault_after, flush=True)
                elif args.fault_after_receipt and 'OFFLINE_INJECTED_AFTER_PAID_RECEIPT' in text:
                    print('EXPECTED OFFLINE INTERRUPTION after receipt', args.fault_after_receipt, flush=True)
                else:
                    raise
    last = trans.store.load()
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    rows = [json.loads(line) for line in (args.root / 'agent_0/interactions.jsonl').read_text().splitlines()]
    print(json.dumps({'framework': args.framework, 'game': args.game, 'step': last['step'],
                      'calls': len(calls), 'logged': len(rows),
                      'actions': [r['output']['parsed_action']['tool_name'] for r in rows],
                      'total_tokens': last['ledger']['total_tokens']}), flush=True)
    if len(calls) != last['step'] + bool(args.fault_after_receipt) or len(rows) != last['step']:
        raise RuntimeError('Committed step/call/log count differs')
    if [r['interaction_id'] for r in rows] != list(range(1, last['step'] + 1)):
        raise RuntimeError('Interaction IDs are not contiguous')


if __name__ == '__main__':
    main()
