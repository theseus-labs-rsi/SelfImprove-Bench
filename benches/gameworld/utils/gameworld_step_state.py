"""Durable, fail-closed state storage for GameWorld mid-episode recovery.

This module intentionally cannot resume an episode by itself: engine and agent
session restore must pass independent no-LLM equivalence checks first. Checkpoints
are immutable; an in-flight model request is NOT a committed game step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


class UnsafeResume(RuntimeError):
    """The caller MUST NOT continue or issue another model request."""


def encode(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sync_dir(path: Path) -> None:
    if os.name == 'nt':
        return  # Directory fsync is unavailable through os.open on Windows.
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    """Never expose truncated metadata even if the process exits during write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class StepStore:
    root: Path
    run_id: str
    source_digest: str
    # Default preserves the existing per-call complete audit for ACE and older runs.
    # Opt-in AHE mode audits the whole chain at every Nth boundary and on every
    # process restart, while each paid response and playable world is still
    # persisted at EVERY step. No duplicate model calls are permitted.
    integrity_audit_interval: int = 1
    _verified: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.integrity_audit_interval, int) or
                isinstance(self.integrity_audit_interval, bool) or
                not 1 <= self.integrity_audit_interval <= 100):
            raise ValueError('integrity_audit_interval must be an integer in 1..100')

    @property
    def steps(self) -> Path:
        return self.root / 'steps'

    def _read(self, path: Path) -> dict:
        try:
            result = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise UnsafeResume(f'Unreadable recovery record: {path}') from exc
        if not isinstance(result, dict):
            raise UnsafeResume(f'Unexpected recovery record shape: {path}')
        return result

    def _verify(self, n: int, prior_digest: str | None) -> tuple[dict, str]:
        manifest = self._read(self.steps / f'{n:04d}.json')
        path = self.steps / f'{n:04d}.world.json'
        receipt_path = self.steps / f'{n:04d}.receipt.json'
        try:
            world_bytes = path.read_bytes()
            receipt_bytes = receipt_path.read_bytes()
        except OSError as exc:
            raise UnsafeResume(f'Missing immutable world/paid response receipt at step {n}') from exc
        if (manifest.get('schema') != 1 or manifest.get('step') != n
                or manifest.get('run_id') != self.run_id
                or manifest.get('source_digest') != self.source_digest
                or manifest.get('parent_digest') != prior_digest
                or manifest.get('world_sha256') != sha256(world_bytes)
                or manifest.get('model_result_sha256') != sha256(receipt_bytes)):
            raise UnsafeResume(f'Step {n} is inconsistent, corrupted or from a different run')
        receipt = self._read(receipt_path)
        if (receipt.get('request_id') != manifest.get('model_request_id')
                or receipt.get('run_id') != self.run_id
                or receipt.get('source_digest') != self.source_digest
                or receipt.get('step') != n):
            raise UnsafeResume(f'Step {n} paid response receipt identity mismatch')
        try:
            world = json.loads(world_bytes)
        except json.JSONDecodeError as exc:
            raise UnsafeResume(f'Invalid step {n} world JSON') from exc
        if not isinstance(world, dict) or not world:
            raise UnsafeResume(f'Step {n} has no game save')
        if (not isinstance(manifest.get('observed_state'), dict)
                or not isinstance(manifest.get('agent_state'), dict)
                or not isinstance(manifest.get('ledger'), dict)):
            raise UnsafeResume(f'Step {n} lacks required observed/agent/cost evidence')
        return manifest, sha256(encode(manifest))

    def load(self) -> dict | None:
        """Audit every persisted step after restart and at configured boundaries.

        In the opt-in mode, the exclusive episode owner caches only a verified
        prefix. An advanced pointer verifies exactly the new immutable step;
        all 20 steps are re-read at the next full audit. The default (interval
        1) retains the original full-chain check on every call.
        """
        if not self.root.exists():
            return None
        manifests = sorted(self.steps.glob('[0-9][0-9][0-9][0-9].json'))
        snapshots = sorted(self.steps.glob('[0-9][0-9][0-9][0-9].world.json'))
        receipts = sorted(self.steps.glob('[0-9][0-9][0-9][0-9].receipt.json'))
        if len(manifests) != len(snapshots) or len(manifests) != len(receipts):
            raise UnsafeResume('Orphaned immutable snapshot/manifest: interrupted commit')
        if not manifests:
            if snapshots or receipts:
                raise UnsafeResume('No committed recovery step; orphan receipt or world exists')
            self._verified.clear()
            return None
        count = len(manifests)
        if (manifests[-1].name != f'{count:04d}.json' or
                snapshots[-1].name != f'{count:04d}.world.json' or
                receipts[-1].name != f'{count:04d}.receipt.json'):
            raise UnsafeResume('Recovery steps are not a contiguous prefix')
        pointer = self._read(self.root / 'latest.json')
        cached = self._verified
        if self.integrity_audit_interval > 1 and cached:
            prior = cached['step']
            if count == prior and pointer == {'step': prior, 'digest': cached['digest']}:
                return deepcopy(cached['manifest'])
            if (count == prior + 1 and count % self.integrity_audit_interval != 0
                    and pointer.get('step') == count):
                latest, digest = self._verify(count, cached['digest'])
                if pointer != {'step': count, 'digest': digest}:
                    raise UnsafeResume('Latest pointer does not match the committed step prefix')
                cached.update(step=count, manifest=latest, digest=digest)
                return deepcopy(latest)
            # Changed pointer or scheduled full audit: never trust cached state.
        previous = None
        latest = None
        for step, path in enumerate(manifests, 1):
            if path.name != f'{step:04d}.json':
                raise UnsafeResume('Recovery steps are not a contiguous prefix')
            latest, previous = self._verify(step, previous)
        if pointer != {'step': count, 'digest': previous}:
            raise UnsafeResume('Latest pointer does not match the committed step prefix')
        cached.update(step=count, manifest=latest, digest=previous)
        # Runtime restore intentionally mutates diagnostic/evaluator dictionaries.
        # Never lend it references into the verified immutable-prefix cache.
        return deepcopy(latest)

    def mark_model_call(self, step: int, request_id: str) -> None:
        """Set intent BEFORE inference. An uncertain paid call requires audit."""
        if (not isinstance(request_id, str) or not request_id
                or '/' in request_id or '..' in request_id or '\\' in request_id):
            raise ValueError('Model request must have a stable nonempty ID')
        latest = self.load()
        if step != (latest['step'] + 1 if latest else 1):
            raise UnsafeResume('Model call is not for the next uncommitted step')
        path = self.root / 'inflight.json'
        if path.exists() or (self.root / 'model_result.json').exists():
            raise UnsafeResume('Unreconciled model call exists: refusing a duplicate paid request')
        atomic_write(path, encode({'step': step, 'request_id': request_id,
                                   'run_id': self.run_id, 'source_digest': self.source_digest}))

    def authorize_retry_after_unknown(self, *, step: int, operator_reason: str) -> dict:
        """Manual-only: accept potential second charge for THIS step, not the prefix.

        Unknown usage remains unknown forever unless independently audited. A
        provider response receipt cannot be discarded through this path.
        """
        if not isinstance(operator_reason, str) or len(operator_reason.strip()) < 12:
            raise UnsafeResume('Explicit operator explanation is required to risk another charge')
        if (self.root / 'model_result.json').exists():
            raise UnsafeResume('A paid response is already available; replay it instead')
        intent = self._read(self.root / 'inflight.json')
        latest = self.load()
        if (intent.get('step') != step
                or step != (latest['step'] + 1 if latest else 1)
                or intent.get('run_id') != self.run_id
                or intent.get('source_digest') != self.source_digest):
            raise UnsafeResume('Unknown inference is not the next step of this run')
        audit = self.root / 'abandoned_attempts'
        audit.mkdir(parents=True, exist_ok=True)
        record = {'schema': 1, 'intent': intent, 'operator_reason': operator_reason,
                  'usage_status': 'unknown_may_have_billed',
                  'next_retry_may_be_charged': True}
        target = audit / f'{step:04d}.{intent["request_id"]}.json'
        if target.exists() or '/' in intent['request_id'] or '..' in intent['request_id']:
            raise UnsafeResume('Unsafe or duplicate model request ID')
        atomic_write(target, encode(record))
        (self.root / 'inflight.json').unlink()
        _sync_dir(self.root)
        return record

    def save_genesis(self, *, world: dict, observed_state: dict,
                     agent_state: dict, ledger: dict) -> dict:
        """Durable step-zero boundary. Never replace a previous run's start."""
        if self.root.exists() and any(path.name != '.episode.lock' for path in self.root.iterdir()):
            raise UnsafeResume('Run already has recovery records; genesis cannot be replaced')
        if not all(isinstance(value, dict) and value for value in
                   (world, observed_state, agent_state, ledger)):
            raise UnsafeResume('Incomplete initial world/agent/cost snapshot')
        data = {'schema': 1, 'run_id': self.run_id,
                'source_digest': self.source_digest, 'world': world,
                'observed_state': observed_state, 'agent_state': agent_state,
                'ledger': ledger}
        atomic_write(self.root / 'genesis.json', encode(data))
        return data

    def load_genesis(self) -> dict:
        data = self._read(self.root / 'genesis.json')
        if (data.get('schema') != 1 or data.get('run_id') != self.run_id
                or data.get('source_digest') != self.source_digest
                or not all(isinstance(data.get(key), dict) and data[key] for key in
                           ('world', 'observed_state', 'agent_state', 'ledger'))):
            raise UnsafeResume('Initial checkpoint is missing, incomplete or from another run')
        return data

    def record_model_result(self, step: int, result: dict) -> dict:
        """Persist complete response, usage and post-inference session before acting.

        No secret/API headers belong here. An interrupted request without a response
        receipt must be reconciled by the operator, never silently called again.
        """
        intent = self._read(self.root / 'inflight.json')
        latest = self.load()
        if (step != (latest['step'] + 1 if latest else 1)
                or intent.get('step') != step or intent.get('run_id') != self.run_id
                or intent.get('source_digest') != self.source_digest):
            raise UnsafeResume('Model response does not match current request')
        if (not isinstance(result, dict) or not isinstance(result.get('action'), dict)
                or not isinstance(result.get('agent_after_model'), dict)
                or not result['agent_after_model']
                or not isinstance(result.get('usage'), dict) or not result['usage']):
            raise UnsafeResume('Cannot durably record an incomplete model result/usage')
        receipt = {'schema': 1, 'run_id': self.run_id,
                   'source_digest': self.source_digest, 'step': step,
                   'request_id': intent['request_id'], 'result': result}
        path = self.root / 'model_result.json'
        if path.exists():
            if self._read(path) != receipt:
                raise UnsafeResume('Conflicting model response for same paid request')
            return receipt
        atomic_write(path, encode(receipt))
        return receipt

    def pending_model_result(self) -> dict | None:
        """Return only a verified, durable response for the next uncommitted step."""
        path = self.root / 'model_result.json'
        if not path.exists():
            return None
        receipt = self._read(path)
        intent = self._read(self.root / 'inflight.json')
        latest = self.load()
        if (receipt.get('schema') != 1
                or receipt.get('step') != (latest['step'] + 1 if latest else 1)
                or receipt.get('run_id') != self.run_id
                or receipt.get('source_digest') != self.source_digest
                or receipt.get('request_id') != intent.get('request_id')
                or not isinstance(receipt.get('result'), dict)):
            raise UnsafeResume('Paid response receipt does not match pending inference')
        return receipt['result']

    def commit(self, *, step: int, world: dict, observed_state: dict,
               agent_state: dict, ledger: dict) -> dict:
        """Commit a complete boundary only; never assume the browser can restore it."""
        inflight = self.root / 'inflight.json'
        intent = self._read(inflight)
        previous = self.load()
        expected = previous['step'] + 1 if previous else 1
        if (step != expected or intent.get('step') != step
                or intent.get('run_id') != self.run_id
                or intent.get('source_digest') != self.source_digest):
            raise UnsafeResume('Boundary does not match current in-flight request or parent')
        if (not isinstance(world, dict) or not world
                or not isinstance(observed_state, dict) or not observed_state
                or not isinstance(agent_state, dict) or not agent_state
                or not isinstance(ledger, dict) or not ledger):
            raise UnsafeResume('Incomplete game/agent/usage snapshot: refusing to commit')
        unknown = sorted((self.root / 'abandoned_attempts').glob('*.json'))
        if unknown:
            ledger = {**ledger, 'billing_complete': False,
                      'total_tokens': None,
                      'unreconciled_attempts': [p.name for p in unknown]}
        self.steps.mkdir(parents=True, exist_ok=True)
        saved = encode(world)
        snapshot_path = self.steps / f'{step:04d}.world.json'
        manifest_path = self.steps / f'{step:04d}.json'
        archived_receipt = self.steps / f'{step:04d}.receipt.json'
        if snapshot_path.exists() or manifest_path.exists() or archived_receipt.exists():
            raise UnsafeResume('Immutable recovery step already exists')
        receipt_path = self.root / 'model_result.json'
        if not receipt_path.exists():
            raise UnsafeResume('A charged model request has no durable response receipt')
        self.pending_model_result()  # verify BEFORE writing a world snapshot
        receipt_bytes = receipt_path.read_bytes()
        receipt_digest = sha256(receipt_bytes)
        atomic_write(snapshot_path, saved)
        atomic_write(archived_receipt, receipt_bytes)
        parent = sha256(encode(previous)) if previous else None
        manifest = {'schema': 1, 'step': step, 'run_id': self.run_id,
                    'source_digest': self.source_digest, 'parent_digest': parent,
                    'world_sha256': sha256(saved), 'model_request_id': intent['request_id'],
                    'model_result_sha256': receipt_digest,
                    'observed_state': observed_state, 'agent_state': agent_state,
                    'ledger': ledger}
        atomic_write(manifest_path, encode(manifest))
        atomic_write(self.root / 'latest.json', encode({'step': step, 'digest': sha256(encode(manifest))}))
        # Only remove pending result AFTER immutable checkpoint and pointer are durable.
        receipt = self.root / 'model_result.json'
        if receipt.exists():
            # Immutable per-step receipt (full response and usage) is retained.
            receipt.unlink()
        inflight.unlink()
        _sync_dir(self.root)
        return manifest

    def _quarantine_uncommitted_world(self, committed_step: int) -> None:
        """Preserve a torn world-only write, then replay its durable paid receipt.

        No gameplay action is trusted from a world blob without a step manifest.
        Never quarantine an unrelated or corrupted prefix.
        """
        orphan = self.steps / f'{committed_step + 1:04d}.world.json'
        staged_receipt = self.steps / f'{committed_step + 1:04d}.receipt.json'
        if (not orphan.exists() and not staged_receipt.exists()) or (self.steps / f'{committed_step + 1:04d}.json').exists():
            return
        previous = None
        for step in range(1, committed_step + 1):
            _, previous = self._verify(step, previous)
        pointer = self.root / 'latest.json'
        if committed_step and self._read(pointer) != {'step': committed_step, 'digest': previous}:
            raise UnsafeResume('Cannot quarantine orphan from unverified prefix')
        intent = self._read(self.root / 'inflight.json')
        receipt = self._read(self.root / 'model_result.json')
        if (intent.get('step') != committed_step + 1
                or receipt.get('step') != committed_step + 1
                or intent.get('request_id') != receipt.get('request_id')
                or intent.get('run_id') != self.run_id
                or receipt.get('source_digest') != self.source_digest):
            raise UnsafeResume('Cannot quarantine world without matching paid receipt')
        if staged_receipt.exists() and (sha256(staged_receipt.read_bytes())
                                        != sha256(encode(receipt))):
            raise UnsafeResume('Uncommitted paid receipt differs from durable model result')
        quarantine = self.root / 'quarantine'
        quarantine.mkdir(exist_ok=True)
        for artifact in (orphan, staged_receipt):
            if not artifact.exists():
                continue
            suffix = sha256(artifact.read_bytes())
            target = quarantine / f'{artifact.name}.{suffix}'
            if target.exists():
                raise UnsafeResume('Quarantined artifact already exists; manual audit required')
            os.rename(artifact, target)
        _sync_dir(self.steps)
        _sync_dir(quarantine)

    def recover_finished_commit(self) -> dict | None:
        """Finish only a verified checkpoint written before an interrupted fsync.

        Never repair a lone world blob, or a paid response without a completed
        manifest: those could represent an action that has not been evaluated.
        """
        manifests = sorted(self.steps.glob('[0-9][0-9][0-9][0-9].json'))
        if not manifests:
            self._quarantine_uncommitted_world(0)
            return self.load()
        last_step = len(manifests)
        self._quarantine_uncommitted_world(last_step)
        last_path = manifests[-1]
        if last_path.name != f'{last_step:04d}.json':
            raise UnsafeResume('Cannot finalize a noncontiguous checkpoint')
        pointer = self.root / 'latest.json'
        current = self._read(pointer) if pointer.exists() else None
        if current and current.get('step') == last_step:
            committed = self.load()
            if (self.root / 'inflight.json').exists():
                intent = self._read(self.root / 'inflight.json')
                if intent.get('step') == last_step + 1:
                    # An inference was started after the last fully committed step.
                    return committed
                if intent.get('step') != last_step or intent.get('request_id') != committed.get('model_request_id'):
                    raise UnsafeResume('Unexpected in-flight request after checkpoint commit')
                receipt = self.root / 'model_result.json'
                if receipt.exists():
                    if sha256(receipt.read_bytes()) != committed.get('model_result_sha256'):
                        raise UnsafeResume('Pending paid response differs from committed result')
                    receipt.unlink()
                (self.root / 'inflight.json').unlink()
                _sync_dir(self.root)
            return committed
        if current and current.get('step') != last_step - 1:
            raise UnsafeResume('Unexpected pointer lag; refusing recovery')
        previous = None
        for step in range(1, last_step + 1):
            last, previous = self._verify(step, previous)
        if current and current.get('digest') != last.get('parent_digest'):
            raise UnsafeResume('Checkpoint parent does not match last committed pointer')
        intent = self._read(self.root / 'inflight.json')
        receipt_path = self.root / 'model_result.json'
        if (intent.get('step') != last_step
                or intent.get('request_id') != last.get('model_request_id')
                or not receipt_path.exists()
                or sha256(receipt_path.read_bytes()) != last.get('model_result_sha256')):
            raise UnsafeResume('Cannot finalize checkpoint without matching paid response')
        atomic_write(pointer, encode({'step': last_step, 'digest': previous}))
        receipt_path.unlink()
        (self.root / 'inflight.json').unlink()
        _sync_dir(self.root)
        return self.load()

    def require_safe_to_infer(self) -> dict | None:
        latest = self.load()
        if (self.root / 'model_result.json').exists() and not (self.root / 'inflight.json').exists():
            raise UnsafeResume('Orphaned model response receipt')
        if (self.root / 'inflight.json').exists():
            raise UnsafeResume('Model call was interrupted; reconcile charges/action before inference')
        return latest
