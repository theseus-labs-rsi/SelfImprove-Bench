"""Explicit, isolated entrypoint; no API request without --allow-paid.

Each training round is a new OS process, using the unchanged legacy checkpoint
contract. Official holdout always runs in a separate process from the overlay.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
# Positive integer or "all"; long trajectories use an ordered uniform sample.
REFINEMENT_IMAGES = "8"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def verify_release():
    path = ROOT / "provenance/release_sha256.json"
    if not path.is_file():
        raise RuntimeError("Release manifest missing")
    for relative, expected in json.loads(path.read_text()).items():
        p = ROOT / relative
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Release file changed: {relative}; audit/reseal before paid runs")


def configure_environment(combination=None):
    os.environ["RSI_ACE_CONFIG_PATH"] = str(ROOT / "configs/ace.toml")
    os.environ["RSI_HYPERAGENTS_CONFIG_PATH"] = str(ROOT / "configs/hyperagents.toml")
    os.environ["PYTHONPATH"] = str(CODE)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Re-read the key file so each round's worker picks up a rotated key
    # without restarting a long-running launcher. The key never enters source.
    # A per-combination key file (~/.rsi_openai.<game>_<method>.env, usually a
    # symlink) spreads combinations over gateways; remove it to fall back.
    key_file = Path(os.environ.get("RSI_KEY_FILE", Path.home() / ".rsi_openai.env"))
    if combination and "RSI_KEY_FILE" not in os.environ:
        routed = Path.home() / f".rsi_openai.{combination}.env"
        if routed.is_file():
            key_file = routed
    if key_file.is_file():
        for line in key_file.read_text().splitlines():
            match = re.fullmatch(r"\s*(?:export\s+)?(OPENAI_API_KEY|OPENAI_BASE_URL)=['\"]?([^'\"\s]+)['\"]?\s*", line)
            if match:
                os.environ[match.group(1)] = match.group(2)
    # Same explicitly configured endpoint/key for actors and AHE evolver.
    os.environ["LLM_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["LLM_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "")
    os.environ["LLM_MODEL"] = "gpt-5.6-sol"
    os.environ["GAMEWORLD_RSI_MODEL"] = "gpt-5.6-sol"
    os.environ.setdefault("GAMEWORLD_CONNECTION_MAX_ATTEMPTS", "4")
    # Step-level episode resume (benches/gameworld/astray_step_resume.py). An
    # execution option only: it is not part of the hashed config/task manifest.
    os.environ.setdefault("RSI_STEP_RESUME", "1")
    # Transport bounds (execution options, not hashed). A stalled gateway
    # request fails after 150s instead of the SDK's 600s x 3; the game client's
    # call_with_connection_retry then retries. AHE's whole step is capped at 300s.
    os.environ.setdefault("RSI_OPENAI_REQUEST_TIMEOUT", "150")
    os.environ.setdefault("RSI_OPENAI_MAX_RETRIES", "0")
    os.environ.setdefault("RSI_AHE_STEP_TIMEOUT", "300")
    # One image budget for ACE Reflector, AHE Evolver and HyperAgents inspection.
    os.environ["RSI_MAX_REFINEMENT_IMAGES"] = REFINEMENT_IMAGES
    # Headless Chromium crashes (SIGTRAP) with its profile on the shared network FS.
    tmpdir = Path(os.environ.get(
        "RSI_LOCAL_TMPDIR", str(Path(tempfile.gettempdir()) / "rsi_legacy_mm")))
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)


def load(game, method):
    from benchmark.config import load_config
    return load_config(ROOT / f"configs/{game}_{method}.json")


def paths(game, method):
    cfg = load(game, method)
    out = Path(cfg.output_dir)
    if not out.is_relative_to(ROOT / "runs"):
        raise RuntimeError("Output must be inside this version's runs/")
    return cfg, out


def count(out, phase):
    return len(list((out / "checkpoints").glob(f"{phase}_*.json")))


def audit(out):
    """Observe complete records, image artifacts and known/unknown usage separately."""
    result = {"training_checkpoints": count(out, "train"),
              "holdout_checkpoints": count(out, "holdout"),
              "inflight": (out / "inflight.json").exists(),
              "checks": [], "warnings": [], "errors": []}
    from benchmark.resume_state import digest
    initial_path = out / "checkpoints/initial.json"
    previous = json.loads(initial_path.read_text()) if initial_path.is_file() else None
    for phase in ("train", "holdout"):
        checkpoints = sorted((out / "checkpoints").glob(f"{phase}_*.json"))
        for ordinal, checkpoint in enumerate(checkpoints, 1):
            if checkpoint.name != f"{phase}_{ordinal:04d}.json":
                result["errors"].append(f"Noncontiguous checkpoint: {checkpoint}")
            cp = json.loads(checkpoint.read_text())
            if (previous is None or cp.get("parent_digest") != digest(previous)
                    or cp.get("phase") != phase or cp.get("ordinal") != ordinal
                    or cp.get("config_hash") != previous.get("config_hash")):
                result["errors"].append(f"Invalid checkpoint chain: {checkpoint.name}")
            previous = cp
            candidates = list((out / "runs" / phase).glob(f"{ordinal:04d}_*.json"))
            if len(candidates) != 1:
                result["errors"].append(f"Expected one record for {checkpoint.name}")
                continue
            record = json.loads(candidates[0].read_text())
            score = record.get("relative_score")
            if (record.get("status") != "ok" or not isinstance(score, (int, float))
                    or isinstance(score, bool) or not math.isfinite(score)
                    or record.get("run_id") != cp.get("run_id")
                    or digest(record) != cp.get("record_digest")):
                result["errors"].append(f"Invalid score/status/digest: {checkpoint.name}")
            steps = record.get("interactions", [])
            if not steps or len(steps) > 100:
                result["errors"].append(f"Unexpected interaction count {len(steps)}: {checkpoint.name}")
            for step in steps:
                images = step.get("image_records") or []
                if not images or any(not Path(x["path"]).is_file() for x in images):
                    result["errors"].append(f"Missing step image: {checkpoint.name}/{step.get('step_index')}")
            if not record.get("token_coverage", {}).get("complete"):
                result["warnings"].append(f"Unknown token component: {checkpoint.name} (NOT zero)")
            if phase == "train" and (record.get("selection") or {}).get("status") not in {
                    "committed", "completed", "unchanged"}:
                result["errors"].append(f"Invalid selection status: {checkpoint.name}")
            result["checks"].append({
                "checkpoint": checkpoint.name, "score": score, "steps": len(steps),
                "total_tokens": record.get("total_tokens"),
                "total_seconds": record.get("total_seconds"),
                "inference_tokens": record.get("inference_tokens"),
                "refinement_tokens": record.get("refinement_tokens"),
                "candidate_validation_tokens": record.get("candidate_validation_tokens"),
            })
    return result


def preflight():
    from benchmark.tasks import build_manifest
    from benchmark.visual_evidence import refinement_image_limit
    verify_release()
    data = {"scope": "offline configuration/import checks, NOT live API/game acceptance",
            "model": "gpt-5.6-sol", "reasoning_effort": "low",
            "max_refinement_images": refinement_image_limit() or "all",
            "python": sys.version, "configs": [], "dependencies": {},
            "node": shutil.which("node"), "paid_calls": 0}
    for game in ("minecraft", "astray"):
        for method in ("ace", "hyperagents", "ahe"):
            cfg, out = paths(game, method)
            manifest = build_manifest(cfg)
            assert len(manifest.training) == 20 and len(manifest.holdout) == 5
            data["configs"].append({"game": game, "method": method,
                                    "train": 20, "holdout": 5, "output": str(out)})
    for name in ("nexau", "openai", "docker", "GitPython", "playwright", "PyYAML", "Pillow"):
        data["dependencies"][name] = importlib.metadata.version(name)
    from rsi_methods.ACE.ACEAdaptor import ACEAdaptor
    from rsi_methods.HyperAgents.HyperAgentsAdaptor import HyperAgentsAdaptor
    from rsi_methods.AHE.AHEAdaptor import AHEAdaptor
    data["adaptor_imports"] = [c.__name__ for c in (ACEAdaptor, HyperAgentsAdaptor, AHEAdaptor)]
    write_json(ROOT / "reports/preflight.json", data)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def worker(cfg, out, phase):
    from benchmark.runner import run_benchmark
    # AHE candidate actors get the same step timeout as the parent actor.
    os.environ["RSI_AHE_ACTOR_TIMEOUT"] = os.environ.get(
        "RSI_AHE_STEP_TIMEOUT", str(getattr(getattr(cfg, "agent", None), "timeout", 300)))
    report = run_benchmark(
        cfg, resume=(out / "checkpoints/initial.json").is_file(),
        max_new_train_runs=1 if phase == "train" else None)
    if report["summary"]["training_errors"] or report["summary"]["holdout_errors"]:
        raise RuntimeError("Legacy runner reported a failure; inspect retained logs before retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "test", "status", "train", "holdout"))
    parser.add_argument("game", nargs="?", choices=("minecraft", "astray"))
    parser.add_argument("method", nargs="?", choices=("ace", "hyperagents", "ahe"))
    parser.add_argument("--rounds", type=int, default=1, help="New training rounds this invocation; default one pilot")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--retry-inflight", action="store_true",
                        help="Explicitly permit restarting an uncommitted whole round (possibly billed)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_environment(f"{args.game}_{args.method}" if args.game and args.method else None)
    if args.command == "test":
        tests = [
            "tests", "code/benchmark/test_ahe_refinement_frequency.py",
            "code/benchmark/test_ahe_xbr_visual.py", "code/benchmark/test_token_ledger.py",
            "code/benchmark/test_resume.py", "code/benchmark/test_metrics_missing.py",
            "code/benches/gameworld/test_network_retry.py",
            "code/rsi_methods/HyperAgents/test_screenshot_evidence.py",
            "code/rsi_methods/HyperAgents/test_runtime_safety.py",
        ]
        raise SystemExit(subprocess.call([
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests,
            "--junitxml=reports/offline.xml"], cwd=ROOT))
    if args.command == "preflight":
        preflight()
        return
    if not args.game or not args.method:
        parser.error("Specify game and method")
    if args.method == "hyperagents":
        # The default daemon keeps its data on the small in-memory /tmp, which
        # filled up mid-run; this project's daemon keeps it on /data. Its
        # containers have no bridge network and reach the model via the host.
        explicit_socket = os.environ.get("RSI_HYPERAGENTS_DOCKER_SOCK")
        if explicit_socket or "DOCKER_HOST" not in os.environ:
            sock = Path(explicit_socket or "/var/run/docker.sock")
            if args.command in ("train", "holdout") and not sock.is_socket():
                parser.error(f"Docker daemon socket missing: {sock}; see README.md")
            os.environ["DOCKER_HOST"] = f"unix://{sock}"
        os.environ["HYPERAGENTS_DOCKER_NETWORK"] = "host"
    cfg, out = paths(args.game, args.method)
    if args.command == "status":
        print(json.dumps(audit(out), ensure_ascii=False, indent=2))
        return
    verify_release()
    if not args.allow_paid:
        parser.error("No API calls started. Paid commands require --allow-paid.")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        parser.error("Set OPENAI_API_KEY and OPENAI_BASE_URL in environment; never store the key in configs")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.command == "holdout" and count(out, "train") != 20:
        parser.error("Holdout requires the fully committed 20-round agent")
    if args.command == "train" and count(out, "train") >= 20:
        parser.error("20 rounds already committed. Use holdout in a fresh process.")
    if (out / "inflight.json").exists() and not args.retry_inflight:
        parser.error("Uncommitted attempt exists. Inspect status/logs; --retry-inflight repeats that round, NOT a saved game step.")
    if args.worker:
        worker(cfg, out, args.command)
        return
    # Do not mix global relative GameWorld output with another version/run.
    os.environ["GAMEWORLD_RESULTS_ROOT"] = str(out.parent / "episodes")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_time = datetime.datetime.now().astimezone().isoformat()
        rounds = min(args.rounds, 20 - count(out, "train")) if args.command == "train" else 1
        for _ in range(rounds):
            index = count(out, "train") + 1 if args.command == "train" else count(out, "holdout") + 1
            stamp = time.time_ns()
            logfile = out.parent / f"{args.command}_{index:04d}_{stamp}.log"
            command = [sys.executable, str(Path(__file__).resolve()), args.command,
                       args.game, args.method, "--worker", "--allow-paid"]
            if args.retry_inflight:
                command.append("--retry-inflight")
            started = time.monotonic()
            print(f"Starting {args.game}/{args.method} {args.command} {index}; log={logfile}", flush=True)
            with logfile.open("w") as log:
                process = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
            receipt = audit(out)
            receipt.update(command=args.command, game=args.game, method=args.method,
                           launched_at=launch_time, wall_seconds=time.monotonic() - started,
                           exit_code=process.returncode, log=str(logfile),
                           model=cfg.agent.model, reasoning_effort=cfg.agent.reasoning_effort,
                           base_url=os.environ["OPENAI_BASE_URL"],
                           max_refinement_images=os.environ["RSI_MAX_REFINEMENT_IMAGES"],
                           release_manifest_sha256=hashlib.sha256(
                               (ROOT / "provenance/release_sha256.json").read_bytes()).hexdigest())
            write_json(out.parent / f"audit_{args.command}_{index:04d}_{stamp}.json", receipt)
            print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
            if process.returncode or receipt["errors"] or receipt["inflight"]:
                raise RuntimeError(f"Run/audit failed; artifacts retained, no automatic whole-round paid retry. See {logfile}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
