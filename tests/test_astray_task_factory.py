from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import json
import io
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from urllib.parse import parse_qs

import yaml

from benches.gameworld.astray_task_factory import (
    add_cases_to_archive, generate_batch, generate_task, materialize_task,
)
from benches.gameworld.game import generate
from benchmark.config import ReplayConfig, config_from_mapping, load_config
from benchmark.tasks import build_manifest

ROOT = Path(__file__).resolve().parents[1]


class AstrayTaskFactoryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for browser-script tests")
    def test_browser_generator_and_bfs_progress(self):
        result = subprocess.run(
            ["node", str(ROOT / "tests/test_astray_browser.js")],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tiers_and_reproducibility(self):
        for difficulty in range(1, 6):
            task = generate_task(difficulty, seed=123, task_id="example")
            self.assertEqual(task, generate_task(difficulty, seed=123, task_id="example"))
            query = parse_qs(task["game_url_suffix"][1:])
            self.assertEqual(query, {"level": [str(difficulty)], "train_seed": ["123"]})
            dimension = 2 * difficulty + 9
            self.assertIn(f"{dimension}x{dimension}", task["task_prompt"])
            self.assertEqual(task["max_steps"], 100)
            self.assertEqual(task["evaluator_config"]["score_field"],
                             "game_state.completion_progress")

    def test_invalid_inputs(self):
        for difficulty in (0, 6, True, 3.0):
            with self.assertRaises(ValueError):
                generate_task(difficulty, seed=1, task_id="example")
        for seed in (-1, 2**32, True):
            with self.assertRaises(ValueError):
                generate_task(3, seed=seed, task_id="example")
        with self.assertRaises(ValueError):
            generate_task(3, seed=1, task_id="../escape")

    def test_batch_default_d3_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_batch(output_dir=directory)
            self.assertEqual(len(paths), 20)
            seeds = []
            for path in paths:
                task = yaml.safe_load(path.read_text())
                self.assertEqual(path.stem, task["task_id"])
                query = parse_qs(task["game_url_suffix"][1:])
                self.assertEqual(query["level"], ["3"])
                seeds.append(int(query["train_seed"][0]))
            self.assertEqual(seeds, list(range(20260923, 20260943)))
            contents = paths[0].read_bytes()
            with self.assertRaises(FileExistsError):
                generate_batch(output_dir=directory)
            self.assertEqual(paths[0].read_bytes(), contents)

    def test_legacy_catalog_tasks_can_be_reconstructed(self):
        directory = ROOT / "benches/gameworld/assets/tasks/train/astray"
        for path in directory.glob("*.yaml"):
            original = yaml.safe_load(path.read_text())
            query = parse_qs(original["game_url_suffix"][1:])
            generated = generate_task(3, seed=int(query["train_seed"][0]), task_id=path.stem)
            self.assertEqual(generated, original)

    def test_runtime_generation_and_materialization(self):
        options = {"train_generator": "astray", "max_steps": 100}
        task = generate(3, seed=2**40 + 123, task_id="train-example", options=options)
        self.assertEqual(task["generated_task"], generate_task(3, seed=123, task_id="train-example"))
        changed = generate(4, seed=2**40 + 123, task_id="train-example", options=options)
        self.assertNotEqual(task["generated_task"], changed["generated_task"])
        with tempfile.TemporaryDirectory() as directory:
            root = materialize_task(task, directory)
            path = root / "astray/train-example.yaml"
            self.assertEqual(yaml.safe_load(path.read_text()), task["generated_task"])
            self.assertEqual(materialize_task(task, directory), root)
            path.write_text("task_id: tampered\n")
            with self.assertRaises(ValueError):
                materialize_task(task, directory)

    def test_main_training_is_generated_d3(self):
        maps = []
        for method in ("ace", "ahe", "hyperagents"):
            config = load_config(ROOT / f"configs/main/astray_{method}.json")
            self.assertEqual(config.seed, 20260923)
            self.assertEqual(config.holdout_seed, 100)
            manifest = build_manifest(config)
            self.assertEqual(manifest, build_manifest(config))
            normal = [t for t in manifest.training if not t.replay]
            self.assertEqual(len(normal), 20)
            urls = [t.task["generated_task"]["game_url_suffix"] for t in normal]
            self.assertEqual(len(set(urls)), 20)
            self.assertTrue(all(parse_qs(url[1:])["level"] == ["3"] for url in urls))
            maps.append(urls)
            for task in manifest.training:
                if task.replay:
                    original = next(t for t in normal if t.run_id == task.replay_of)
                    self.assertEqual(task.task, original.task)
            self.assertTrue(all("generated_task" not in t.task for t in manifest.holdout))
            initial = build_manifest(replace(config, schedule=(), replay=ReplayConfig()))
            self.assertEqual(initial.holdout, manifest.holdout)
            restored = type(manifest).from_mapping(manifest.as_dict())
            self.assertEqual(restored, manifest)
        self.assertEqual(maps[0], maps[1])
        self.assertEqual(maps[0], maps[2])

    def test_config_rejects_ambiguous_generator_and_bad_difficulty(self):
        raw = json.loads((ROOT / "configs/main/astray_ace.json").read_text())
        for generator in ("unknown", False, {}):
            changed = deepcopy(raw)
            changed["game_options"]["gameworld"]["train_generator"] = generator
            with self.assertRaises(ValueError):
                config_from_mapping(changed)
        changed = deepcopy(raw)
        changed["game_options"]["gameworld"]["train"] = [{"game": "03_astray", "task": "03_t01"}]
        with self.assertRaises(ValueError):
            config_from_mapping(changed)
        changed = deepcopy(raw)
        changed["schedule"][0]["difficulty"] = 6
        with self.assertRaises(ValueError):
            config_from_mapping(changed)

    def test_case_archive_preserves_originals_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "case.tar.gz"
            originals = {
                "case/farm/farm_d3_001.json": b'{"difficulty": 3}\n',
                "case/astray/03_01.json": b'{"task_id": "03_01"}\n',
            }
            with tarfile.open(archive, "w:gz") as stream:
                for name, data in originals.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    stream.addfile(member, io.BytesIO(data))
            added = add_cases_to_archive(archive)
            self.assertEqual(len(added), 100)
            with tarfile.open(archive) as stream:
                for name, data in originals.items():
                    self.assertEqual(stream.extractfile(name).read(), data)
                for difficulty in range(1, 6):
                    for index in range(20):
                        task_id = f"astray_d{difficulty}_{index + 1:03d}"
                        case = json.load(stream.extractfile(f"case/astray/{task_id}.json"))
                        seed = 20260923 + (difficulty - 1) * 20 + index
                        self.assertEqual(case.pop("difficulty"), difficulty)
                        self.assertEqual(case.pop("seed"), seed)
                        self.assertEqual(case, generate_task(difficulty, seed=seed, task_id=task_id))
            contents = archive.read_bytes()
            self.assertEqual(add_cases_to_archive(archive), [])
            self.assertEqual(archive.read_bytes(), contents)
            with self.assertRaises(ValueError):
                add_cases_to_archive(archive, seed=123)
            self.assertEqual(archive.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
