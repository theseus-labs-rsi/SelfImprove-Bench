"""Offline tests for selected XBR bounds while retaining visual GameWorld accounting."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from benchmark import resume_state
from benchmark.rsi_session import RSIBatchSession, refinement_token_metrics
from rsi_methods.AHE.AHEAdaptor import AHEAdaptor
from rsi_methods.AHE.agent import make_nexau_message, compact_prior_game_turns, compact_prior_screenshots
from rsi_methods.AHE.init import apply_agent_limits


class AHEVisualPolicyTests(unittest.TestCase):
    def test_initial_limits_and_multimodal_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = AHEAdaptor(run_dir=Path(tmp) / "ahe")
            state.init()  # Copies source YAML; no NexAU process or API call.
            code = yaml.safe_load((state.run_dir / "workspace" / "code_agent.yaml").read_text())
            evolve = yaml.safe_load((state.run_dir / "evolve_agent" / "evolve_agent.yaml").read_text())
            self.assertEqual((code["max_context_tokens"], code["max_iterations"], code["llm_config"]["max_tokens"]),
                             (100_000, 16, 4_096))
            self.assertEqual((evolve["max_context_tokens"], evolve["max_iterations"], evolve["llm_config"]["max_tokens"]),
                             (120_000, 64, 8_192))
            image = Path(tmp) / "frame.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            message = make_nexau_message("Game screen:", [image])
            parts = message[0].metadata["response_items"][0]["content"]
            self.assertEqual([part["type"] for part in parts], ["input_text", "input_image"])
            self.assertTrue(parts[-1]["image_url"].startswith("data:image/png;base64,"))
            state.close()

    def test_visual_history_stays_bounded_without_discarding_action_trace(self):
        from nexau.core.messages import Message
        from nexau.archs.main_sub.history_list import HistoryList
        from nexau.archs.main_sub.execution.llm_caller import _prepare_responses_api_input
        from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
        history = HistoryList()
        history.append(Message.user("prelude that is not a game turn"))
        agent = SimpleNamespace(history=history)
        for step in range(60):
            compact_prior_game_turns(agent, keep_recent=8)
            compact_prior_screenshots(agent)
            frame = Message.user("# Current Observation\nGame screen:")
            frame.metadata["response_items"] = [{
                "type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": f"# Current Observation\nGame screen: step {step}"},
                    {"type": "input_image", "image_url": "data:image/png;base64,aA=="},
                ],
            }]
            history.append(frame)
            history.append(Message.assistant(f"action {step}"))
            items, _ = _prepare_responses_api_input(messages_to_legacy_openai_chat(
                list(history), tool_image_policy="embed_in_tool_message"))
            images = sum(part.get("type") == "input_image" for msg in items
                         for part in msg.get("content", []) if isinstance(part, dict))
            self.assertLessEqual(images, 9)
            self.assertLessEqual(len(history), 19)
        self.assertIn("prelude", history[0].content[0].text)
        self.assertIn("step 51", str(history[1].metadata["response_items"]))

    def test_refine_every_five_resets_runtime_and_records_real_zero(self):
        class FakeAgent:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            ahe = AHEAdaptor(run_dir=Path(tmp) / "ahe")
            ahe.init()
            agents = []
            def make_agent(*args, **kwargs):
                agent = FakeAgent()
                agents.append(agent)
                return agent
            def evolve(*args, **kwargs):
                Path(kwargs["output_dir"], "ahe_refine.json").write_text(json.dumps({
                    "token_usage_report": {"complete": True, "total_tokens": 11,
                                           "prompt_num_tokens": 8, "response_num_tokens": 3},
                }))
                return SimpleNamespace(candidates=())
            with patch("rsi_methods.AHE.AHEAdaptor.build_agent", side_effect=make_agent), \
                 patch("rsi_methods.AHE.AHEAdaptor.generate_candidates", side_effect=evolve) as generate:
                for turn in range(1, 6):
                    ahe.load()
                    if turn < 5:
                        self.assertIsNone(ahe.refine([SimpleNamespace(task_id="t")], output_dir=tmp))
                        self.assertTrue(agents[-1].closed)
                        self.assertEqual(refinement_token_metrics(ahe.last_refinement_report)["total_tokens"], 0)
                # New task never inherits the previous NexAU runtime/history.
                self.assertEqual(len(agents), 5)
                self.assertEqual(ahe._observed_training_tasks, 4)
                ahe.refine([SimpleNamespace(task_id="t")], output_dir=tmp)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(ahe._refinement_round, 1)
            ahe.close()

    def test_checkpoint_round_trips_counter_and_rejects_old_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ahe = AHEAdaptor(run_dir=base / "rsi" / "ahe" / "batch" / "state")
            session = RSIBatchSession("ahe", base / "rsi", task_count=20, batch_id="batch", adaptor=ahe)
            ahe._observed_training_tasks = 4
            checkpoint = resume_state.export(session)
            fresh = AHEAdaptor(run_dir=ahe.run_dir)
            recovered = RSIBatchSession("ahe", base / "rsi", task_count=20, batch_id="batch",
                                        adaptor=fresh, resume=True)
            resume_state.restore(recovered, checkpoint)
            self.assertEqual(fresh._observed_training_tasks, 4)
            checkpoint.pop("ahe_policy")
            with self.assertRaisesRegex(RuntimeError, "predates or differs"):
                resume_state.restore(recovered, checkpoint)
            recovered.close()
            session.close()

    def test_invalid_limits_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "agent.yaml"
            config.write_text("max_iterations: 300\nllm_config: {max_tokens: 32000}\n")
            with self.assertRaises(ValueError):
                apply_agent_limits(config, {"max_context_tokens": True, "max_iterations": 16, "max_tokens": 4096})


if __name__ == "__main__":
    unittest.main()
