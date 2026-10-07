"""CLI entry point for the fixed HyperDSH refinement MetaAgent."""

from __future__ import annotations

import argparse
import os

from meta_agent import MetaAgent
from utils.git_utils import diff_versus_commit, reset_paths_to_commit


_FIXED_PATHS = [
    ".gitignore",
    "README.md",
    "Dockerfile",
    "Dockerfile.runtime",
    "requirements.txt",
    "agent",
    "benchmark",
    "meta_agent.py",
    "run_meta_agent.py",
    "task_agent.py",
    "utils",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--chat_history_file", required=True)
    parser.add_argument("--reasoning_effort")
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--api_mode", default="responses")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--base_url_env", default="OPENAI_BASE_URL")
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--evals_folder", required=True)
    parser.add_argument("--iterations_left", type=int)
    parser.add_argument("--git_dir", required=True)
    parser.add_argument("--base_commit", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    MetaAgent(
        model=args.model,
        chat_history_file=args.chat_history_file,
        reasoning_effort=args.reasoning_effort,
        backend=args.backend,
        api_mode=args.api_mode,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
    ).forward(
        repo_path=args.repo_path,
        eval_path=args.evals_folder,
        iterations_left=args.iterations_left,
    )

    # The model is instructed to stay in harness/. This reset is a second,
    # deterministic boundary before the portable patch is produced.
    reset_paths_to_commit(args.git_dir, args.base_commit, _FIXED_PATHS)
    patch = diff_versus_commit(args.git_dir, args.base_commit)
    output = os.path.join(args.outdir, "model_patch.diff")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(patch)


if __name__ == "__main__":
    main()
