import hashlib
import json
import math
import os
import random
import shutil
import stat

import git

from .result_store import get_selection_score
from .hyperagents.utils.constants import REPO_NAME
from .hyperagents.utils.docker_utils import copy_to_container, log_container_output
from .hyperagents.utils.git_utils import commit_repo, get_git_commit_hash


_OUTER_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AGENT_SOURCE = os.path.join(_OUTER_DIR, "hyperagents")
RUNTIME_IMAGE_NAME = "hyperagents-runtime:py312-v2"
_RUNTIME_DOCKERFILE = """FROM python:3.12-bookworm

WORKDIR /hyperagents

RUN pip install --no-cache-dir \\
    requests==2.32.4 \\
    python-dotenv==1.2.3 \\
    backoff==2.2.1 \\
    openai==3.2.0 \\
    GitPython==3.1.44

CMD ["tail", "-f", "/dev/null"]
"""
_GENERIC_DOCKERFILE = f"""FROM {RUNTIME_IMAGE_NAME}

WORKDIR /hyperagents
COPY . .

CMD ["tail", "-f", "/dev/null"]
"""
_INITIAL_README = """# Self-Improving HyperAgent

This repository contains the mutable MetaAgent and TaskAgent implementation.
Previous generations and their evaluation results are available through the
evaluation-history path passed to the MetaAgent. Each generation stores a
`result.json` containing `valid`, `selection_score`, task-specific `metrics`,
`feedback`, and optional `artifacts`.
"""
_SEED_META_INSTRUCTION = (
    '        instruction = f"Modify any part of the codebase at `{repo_path}`."'
)
_GENERIC_META_INSTRUCTION = '''        instruction = f"""Modify any part of the codebase at `{repo_path}`.
Recursively inspect evaluation evidence at `{eval_path}`. Read
`evidence_manifest.json` when present, then `result.json` and relevant artifacts.
Artifact paths in `result.json` are relative to `{eval_path}`. Verify referenced
files and ground failure diagnoses in concrete actions and observations.
A batch result contains aggregate metrics and a `cases` list. Inspect the cases
and artifacts, identify recurring failures, and do not optimize for one case ID
or stochastic trace.
There are `{iterations_left}` self-improvement iterations remaining after this call."""'''


def _copy_framework_backend(root_dir):
    """Snapshot benchmark contracts and backend into the candidate repository."""
    code_root = os.path.abspath(os.path.join(_OUTER_DIR, "..", ".."))
    source_package = os.path.join(code_root, "benchmark")
    source_backend = os.path.join(source_package, "backend")
    destination_package = os.path.join(root_dir, "benchmark")
    destination_backend = os.path.join(destination_package, "backend")
    os.makedirs(destination_backend, exist_ok=True)

    for filename in ("__init__.py", "Backend.py", "OpenAIBackend.py"):
        source = os.path.join(source_backend, filename)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Shared backend file not found: {source}")
        shutil.copy2(source, os.path.join(destination_backend, filename))

    contracts_source = os.path.join(source_package, "contracts.py")
    if not os.path.isfile(contracts_source):
        raise FileNotFoundError(f"Shared contracts file not found: {contracts_source}")
    shutil.copy2(contracts_source, os.path.join(destination_package, "contracts.py"))

    for package_dir in (destination_package, destination_backend):
        init_path = os.path.join(package_dir, "__init__.py")
        open(init_path, "a", encoding="utf-8").close()


def is_starting_node(generation_id):
    """Return whether a generation is one of the supported starting nodes."""
    return generation_id == "initial" or generation_id == 0


def _load_node_metadata(output_dir, generation_id):
    metadata_path = os.path.join(
        output_dir, f"gen_{generation_id}", "metadata.json"
    )
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def get_parent_genid(output_dir, generation_id):
    metadata = _load_node_metadata(output_dir, generation_id)
    return None if metadata is None else metadata.get("parent_genid")


def get_node_metadata_key(output_dir, generation_id, key):
    metadata = _load_node_metadata(output_dir, generation_id)
    return None if metadata is None else metadata.get(key)


def update_and_save_archive(output_dir, archive, new_node):
    """Append one valid node and persist the resulting archive snapshot."""
    archive.append(new_node)
    archive_path = os.path.join(output_dir, "archive.jsonl")
    with open(archive_path, "a", encoding="utf-8") as archive_file:
        archive_file.write(
            json.dumps(
                {"current_genid": new_node, "archive": archive},
                ensure_ascii=False,
            )
            + "\n"
        )
    return archive


def load_archive_data(filepath, last_only=True):
    """Load archive snapshots from the JSONL control-state file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Archive file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as archive_file:
        archive_data = [json.loads(line) for line in archive_file if line.strip()]
    if not archive_data:
        raise ValueError(f"Archive file is empty: {filepath}")
    return archive_data[-1] if last_only else archive_data


def _filter_patch_by_files(patch_text, excluded_files):
    """Remove complete Git diff blocks targeting excluded candidate files."""
    filtered_lines = []
    include_block = False
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            include_block = not any(
                f"a/{target}" in line and f"b/{target}" in line
                for target in excluded_files
            )
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines) + ("\n" if filtered_lines else "")


def process_meta_patch_files(
    meta_patch_files,
    output_dir,
    reset_task_agent=False,
    reset_meta_agent=False,
):
    """Create portable initial patches after applying optional reset controls."""
    processed_paths = []
    patch_dir = os.path.join(output_dir, "meta_patch_files")
    os.makedirs(patch_dir, exist_ok=True)
    for index, patch_file in enumerate(meta_patch_files or []):
        with open(patch_file, "r", encoding="utf-8") as source_file:
            patch_text = source_file.read()
        if reset_task_agent:
            patch_text = _filter_patch_by_files(patch_text, ["task_agent.py"])
        if reset_meta_agent:
            patch_text = _filter_patch_by_files(patch_text, ["meta_agent.py"])
        processed_path = os.path.join(patch_dir, f"model_patch_{index}.diff")
        with open(processed_path, "w", encoding="utf-8") as output_file:
            output_file.write(patch_text)
        processed_paths.append(processed_path)
    return processed_paths


def _write_run_config(output_dir, run_config):
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, ensure_ascii=False, indent=2)


def _validate_agent_source(source_dir):
    required_files = ("meta_agent.py", "task_agent.py", "run_meta_agent.py")
    missing = [name for name in required_files if not os.path.isfile(os.path.join(source_dir, name))]
    if missing:
        raise FileNotFoundError(f"Initial HyperAgent source is missing: {missing}")


def _adapt_seed_meta_agent(root_dir):
    meta_agent_path = os.path.join(root_dir, "meta_agent.py")
    with open(meta_agent_path, "r", encoding="utf-8") as meta_agent_file:
        source = meta_agent_file.read()
    if _SEED_META_INSTRUCTION not in source:
        return
    source = source.replace(
        _SEED_META_INSTRUCTION,
        _GENERIC_META_INSTRUCTION,
        1,
    )
    with open(meta_agent_path, "w", encoding="utf-8") as meta_agent_file:
        meta_agent_file.write(source)


def _adapt_runtime_boundaries(root_dir):
    run_meta_agent_path = os.path.join(root_dir, "run_meta_agent.py")
    with open(run_meta_agent_path, "r", encoding="utf-8") as run_meta_file:
        source = run_meta_file.read()
    source = source.replace(
        'paths=["domains/"]',
        'paths=["Dockerfile", "Dockerfile.runtime", "requirements.txt"]',
        1,
    )
    with open(run_meta_agent_path, "w", encoding="utf-8") as run_meta_file:
        run_meta_file.write(source)


_VENDORED_IMAGE_SAMPLE = (
    "        if len(references) > 8:\n"
    "            references = [references[round(i * (len(references) - 1) / 7)] for i in range(8)]\n"
)
_CONFIGURED_IMAGE_SAMPLE = (
    "        limit_path = Path(__file__).resolve().parent / \"visual_evidence_limit.json\"\n"
    "        limit = json.loads(limit_path.read_text())[\"max_refinement_images\"] if limit_path.is_file() else 8\n"
    "        if limit is not None and len(references) > limit:\n"
    "            references = ([references[-1]] if limit == 1 else\n"
    "                          [references[round(i * (len(references) - 1) / (limit - 1))] for i in range(limit)])\n"
)
_SCREENSHOT_PROMPT = "These are ordered screenshots from an agent's completed game evaluation. "
_SCREENSHOT_TIMING = "Each image is the observation shown to the agent before that step's action. "


def _install_visual_evidence_hook(root_dir):
    """Add parent-versioned visual support when the pinned upstream lacks it."""
    from benchmark.visual_evidence import refinement_image_limit
    helper_source = os.path.join(_OUTER_DIR, "visual_evidence.py")
    shutil.copy2(helper_source, os.path.join(root_dir, "visual_evidence.py"))
    # The MetaAgent runs in Docker without the launcher's environment; the
    # shared image budget travels with the candidate repository instead.
    with open(os.path.join(root_dir, "visual_evidence_limit.json"), "w", encoding="utf-8") as stream:
        json.dump({"max_refinement_images": refinement_image_limit()}, stream)
    meta_path = os.path.join(root_dir, "meta_agent.py")
    with open(meta_path, encoding="utf-8") as stream:
        source = stream.read()
    # Local vendored sources may already carry the equivalent helper. Do not
    # double-call the visual model; still bundle the pinned parent-owned hook.
    if "visual_observations = self._inspect_game_screenshots(eval_path)" in source:
        if _CONFIGURED_IMAGE_SAMPLE in source:
            return
        if source.count(_VENDORED_IMAGE_SAMPLE) != 1 or source.count(_SCREENSHOT_PROMPT) != 1:
            raise RuntimeError("Cannot apply the shared image budget to this MetaAgent version")
        source = source.replace(_VENDORED_IMAGE_SAMPLE, _CONFIGURED_IMAGE_SAMPLE)
        source = source.replace(_SCREENSHOT_PROMPT, _SCREENSHOT_PROMPT + _SCREENSHOT_TIMING)
        with open(meta_path, "w", encoding="utf-8") as stream:
            stream.write(source)
        return
    if "visual_observations = inspect_game_screenshots(self, eval_path)" in source:
        return
    marker = "        new_msg_history = chat_with_agent("
    if source.count(marker) != 1:
        raise RuntimeError("Cannot install visual evidence hook in this MetaAgent version")
    insertion = r'''        from visual_evidence import inspect_game_screenshots
        visual_observations = inspect_game_screenshots(self, eval_path)
        if visual_observations:
            instruction += (
                "\nThese evaluation screenshots were inspected in a separate "
                "multimodal call. Visual observations (not ground truth) follow; "
                "check against the scored trajectory before editing code:\n"
                + visual_observations[:6000]
            )
'''
    with open(meta_path, "w", encoding="utf-8") as stream:
        stream.write(source.replace(marker, insertion + marker, 1))


def _remove_generated_tree(path):
    """Remove a generated candidate even when Git objects are read-only."""
    for current_root, dirnames, filenames in os.walk(path):
        for name in (*dirnames, *filenames):
            item = os.path.join(current_root, name)
            os.chmod(item, os.stat(item).st_mode | stat.S_IWRITE)
    shutil.rmtree(path)


def setup_initial_candidate(
    output_dir, copy_root_dir=None, resume=False, run_config=None
):
    """Create or restore the initial HyperAgent candidate repository."""
    root_dir = os.path.abspath(
        os.path.join(output_dir, f"gen_initial/{REPO_NAME}")
    )
    if resume:
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Initial candidate not found: {root_dir}")
        repo = git.Repo(root_dir)
        roots = repo.git.rev_list("--max-parents=0", "HEAD").splitlines()
        if not roots:
            raise RuntimeError("Failed to locate the candidate root commit.")
        return root_dir, roots[0]

    source_dir = os.path.abspath(copy_root_dir or _DEFAULT_AGENT_SOURCE)
    _validate_agent_source(source_dir)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(root_dir):
        _remove_generated_tree(root_dir)

    ignored = shutil.ignore_patterns(
        ".git", ".claude", "outputs", "analysis", "misc", "baselines",
        "domains", "__pycache__", "*.pyc", "*.png", "outputs_os*",
        "generate_loop.py", "select_next_parent.py", "ensemble.py",
        "run_task_agent.py", "run_select_next_parent.py", "run_ensemble.py",
        "gl_utils.py",
    )
    shutil.copytree(source_dir, root_dir, ignore=ignored)
    _copy_framework_backend(root_dir)
    _adapt_seed_meta_agent(root_dir)
    _adapt_runtime_boundaries(root_dir)
    _install_visual_evidence_hook(root_dir)
    from .runtime_safety import stage_runtime_safety
    stage_runtime_safety(root_dir)
    with open(os.path.join(root_dir, "Dockerfile"), "w", encoding="utf-8") as dockerfile:
        dockerfile.write(_GENERIC_DOCKERFILE)
    with open(
        os.path.join(root_dir, "Dockerfile.runtime"), "w", encoding="utf-8"
    ) as runtime_dockerfile:
        runtime_dockerfile.write(_RUNTIME_DOCKERFILE)
    with open(os.path.join(root_dir, "README.md"), "w", encoding="utf-8") as readme:
        readme.write(_INITIAL_README)

    _write_run_config(output_dir, run_config or {"schema_version": 1})

    git.Repo.init(root_dir)
    commit_result = commit_repo(
        root_dir,
        commit_message="Initialize HyperAgent candidate",
    )
    if commit_result is None:
        raise RuntimeError("Failed to initialize the candidate Git repository.")

    # commit_repo parses normal commit output, but a repository's first commit
    # includes the extra "(root-commit)" token. Read HEAD directly so callers
    # always receive the actual commit hash.
    root_commit = get_git_commit_hash(root_dir)
    if not root_commit:
        raise RuntimeError("Failed to read the initial candidate commit hash.")
    persisted_config = dict(run_config or {"schema_version": 1})
    persisted_config["root_commit"] = root_commit
    _write_run_config(output_dir, persisted_config)
    return root_dir, root_commit


def _run_checked_container_command(container, command, workdir, description):
    exec_result = container.exec_run(command, workdir=workdir)
    if exec_result.exit_code != 0:
        output = exec_result.output.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{description} failed with exit code {exec_result.exit_code}: {output}"
        )
    log_container_output(exec_result)
    return exec_result


def apply_diffs_checked(container, patch_files, repo_name=REPO_NAME):
    """Apply lineage patches atomically enough to reject partial reconstruction."""
    repo_path = f"/{repo_name}"
    for patch_file in patch_files or []:
        if not os.path.isfile(patch_file):
            raise FileNotFoundError(f"Lineage patch not found: {patch_file}")
        if os.path.getsize(patch_file) == 0:
            continue
        container_patch = f"{repo_path}/parent_patch.diff"
        try:
            # Preserve the exact Git diff bytes produced by the MetaAgent.
            copy_to_container(container, patch_file, container_patch)
            _run_checked_container_command(
                container,
                ["git", "apply", "--check", container_patch],
                repo_path,
                f"Patch validation for {patch_file}",
            )
            _run_checked_container_command(
                container,
                ["git", "apply", container_patch],
                repo_path,
                f"Patch application for {patch_file}",
            )
        finally:
            container.exec_run(["rm", "-f", container_patch], workdir=repo_path)

    _run_checked_container_command(
        container, ["git", "add", "--all"], repo_path, "Git staging"
    )
    status = _run_checked_container_command(
        container, ["git", "status", "--porcelain"], repo_path, "Git status"
    ).output.decode("utf-8", errors="replace").strip()
    if status:
        _run_checked_container_command(
            container,
            [
                "git", "-c", "user.name=user", "-c", "user.email=you@example.com",
                "commit", "-m", "Apply HyperAgent lineage patches",
            ],
            repo_path,
            "Git commit",
        )
    return _run_checked_container_command(
        container, ["git", "rev-parse", "HEAD"], repo_path, "Git revision lookup"
    ).output.decode("utf-8", errors="replace").strip()


def check_candidate_compilation(container, repo_name=REPO_NAME):
    """Check entry points required by evaluation and future refinement."""
    repo_path = f"/{repo_name}"
    checks = (
        (["python", "-c", "from meta_agent import MetaAgent"], "MetaAgent import"),
        (["python", "-c", "from task_agent import TaskAgent"], "TaskAgent import"),
        (["python", "-m", "py_compile", "run_meta_agent.py"], "MetaAgent runner"),
    )
    for command, description in checks:
        _run_checked_container_command(
            container, command, repo_path, f"{description} check"
        )


def describe_patch(patch_path, generation_dir):
    """Create a portable integrity record for one generated code patch."""
    if not os.path.isfile(patch_path):
        return None
    digest = hashlib.sha256()
    with open(patch_path, "rb") as patch_file:
        for chunk in iter(lambda: patch_file.read(65536), b""):
            digest.update(chunk)
    return {
        "path": os.path.relpath(patch_path, generation_dir),
        "size_bytes": os.path.getsize(patch_path),
        "sha256": digest.hexdigest(),
    }


def patch_paths_for_metadata(output_dir, patch_files):
    """Store lineage patches relative to the movable run directory."""
    output_root = os.path.abspath(output_dir)
    relative_paths = []
    for patch_file in patch_files or []:
        patch_path = os.path.abspath(patch_file)
        if os.path.commonpath([output_root, patch_path]) != output_root:
            raise ValueError(f"Patch is outside the run directory: {patch_file}")
        relative_paths.append(os.path.relpath(patch_path, output_root))
    return relative_paths


def get_patch_files(output_dir, generation_id):
    """Resolve one node's stored lineage patch paths for local use."""
    metadata_path = os.path.join(output_dir, f"gen_{generation_id}", "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Node metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    stored_paths = metadata.get("prev_patch_files", []) + metadata.get(
        "curr_patch_files", []
    )
    output_root = os.path.abspath(output_dir)
    resolved_paths = []
    for stored_path in stored_paths:
        patch_path = (
            stored_path
            if os.path.isabs(stored_path)
            else os.path.join(output_root, stored_path)
        )
        patch_path = os.path.abspath(patch_path)
        if os.path.commonpath([output_root, patch_path]) != output_root:
            raise ValueError(f"Stored patch leaves the run directory: {stored_path}")
        if not os.path.isfile(patch_path):
            raise FileNotFoundError(f"Stored patch not found: {patch_path}")
        resolved_paths.append(patch_path)
    return resolved_paths


def _candidate_scores(archive, output_dir):
    candidates = {}
    for generation_id in archive:
        valid_parent = get_node_metadata_key(
            output_dir, generation_id, "valid_parent"
        )
        if valid_parent is not True:
            continue
        score = get_selection_score(output_dir, generation_id)
        if score is not None:
            candidates[generation_id] = score
    return candidates


def _child_counts(archive, output_dir, candidates):
    counts = {generation_id: 0 for generation_id in candidates}
    for generation_id in archive:
        parent_id = get_parent_genid(output_dir, generation_id)
        if parent_id in counts:
            counts[parent_id] += 1
    return counts


def _score_weights(candidates):
    scores = list(candidates.values())
    midpoint = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
    return {
        generation_id: 1 / (1 + math.exp(-10 * (score - midpoint)))
        for generation_id, score in candidates.items()
    }


def select_parent_from_candidates(
    candidates, method="best", child_counts=None
):
    """Select a parent from an ID-to-selection-score mapping."""
    if not candidates:
        raise ValueError("No valid evaluated candidate is available for parent selection.")
    generation_ids = list(candidates)
    if method == "random":
        return random.choice(generation_ids)
    if method == "latest":
        return generation_ids[-1]
    if method == "best":
        return max(candidates, key=candidates.get)

    weights = _score_weights(candidates)
    if method == "score_child_prop":
        counts = child_counts or {generation_id: 0 for generation_id in candidates}
        weights = {
            generation_id: weight * math.exp(-((counts[generation_id] / 8) ** 3))
            for generation_id, weight in weights.items()
        }
    elif method != "score_prop":
        raise ValueError(f"Unknown parent selection method: {method}")
    return random.choices(
        generation_ids,
        weights=[weights[generation_id] for generation_id in generation_ids],
    )[0]


def select_parent(archive, output_dir, method="best"):
    """Select a parent using canonical result.selection_score values."""
    if not archive:
        raise ValueError("Cannot select a parent from an empty archive.")
    candidates = _candidate_scores(archive, output_dir)
    child_counts = (
        _child_counts(archive, output_dir, candidates)
        if method == "score_child_prop"
        else None
    )
    return select_parent_from_candidates(candidates, method, child_counts)
