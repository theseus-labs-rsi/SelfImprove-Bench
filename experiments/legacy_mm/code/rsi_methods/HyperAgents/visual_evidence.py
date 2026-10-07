"""Visual inspection hook copied into new HyperAgents candidate snapshots.

Kept outside the upstream gitlink so that rematerializing its pinned GitHub
commit does not silently remove the benchmark's visual-evidence bridge.
"""

import json
from pathlib import Path

from agent.llm_withtools import chat_with_agent


def inspect_game_screenshots(meta_agent, eval_path):
    """Send a bounded sample of staged evaluation images to the MetaAgent model."""
    root = Path(eval_path).resolve()
    result_path = root / 'result.json'
    if not result_path.is_file():
        return ''
    result = json.loads(result_path.read_text(encoding='utf-8'))
    references = []
    for case in result.get('cases', []):
        for step in (case.get('feedback') or {}).get('trajectory') or []:
            for image in step.get('image_records') or []:
                path = root / image['path']
                resolved = path.resolve()
                if root not in resolved.parents or not resolved.is_file():
                    raise FileNotFoundError(f'Staged evaluation screenshot is inaccessible: {path}')
                references.append((image['path'], resolved))
    if not references:
        return ''
    limit_path = Path(__file__).resolve().parent / 'visual_evidence_limit.json'
    limit = json.loads(limit_path.read_text())['max_refinement_images'] if limit_path.is_file() else 8
    if limit is not None and len(references) > limit:
        references = ([references[-1]] if limit == 1 else
                      [references[round(i * (len(references) - 1) / (limit - 1))] for i in range(limit)])
    prompt = (
        "These are ordered screenshots from an agent's completed game evaluation. "
        "Each image is the observation shown to the agent before that step's action. "
        "Describe the visible state and any objective-relevant change for each image "
        "in order. State uncertainty; do not invent hidden goals, scores, or actions. "
        "The corresponding evidence paths, in image order, are: "
        + ', '.join(name for name, _ in references)
    )
    history = chat_with_agent(
        prompt, model=meta_agent.model, msg_history=[], logging=meta_agent.log,
        tools_available=[], image_paths=[str(path) for _, path in references],
        reasoning_effort=meta_agent.reasoning_effort, backend=meta_agent.backend,
        api_mode=meta_agent.api_mode, api_key_env=meta_agent.api_key_env,
        base_url_env=meta_agent.base_url_env,
    )
    last = history[-1] if history else {}
    observation = last.get('content') or last.get('text') or ''
    if not isinstance(observation, str) or not observation.strip():
        raise RuntimeError('Screenshot inspection returned no visual observations')
    output = Path(meta_agent.logger_manager.log_file).parent / 'screenshot_analysis.json'
    output.write_text(json.dumps({
        'image_paths': [name for name, _ in references],
        'observation': observation, 'usage': last.get('usage'),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return observation
