"""Select raw game screenshots without an intermediate model call."""

import json
from pathlib import Path


def collect_game_screenshots(eval_path):
    root = Path(eval_path).resolve()
    result_path = root / "result.json"
    if not result_path.is_file():
        return [], ""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    references = []
    for case_index, case in enumerate(result.get("cases", [])):
        for step_index, step in enumerate(
            (case.get("feedback") or {}).get("trajectory") or []
        ):
            for image in step.get("image_records") or []:
                path = (root / image["path"]).resolve()
                if root not in path.parents or not path.is_file():
                    raise FileNotFoundError(
                        f"Staged evaluation screenshot is inaccessible: {path}"
                    )
                references.append((
                    path,
                    case.get("case_id", case_index),
                    step.get("step_index", step_index),
                ))
    limit_path = Path(__file__).resolve().parent / "visual_evidence_limit.json"
    limit = (
        json.loads(limit_path.read_text())["max_refinement_images"]
        if limit_path.is_file() else 8
    )
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("max_refinement_images must be a nonnegative integer or null")
    if limit == 0:
        references = []
    elif limit is not None and len(references) > limit:
        references = (
            [references[-1]] if limit == 1 else
            [references[round(i * (len(references) - 1) / (limit - 1))]
             for i in range(limit)]
        )
    if not references:
        return [], ""
    context = [
        f"Evaluation records: {result_path}",
        "Attached images are observations before the corresponding actions:",
    ]
    context.extend(
        f"Image {index}: case_id={case_id}, step={step}, path={path}"
        for index, (path, case_id, step) in enumerate(references, 1)
    )
    return [str(item[0]) for item in references], "\n".join(context)
