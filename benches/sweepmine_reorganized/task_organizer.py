from . import core_generator, map_generator, map_renderer

def get_task(
    dif: int, task_counts: int, seed: int, *,
    core_mines: int = 4, example_counts: int = 5,
):
    initial_content = (
        "This is a Minesweeper-like rule-inference task. "
        "Every example and task uses the same hidden layered rule to inspect "
        "the original mine locations around each point. The rule may differ "
        "from standard Minesweeper. "
        "Infer it from the complete example boards, then replace every '?' "
        "in each task board with the correct integer. Keep every 'M' unchanged.\n\n"
    )
    prompt=initial_content
    core=core_generator.generate_core(difficulty=dif,active_count=core_mines,seed=seed)

    maps = [map_generator._map_generator(dif=dif,seed=seed+i) for i in range(example_counts)]
    rendered_maps=[map_renderer.render_map(map=m,core=core) for m in maps]
    prompt+="Your Examples:\n"
    for k in range(len(rendered_maps)):
        prompt+=f"Example{k+1}:\n"
        for i in range(len(rendered_maps[k])):
            for j in rendered_maps[k][i]:
                prompt+=str(j)
                prompt+=" "
            prompt+="\n"
    prompt+="\n"
    #print(prompt)
    task_maps = [map_generator._map_generator(dif=dif,seed=seed+example_counts+i) for i in range(task_counts)]
    rendered_task_maps = [map_renderer.render_map(map=m,core=core) for m in task_maps]
    prompt+="Your Tasks:\n"
    for k in range(len(rendered_task_maps)):
        prompt+=f"Task{k+1}:\n"
        for i in range(len(rendered_task_maps[k])):
            for j in rendered_task_maps[k][i]:
                prompt+="M" if (j=="M") else "?"
                prompt+=" "
            prompt+="\n"
    prompt+=(
        "\nReturn only one valid JSON object with an 'answers' array. "
        "Include exactly one object per task, in task order. Each object must "
        "contain 'task_id' (the 1-based Task number), 'board' (a 2D array "
        "with the same dimensions as the task; use integer numbers and the "
        "string 'M'), and 'reasoning' (one or two short sentences explaining "
        "the inferred rule). Prioritize completing the JSON answer; do not "
        "list per-cell calculations. Do not include '?' or Markdown.\n"
    )
    details = {
        "examples": [
            {"task_id": index, "board": board}
            for index, board in enumerate(rendered_maps, start=1)
        ],
        "queries": [
            {"task_id": index, "board": board}
            for index, board in enumerate(rendered_task_maps, start=1)
        ],
    }
    return prompt, rendered_task_maps, details

