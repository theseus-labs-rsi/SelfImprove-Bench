# Supported task-tool catalog

`tools.json` contains an `enabled` array. Only the following DeepSeek Harness
tool row ids are accepted; every other listed row is forcibly disabled by the
runtime overlay:

- `tool-fs`
- `tool-fs-search`
- `tool-todo`
- `tool-goal`
- `tool-result-pruner`
- `tool-skill`

The task workspace is read-only. Shell, subagent, workflow, web, editing and
background-job tools are always disabled because benchmark prompts must remain
self-contained and candidate evaluation must not acquire external side effects.
