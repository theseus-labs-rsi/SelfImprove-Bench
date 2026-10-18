# nexau tool implementations

## Directory layout (4 categories)

```
tools/
├── _sandbox_utils.py         # shared sandbox helpers
├── file_tools/               # file operations
│   ├── read_file.py, write_file.py, replace.py, apply_patch.py
│   ├── glob_tool.py, list_directory.py, read_many_files.py, search_file_content.py
├── shell_tools/              # shell
│   └── run_shell_command.py
├── web_tools/                # web
│   ├── google_web_search.py, web_fetch.py
└── session_tools/            # session/state
    ├── write_todos.py, complete_task.py, save_memory.py, ask_user.py
```

## Import paths

- `nexau.archs.tool.builtin.file_tools:read_file`, `write_file`, `replace`, `apply_patch`, etc.
- `nexau.archs.tool.builtin.shell_tools:run_shell_command`
- `nexau.archs.tool.builtin.web_tools:google_web_search`, `web_fetch`
- `nexau.archs.tool.builtin.session_tools:write_todos`, `complete_task`, `save_memory`, `ask_user`

## Where they are referenced

- `examples/deep_research/quickstart.py` -> session_tools, web_tools
- `examples/mcp/minimax_voice_deep_research.py` -> shell_tools, web_tools
- `docs/core-concepts/tools.md`, `agents.md` -> web_tools, session_tools
- `docs/getting-started.md`, `README.md`, `README_CN.md` -> all categories
- `tests/integration/test_tool_integration.py` -> file_tools, shell_tools
- `tests/integration/test_config_integration.py` -> shell_tools
