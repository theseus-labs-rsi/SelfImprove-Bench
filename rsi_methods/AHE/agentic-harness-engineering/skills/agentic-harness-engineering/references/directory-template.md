# HARNESS.md Directory Template

The following is a starting directory structure that complies with the HARNESS.md v1.0
standard, used to initialize a new Agent workspace.

> ⚠️ Note: the directory structure created by the template works together with
> `validate_harness.py`.
> Running `python init_harness.py my-agent` automatically generates a complete, usable
> workspace.

```
<project-root>/
│
├── AGENTS.md                   # [System Rules] core behavior rules (required)
├── SOUL.md                     # [System Rules] identity statement (recommended)
│
├── MEMORY.md                   # [Long-Term Memory] long-lived stable facts (recommended)
│
├── tool_descriptions/          # [Tool Descriptions] (required, at least one tool)
│   └── example.tool.yaml       # example tool description
│
├── tools/                      # [Tool Implementations] (required, at least one tool)
│   └── example_tool.py         # example tool implementation
│
├── middleware/                 # [Middleware] (optional)
│
├── skills/                     # [Skills] (optional)
│
├── sub_agents/                 # [Sub-Agents] (optional)
│
├── manifests/                  # Change Manifest storage directory (recommended)
│
├── agent.yaml                  # [Registry] component registry (framework-dependent)
│
└── .gitignore
```

### Minimal runnable file examples

#### AGENTS.md
```markdown
# AGENTS.md

> Follows HARNESS.md v1.0

## Component overview
- System Rules → AGENTS.md + SOUL.md
- Tool Descriptions → tool_descriptions/
- Tool Implementations → tools/
- Long-Term Memory → MEMORY.md
```

#### tool_descriptions/example.tool.yaml
```yaml
name: example_tool
description: An example tool
input_schema:
  type: object
  properties:
    message:
      type: string
      description: Input message
  required: [message]
```

#### tools/example_tool.py
```python
def example_tool(message: str) -> str:
    return f"Echo: {message}"
```

#### .gitignore
```
# HARNESS.md
manifests/*.json
__pycache__/
```
