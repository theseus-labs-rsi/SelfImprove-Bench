# HARNESS.md — Agent Harness Specification v1.0

> **A cross-agent Harness structural specification.**
> It gives any Agent's "outer system" (Hermes / OpenClaw / Claude Code / Codex / Cursor) a
> shared language, structure, and evolvability.
>
> Inspiration: Agentic Harness Engineering: Observability-Driven Automatic Evolution of
> Coding-Agent Harnesses (2026).

---

## Why this specification exists

By 2026 the industry consensus is clear: **what determines whether an AI Agent can work
reliably is not the model itself but the Harness around it.**

But today:
- Every agent framework has a different harness structure, with its own component names,
  directory conventions, and registration mechanisms.
- There is no shared language to describe "I changed a tool description" or "I added
  middleware" — different agents are not speaking the same language.
- Whether a harness change is good or bad cannot be verified — a change is just a change,
  and nobody checks whether it fixed or broke something.

**HARNESS.md solves exactly these three problems: a shared language, a standard structure,
and evolvability.**

---

## Core architecture: 7 orthogonal components

A complete Agent Harness consists of 7 orthogonal components. **Orthogonal** means: changing
one does not affect the others.

| # | Component | Responsibility | Example files | Analogy |
|---|-----------|----------------|---------------|---------|
| 1 | **System Rules** | The agent's behavior rules, workflow guidance, and boundaries | `AGENTS.md`, `SOUL.md`, `systemprompt.md`, `.cursorrules` | Constitution |
| 2 | **Tool Descriptions** | Each tool's schema, purpose, and usage pitfalls | `tool_descriptions/*.yaml` | Product manual |
| 3 | **Tool Implementations** | The code logic that executes a tool | `tools/*.py`, `tools/*.js` | Robot factory |
| 4 | **Middleware** | Hooks in the execution pipeline — intercept, transform, enhance | `middleware/*.py` | Security checkpoint |
| 5 | **Skills** | Reusable workflow patterns | `skills/*/SKILL.md` | SOP manual |
| 6 | **Sub-Agents** | Delegatable sub-agent units | `sub_agents/*/` | Outsourced team |
| 7 | **Long-Term Memory** | Persistent knowledge storage across sessions | `MEMORY.md`, `experiences.md` | Personal notes |

### Component interaction principles

```
user input
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  System Rules ───── how the agent thinks and decides          │
├──────────────────────────────────────────────────────────────┤
│  Middleware ─────── intercept/transform input and output      │
├──────────────────────────────────────────────────────────────┤
│  Skills ─────────── match and reuse workflows                 │
├──────────────────────────────────────────────────────────────┤
│  Tool Descriptions ─ tell the LLM what tools exist and how    │
│  Tool Implementations ─ execute the tool logic                │
├──────────────────────────────────────────────────────────────┤
│  Sub-Agents ─────── delegate focused subtasks                 │
├──────────────────────────────────────────────────────────────┤
│  Long-Term Memory ─ read/write cross-session knowledge        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
  output
```

### Anti-patterns

- ❌ Writing tool implementation logic into System Rules (mixing rules and implementation)
- ❌ Cramming long-term memory into System Rules (it bloats context on every load)
- ❌ Multiple components doing the same thing (e.g. both prompt and middleware formatting output)
- ❌ Changing a tool without updating its description (implementation and description drift apart)

---

## Directory structure standard

An Agent workspace that conforms to HARNESS.md should follow this structure:

```
<workspace>/
├── AGENTS.md                   # System Rules (core rules file)
├── SOUL.md                     # System Rules (identity statement)
├── MEMORY.md                   # Long-Term Memory
├── experiences.md              # Long-Term Memory (lessons learned)
│
├── tool_descriptions/          # Tool Descriptions
│   ├── search.tool.yaml
│   ├── read_file.tool.yaml
│   └── run_command.tool.yaml
│
├── tools/                      # Tool Implementations
│   ├── search_tools.py
│   ├── file_tools.py
│   └── shell_tools.py
│
├── middleware/                 # Middleware
│   ├── context_compaction.py
│   ├── output_formatter.py
│   └── tool_output_truncator.py
│
├── skills/                     # Skills
│   ├── code-review/
│   │   └── SKILL.md
│   ├── data-analysis/
│   │   └── SKILL.md
│   └── ...
│
├── sub_agents/                 # Sub-Agents
│   ├── researcher/
│   │   └── config.yaml
│   └── debugger/
│       └── config.yaml
│
└── agent.yaml                  # Component registry (optional, framework-dependent)
```

### Minimal starting state

A working harness needs only 2 components at minimum:
1. **System Rules** — at least one rules file
2. **Tool Descriptions + Implementations** — at least one tool

The other components (Middleware, Skills, Sub-Agents, Long-Term Memory) should be **added
only when needed**. The AHE paper shows that evolving from a minimal state works best — every
improvement is "earned", not "preset".

---

## Component registration standard

When an agent framework supports a registry (such as Hermes's `config.yaml` or AHE's
`code_agent.yaml`), the registration format should follow these patterns.

### Tools registration

```yaml
tools:
  - name: read_file
    description_path: ./tool_descriptions/read_file.tool.yaml
    binding: tools.file_tools:read_file
```

Every tool needs three things:
1. **Description YAML** — what the LLM sees (purpose, parameter schema, usage pitfalls)
2. **Implementation binding** — the code path that actually executes
3. **Registry entry** — declared in the config file

### Middleware registration

```yaml
middlewares:
  - import: middleware.long_tool_output:LongToolOutputMiddleware
    params:
      max_output_chars: 10000
```

### Skills registration

```yaml
skills:
  - ./skills/code-review
```

### Sub-Agents registration

```yaml
sub_agents:
  - name: researcher
    config: ./sub_agents/researcher/config.yaml
```

---

## Change Manifest standard format

**This is the core of the whole specification.** Every change to a Harness component should
come with a standard Change Manifest so the change can be verified, attributed, and rolled
back.

### Manifest schema

```json
{
  "manifest_version": "1.0",
  "harness_spec_version": "1.0",
  "iteration": 3,
  "timestamp": "2026-05-21T10:30:00+08:00",
  "author": "agent-name-or-human",
  "changes": [
    {
      "change_id": "ch_001",
      "component": "tool_descriptions",
      "subtype": "update",
      "file_path": "tool_descriptions/search.tool.yaml",
      "summary": "Document the pagination parameters of the search tool",
      "failure_evidence": "Task T-042 failed when the search returned more than 50 results; the trace shows the agent trying to pass page=2 but the tool description never declares that parameter",
      "root_cause": "The search tool actually supports pagination, but the tool description did not declare page_size or offset, so the LLM did not know it could paginate",
      "targeted_fix": "Add page_size (integer) and offset (integer) to input_schema.properties in search.tool.yaml and document the pagination usage in the description",
      "predicted_impact": {
        "expected_fixes": ["T-042", "T-057"],
        "at_risk_regressions": ["T-013"],
        "rationale": "T-042 and T-057 both failed due to search result truncation; T-013 depends on search but uses a different call pattern"
      }
    }
  ],
  "verification": {
    "status": "pending",
    "scheduled_at": "2026-05-22T10:30:00+08:00"
  }
}
```

### Field descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `manifest_version` | ✅ | Manifest format version |
| `harness_spec_version` | ✅ | The HARNESS.md version being followed |
| `iteration` | ✅ | Iteration number |
| `timestamp` | ✅ | Change time |
| `author` | ✅ | Who made the change (agent name or human) |
| `changes[].component` | ✅ | The component type changed (must be one of the 7) |
| `changes[].failure_evidence` | ✅ | **What failed** + citable trace evidence |
| `changes[].root_cause` | ✅ | **Why it failed** (root-cause analysis) |
| `changes[].targeted_fix` | ✅ | **What was changed** (concrete change description) |
| `changes[].predicted_impact` | ✅ | **What is predicted to improve/regress** |
| `verification.status` | ✅ | `pending` / `verified` / `reverted` |
| `verification.scheduled_at` | Recommended | Time of the next verification |

### Core principle: falsifiable

> **Every change is a falsifiable hypothesis.**

The verification result after the next round of evaluation should be written back into the
manifest:

```json
{
  "verification": {
    "status": "verified",
    "scheduled_at": "2026-05-22T10:30:00+08:00",
    "completed_at": "2026-05-22T10:45:00+08:00",
    "result": {
      "expected_fixes_verified": ["T-042", "T-057"],
      "unexpected_fixes": [],
      "regressions_observed": [],
      "false_predictions": []
    },
    "verdict": "keep"
  }
}
```

Three verdicts:
- `keep` — the change worked; keep it
- `revert` — the change was ineffective or caused a regression; roll it back
- `partial` — partially effective; needs adjustment

---

## Standard evolution loop

The standard evolution loop defined by HARNESS.md:

```
┌──────────────────────────────────────────────────┐
│                    round N                        │
│                                                    │
│  ① Evaluate ── run test tasks with the current    │
│      │          Harness                            │
│      │          output: traces + scores            │
│      ▼                                              │
│  ② Analyze ─── analyze traces, extract patterned  │
│      │          failure evidence                   │
│      │          output: root-cause report           │
│      ▼                                              │
│  ③ Improve ─── edit Harness components based on   │
│      │          the evidence                        │
│      │          output: Change Manifest             │
│      ▼                                              │
│  ④ Verify ──── next-round evaluation, verify the  │
│                 change predictions                  │
│                 output: Verdict (keep / revert)     │
└──────────────────────────────────────────────────┘
```

### Outputs per step

| Step | Input | Output |
|------|-------|--------|
| **Evaluate** | Current Harness (Hₙ) + test task set | Trace files + pass/fail results |
| **Analyze** | Trace files | Root-cause analysis report (overview + per-task) |
| **Improve** | Root-cause report + current Harness | Modified Harness (Hₙ₊₁) + Change Manifest |
| **Verify** | Change Manifest + next-round evaluation results | Verdict (keep / revert / partial) |

### The three observability layers

1. **Component observability** — each component maps to a unique file; changes are visible and reversible
2. **Experience observability** — traces are distilled into consumable root-cause reports rather than raw logs
3. **Decision observability** — every change carries a falsifiable prediction that can be verified and attributed

---

## Compliance checklist

To claim "follows HARNESS.md v1.0", an Agent must satisfy the following.

### Hard requirements

- [ ] The 7 harness components are clearly separated into distinct files/directories
- [ ] Every tool has a separate description file (the schema the LLM sees) and implementation file
- [ ] Every change to the Harness comes with a Change Manifest
- [ ] The Change Manifest contains failure_evidence + root_cause + targeted_fix + predicted_impact
- [ ] Every change is verified (verified / reverted / partial)

### Soft requirements

- [ ] Evolution starts from a minimal Harness (only System Rules + 1-2 tools)
- [ ] The workspace directory structure follows the standard template
- [ ] There is an automated Evaluate process (cronjob / CI)
- [ ] There is an automated Analyze process (trace distillation)
- [ ] Change Manifests are stored in version control (git-tracked)

### Anti-patterns

- [ ] ❌ Writing tool implementation logic into System Rules
- [ ] ❌ Changing only the implementation without changing the description (tool out of sync)
- [ ] ❌ Modifying core components without verification

---

## Adoption guide

### For existing agent frameworks

If your agent already has its own structure (such as Hermes's H3M or Claude Code's
CLAUDE.md), you do not need to start over. Just:

1. **Do a Harness Audit** — check whether the current components are orthogonal and the boundaries are clear
2. **Fill in Change Manifests** — add manifests for the most recent changes
3. **Declare conformance** — add a line `follows HARNESS.md v1.0` to AGENTS.md

### For new agent projects

Initialize the workspace directly with the directory structure above and start from a minimal
state.

### Implementation references

- [agentic-harness-engineering skill](../SKILL.md) — the Skill that implements this spec (with audit, generate-manifest, and verify)
- The AHE paper — the academic basis for this spec (2026)

---

## Version history

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-05-21 | Initial version |

---

## License

MIT — free to use, modify, and distribute.

---

*"The harness is the dataset. Your competitive advantage is the trajectories your harness captures."*
*— Philipp Schmid, 2026*
