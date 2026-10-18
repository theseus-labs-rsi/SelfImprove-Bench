# 🦁 HARNESS.md + AHE.skill

> **This file is a backup of the original project repository README (kept as reference material for the skill).**
> Relative paths mentioned in the document (such as `AHE.skill/scripts/...` or `examples/...`)
> describe the original GitHub repository layout; their equivalents inside this skill are:
> - `AHE.skill/scripts/` → `scripts/`
> - `AHE.skill/references/` → `references/`
> - `examples/`, `analysis/`, `docs/`, `HARNESS.md` → `references/examples/`, `references/analysis/`, `references/docs/`, `references/HARNESS.md`

> **Status: v0.1 experimental**
>
> Distills the core ideas of the Agentic Harness Engineering paper into a cross-agent Harness
> specification and a first-generation audit tool.
>
> ⚠️ The current version is a normalized excerpt of the AHE ideas plus a static audit
> prototype, not the full AHE automatic-evolution framework. See "Current capability
> boundaries" below.

---

## What this is

By 2026 the industry consensus is clear: **what determines whether an AI Agent can work
reliably is not the model itself but the Harness around it.**

The AHE paper demonstrates that driving automatic Harness evolution with three layers of
observability can lift a coding agent's Terminal-Bench 2 score from 69.7% to 77.0%,
surpassing the hand-written Codex CLI.

But a paper is a paper, and putting it into practice is another matter.

This repository distills the core insights of AHE into two things:

### 📄 HARNESS.md — Agent Harness Specification v1.0

A **cross-agent Harness structural specification**. Any Agent (Hermes / OpenClaw / Claude
Code / Codex / Cursor) can declare that it "follows HARNESS.md v1.0", giving harness
components a shared language, structure, and evolvability.

**Core contents:**
- The definition and responsibilities of the 7 orthogonal components
- Standard directory structure
- **Change Manifest standard** — every change carries a falsifiable prediction that can be verified, attributed, and rolled back
- The standard evolution loop (Evaluate → Analyze → Improve → Verify)
- Compliance checklist

### 🔧 AHE.skill — a Harness audit tool on Hermes

Brings the HARNESS.md specification to the Hermes Agent as an executable Skill.

**Currently implemented:**
- ✅ **Harness Audit** — check workspace structure compliance (supports multiple profiles: hermes / openclaw / codex / generic)

**Planned (not implemented):**
- ⏳ Generate Manifest — automatically generate a Change Manifest (currently only a template and schema exist)
- ⏳ Verify Changes — verify change predictions (currently only the process documentation exists)
- ⏳ Harness Init — automatically initialize a workspace (currently there is an `init_harness.py` script)

---

## Quick start

### 1. Read the specification

```bash
cat HARNESS.md
```

### 2. Audit your Agent workspace

```bash
# Generic audit
python AHE.skill/scripts/validate_harness.py /path/to/workspace

# Specify an Agent framework (more precise rule checks)
python AHE.skill/scripts/validate_harness.py /path/to/workspace --profile hermes
python AHE.skill/scripts/validate_harness.py /path/to/workspace --profile openclaw
python AHE.skill/scripts/validate_harness.py /path/to/workspace --profile codex

# JSON output (for programmatic consumption)
python AHE.skill/scripts/validate_harness.py /path/to/workspace --json

# Positive example: audit a minimal compliant workspace
python AHE.skill/scripts/validate_harness.py examples/minimal-workspace
```

> ⚠️ **Do not run audit on this repository itself.** This repository is a specification +
> tooling repository, not an Agent workspace.
> Run it **against a target Agent workspace**.
> For a positive example use `examples/minimal-workspace/`.

### 3. Initialize a new workspace from scratch

```bash
python AHE.skill/scripts/init_harness.py my-agent-workspace [--profile hermes]
```

### 4. Declare conformance in your AGENTS.md

```markdown
# AGENTS.md

> 🏷️ Follows HARNESS.md v1.0
```

---

## Current capability boundaries

### Implemented
- ✅ **HARNESS.md specification** — component separation, directory standard, Change Manifest format, evolution loop
- ✅ **validate_harness.py** — static directory-structure audit (supports 4 profiles)
- ✅ **Change Manifest JSON Schema** — validates the manifest format
- ✅ **directory-template.md** — directory structure template
- ✅ **examples/minimal-workspace/** — a positive example
- ✅ **profiles/** — Hermes / OpenClaw / Codex / generic adapters

### Not implemented (future iterations)
- ⏳ **Experience Observability** — trace collection format, evaluation run data structures, failure-analysis report generator
- ⏳ **Automatic Generate Manifest** — generate from a git diff / user input
- ⏳ **Automatic Verify Manifest** — read evaluation results, verify predictions, write back the verdict
- ⏳ **Evolution loop** — an executable Evaluate → Analyze → Improve → Verify loop
- ⏳ **More profiles** — adapters for frameworks such as Claude Code and Cursor

### Known limitations
- validate_harness.py can only do static file/directory checks; it cannot check component content quality
- For frameworks such as OpenClaw / Codex, tool_descriptions may not fully match their runtime injection mechanism
- In some Agent frameworks the tool descriptions do not necessarily exist as files in the workspace

---

## Project structure

```
├── HARNESS.md                              # specification document (cross-agent)
├── README.md                               # this file
├── AHE.skill/
│   ├── SKILL.md                            # Hermes Skill definition
│   ├── references/
│   │   ├── HARNESS.md                      # copy of the specification
│   │   ├── change-manifest-schema.json     # Manifest JSON Schema
│   │   └── directory-template.md           # directory template
│   ├── scripts/
│   │   ├── validate_harness.py             # compliance check script
│   │   ├── init_harness.py                 # initialization script
│   │   ├── generate_manifest.py            # Manifest generation tool (in development)
│   │   └── verify_manifest.py              # Manifest verification tool (in development)
│   └── profiles/
│       ├── hermes.yaml                     # Hermes framework adapter
│       ├── openclaw.yaml                   # OpenClaw framework adapter
│       └── codex.yaml                      # Codex framework adapter
├── examples/
│   └── minimal-workspace/                  # minimal compliant workspace example
└── analysis/                               # experience observability standard documents
    ├── README.md                           # analysis report standard notes
    ├── overview-template.md                # overview report template
    └── detail-template.md                  # per-task analysis template
```

---

## Background

This work builds on the following research:

- **Agentic Harness Engineering** — proposes a three-layer observability framework (component / experience / decision) that lets a coding agent's Harness evolve automatically
  - Terminal-Bench 2 #3 (84.7% pass@1), surpassing the hand-written Codex CLI and ACE/TF-GRPO
  - The frozen Harness transfers across benchmarks and models

- **Agent Harness for LLM Agents: A Survey** — a formal six-component definition of a Harness, H=(E,T,C,S,L,V)

---

## License

MIT
