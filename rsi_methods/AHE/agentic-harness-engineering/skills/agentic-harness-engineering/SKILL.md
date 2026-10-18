---
name: agentic-harness-engineering
description: Agent Harness Evolution — an evaluation, audit, and improvement workflow based on the HARNESS.md v1.0 specification. On load it automatically checks the current workspace structure for compliance and guides evidence-driven Harness improvements. Triggered when the user asks for a Harness audit, to initialize an Agent workspace, to generate or verify a Change Manifest, to make an evidence-driven improvement to a harness component, or when the same failure pattern recurs repeatedly.
---

# AHE Skill — Agent Harness Evolution

> Implements the core patterns of the AHE (Agentic Harness Engineering) paper as an
> executable Skill.
> Follows the [HARNESS.md v1.0](references/HARNESS.md) specification.

---

## When to load this Skill

- **Doing a Harness Audit** — you want to know whether the current workspace structure complies → load the skill and run the audit flow
- **You changed a core component** — you modified the system prompt, a tool, middleware, etc. → load the skill and generate a Change Manifest
- **You want to verify a previous change** — you generated a Change Manifest but have not verified it yet → load the skill and run the verify flow
- **The Harness keeps hitting the same problem** — the same failure pattern occurs 2+ times → load the skill and make an evidence-driven improvement
- **Setting up a new Agent** — initialize a new Agent workspace → load the skill and initialize with the HARNESS.md template

---

## Workflow 1: Harness Audit

Evaluate whether the current workspace complies with the HARNESS.md v1.0 specification.

### Steps

1. **Scan the current workspace structure**
   - Check whether the 7 HARNESS.md component directories exist
   - Count the number and type of files under each component

2. **Check component orthogonality**
   - Is there tool implementation logic inside System Rules? → suggest moving it to tools/
   - Do Tool Descriptions and Tool Implementations map one-to-one? → check for gaps
   - Is there session-level temporary information inside MEMORY.md? → suggest moving it to workbench/

3. **Check Change Manifest existence**
   - Is there a HARNESS_MANIFEST.json or manifests/ directory at the root?
   - Do the most recent changes have corresponding manifests?

4. **Emit the audit report**
   - Compliance score (out of 10)
   - Status of each component (✅ compliant / ⚠️ needs attention / ❌ non-compliant)
   - A list of improvement suggestions

### Example output

```markdown
## Harness Audit Report — 2026-05-21

**Compliance score: 7/10**

| Component | Status | Notes |
|-----------|--------|-------|
| System Rules | ✅ | AGENTS.md + SOUL.md cleanly separated |
| Tool Descriptions | ✅ | Every tool has its own YAML |
| Tool Implementations | ⚠️ | Some tool descriptions are out of sync with the implementation |
| Middleware | ❌ | No middleware directory defined |
| Skills | ✅ | skills/ directory is well organized |
| Sub-Agents | ⚠️ | sub_agents/ directory exists but is not registered |
| Long-Term Memory | ✅ | MEMORY.md + experiences.md separated |

**Improvement suggestions:**
1. Audit the synchronization between tools/ and tool_descriptions/
2. Consider adding a middleware/ directory
3. Register sub_agents in code_agent.yaml
```

---

## Workflow 2: Generate Change Manifest

Generate a standardized Change Manifest when making a change to a Harness component.

### When to use

You have found a problem and are ready to change something. Before or after the change,
run this flow to record the modification.

### Steps

1. **Confirm the component type being changed** (must be one of the 7)
2. **Collect failure evidence** — describe what failed, with trace or log excerpts
3. **Analyze the root cause** — why did it fail? (not "the tool errored" but "the tool description did not declare a parameter, so the LLM did not know it could paginate")
4. **Describe the change** — exactly what was changed
5. **Predict the impact** — which tasks should be fixed, which might regress
6. **Write the manifest file**

### Change Manifest template

Write the following to `manifests/change_{timestamp}.json`:

```json
{
  "manifest_version": "1.0",
  "harness_spec_version": "1.0",
  "iteration": <current iteration number>,
  "timestamp": "<current time in ISO format>",
  "author": "<agent-name>",
  "changes": [
    {
      "change_id": "ch_<number>",
      "component": "<system_rules|tool_descriptions|tool_implementations|middleware|skills|sub_agents|long_term_memory>",
      "subtype": "<update|create|delete|register>",
      "file_path": "<path of the changed file>",
      "summary": "<one-line description of the change>",
      "failure_evidence": "<what failed + evidence>",
      "root_cause": "<root-cause analysis>",
      "targeted_fix": "<concrete change made>",
      "predicted_impact": {
        "expected_fixes": ["<tasks expected to be fixed>"],
        "at_risk_regressions": ["<tasks at risk of regressing>"],
        "rationale": "<why this prediction>"
      }
    }
  ],
  "verification": {
    "status": "pending",
    "scheduled_at": "<next verification time>"
  }
}
```

---

## Workflow 3: Verify Changes

Verify whether the predictions in a previously generated Change Manifest were accurate.

### When to use

- A pending manifest was generated and its scheduled_at time has arrived
- A round of evaluation just finished and you want to see the effect of an earlier change

### Steps

1. **Read all pending manifests** under manifests/
2. **Collect evaluation results** — this round's pass/fail data
3. **Verify each prediction**
   - Did the tasks in `expected_fixes` actually pass?
   - Did the tasks in `at_risk_regressions` actually avoid regression?
4. **Update the verification field in the manifest**
   - Set verdict: keep / revert / partial
5. **If it is a revert** — perform the rollback (git revert or file restore)

### Verification update template

```json
{
  "verification": {
    "status": "<verified|reverted|partial>",
    "scheduled_at": "<the previously set verification time>",
    "completed_at": "<current time>",
    "result": {
      "expected_fixes_verified": ["<tasks actually fixed>"],
      "unexpected_fixes": ["<tasks unexpectedly fixed>"],
      "regressions_observed": ["<observed regressions>"],
      "false_predictions": ["<tasks predicted incorrectly>"]
    },
    "verdict": "<keep|revert|partial>"
  }
}
```

---

## Workflow 4: Harness initialization

Create an Agent workspace that complies with the HARNESS.md specification from scratch.

### Steps

1. **Confirm the Agent type** — Hermes / OpenClaw / Claude Code / Codex / other
2. **Create the directory structure** — following the standard HARNESS.md template
3. **Initialize a minimal Harness**
   - Create the System Rules files (AGENTS.md + SOUL.md)
   - Register 1-2 basic tools
4. **Mark the HARNESS.md version** — add `follows HARNESS.md v1.0` to AGENTS.md
5. **Create the manifests/ directory** — to hold Change Manifests

---

## Principles

### Evidence-driven

> Do not make changes based on intuition, guesswork, or "best practices". Every change must
> be traceable to concrete failure evidence.

### Falsifiable

> Every change is a falsifiable hypothesis. If the next round of evaluation proves the
> prediction wrong, the change should be rolled back.

### Minimal starting point

> Evolve from the fewest components. You do not need all 7 components at once — only System
> Rules + 1-2 tools.

### Component escalation

> If the same failure pattern persists across 2+ iterations and fixing it at the current
> component level is ineffective → roll back and solve it at a different component level.

---

## References

- [HARNESS.md v1.0 — full specification](references/HARNESS.md)
- [Change Manifest JSON Schema](references/change-manifest-schema.json)
