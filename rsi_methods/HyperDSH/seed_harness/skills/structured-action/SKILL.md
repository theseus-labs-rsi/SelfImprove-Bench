---
name: structured-action
description: Produce valid, evidence-grounded actions for structured benchmark environments.
---

# Structured action discipline

Before producing an action, identify the current objective, legal actions,
resource limits, and the most recent environment observation. Prefer actions
whose effects can be checked on the next observation. Never invent hidden state
or return fields outside the task's requested JSON schema.
