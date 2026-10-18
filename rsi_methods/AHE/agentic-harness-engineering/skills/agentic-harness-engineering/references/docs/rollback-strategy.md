# Rollback Strategy — Harness Change Rollback

> Corresponds to the Verify step of the AHE evolution loop: how to roll back safely when a
> change's prediction is falsified.

---

## When a rollback is needed

Based on the value of `verification.verdict` in the Change Manifest:

| Verdict | Meaning | Action |
|---------|---------|--------|
| **keep** | All predictions correct, the change worked | Keep the change, no rollback needed |
| **revert** | The change was ineffective or caused a regression | **Roll back immediately** |
| **partial** | Some predictions correct, some wrong | Roll back the problematic parts, keep the effective ones |

### Concrete conditions that trigger revert

The verdict should be revert when any of the following holds:

1. **An expected fix did not land** — every task in `expected_fixes` still fails this round
2. **A regression was observed** — a task in `at_risk_regressions` fails this round (it previously passed)
3. **Newly introduced failures** — 2+ newly failing tasks that were outside the expected scope
4. **A P0-severity regression** — even a single P0 regression should trigger a revert

---

## Rollback methods

### Method 1: git revert (recommended)

If the Harness files are under version control:

```bash
# List the files touched by the change
cat manifests/change_20260521_103000.json | python3 -c "
import sys, json
m = json.load(sys.stdin)
for c in m['changes']:
    print(c['file_path'])
"

# Run git revert for each changed file
git revert <commit-hash> --no-edit

# Or target a specific file
git checkout HEAD~1 -- tool_descriptions/search.tool.yaml
```

### Method 2: file restore

If there is no git, restore directly from a backup:

```bash
# If there is a pre-update backup
cp .hermes/backups/pre-update/tool_descriptions/search.tool.yaml \
   tool_descriptions/search.tool.yaml
```

### Method 3: partial rollback

If the verdict is partial, roll back only the failing parts:

| Situation | Action |
|-----------|--------|
| Some changes effective, some not | Revert only the ineffective change entries |
| Several changes mixed in one file | Manually edit back the problematic code section |
| Broken dependency chain | If B depends on A's change but A is reverted, B must be reverted too |

---

## After a rollback

### 1. Record a rollback Change Manifest

A rollback is itself a Harness change and should also generate a Change Manifest:

```json
{
  "manifest_version": "1.0",
  "iteration": 4,
  "changes": [{
    "change_id": "ch_rollback_003",
    "component": "tool_descriptions",
    "subtype": "update",
    "file_path": "tool_descriptions/search.tool.yaml",
    "summary": "Revert change ch_001: search pagination caused regression on T-013",
    "failure_evidence": "T-013 regressed from pass to fail after adding pagination parameters",
    "root_cause": "Pagination parameters changed default behavior from 'top results' to 'page 1 results'",
    "targeted_fix": "Revert search.tool.yaml to iteration 2 state",
    "predicted_impact": {
      "expected_fixes": ["T-013"],
      "at_risk_regressions": ["T-042", "T-057"],
      "rationale": "Reverting to known-good state; search truncation may reappear"
    }
  }],
  "verification": {
    "status": "pending",
    "scheduled_at": "2026-05-22T10:30:00+08:00"
  }
}
```

### 2. Update the Harness state

- If using version control: `git push` the revert commit
- If tracking via change manifests: update the original manifest's verification.verdict to "reverted"
- Record an incident log

### 3. Re-analyze

Do not immediately make the next change after a rollback. First answer:

- **Why was the prediction wrong?** — was failure_evidence not accurate enough? was the root_cause misidentified?
- **Should the problem be solved at a different component level?** — if tool_descriptions cannot fix it, should a constraint be added in system_rules, or should middleware do automatic completion?

---

## Preventing ineffective rollbacks

| Don't do this | Why |
|---------------|-----|
| Revert immediately without analyzing the root cause | The same failure will recur next round |
| Revert all changes because of 1 P2 regression | You may lose effective fixes; prefer partial |
| Forget to update the manifest after rolling back | You lose the evolution history and cannot attribute |
| Forget to update the `at_risk_regressions` prediction after rolling back | The same risk will be repeated next time |

---

## Best practices

1. **Change only one component per edit** — partial revert is easier
2. **Keep git commits fine-grained** — one commit per change so revert can target a single file precisely
3. **Pre-fill at_risk_regressions** — write it even when "unsure whether it will regress"; an empty list means "I am certain there is no risk" — which is almost never true
4. **Do not retry immediately after a rollback** — first ask "why was the prediction wrong", then consider a new change strategy
