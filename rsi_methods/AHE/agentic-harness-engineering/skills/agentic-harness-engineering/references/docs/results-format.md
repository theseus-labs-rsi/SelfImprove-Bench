# results.json — evaluation result standard format

> The evaluation result data format used for Verify Changes in HARNESS.md v1.0.

## Format definition

```json
{
  "version": "1.0",
  "iteration": 3,
  "timestamp": "2026-05-21T10:30:00+08:00",
  "total": 12,
  "passed": ["T-001", "T-002", "T-003", "T-005", "T-008", "T-009", "T-011", "T-012"],
  "failed": ["T-004", "T-006", "T-007", "T-010"],
  "traces": {
    "T-001": "traces/task-001.json",
    "T-004": "traces/task-004.json"
  },
  "summary": {
    "pass_rate": 66.7,
    "fail_rate": 33.3,
    "new_failures": ["T-010"],
    "fixed_issues": ["T-003"]
  }
}
```

## Field descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | ✅ | Format version |
| `iteration` | integer | ✅ | Current iteration number (matches the Change Manifest iteration field) |
| `timestamp` | string (ISO 8601) | ✅ | Evaluation completion time |
| `total` | integer | ✅ | Total number of tasks this round |
| `passed` | string[] | ✅ | List of passing task IDs |
| `failed` | string[] | ✅ | List of failing task IDs |
| `traces` | object | Recommended | Mapping of task ID → trace file path |
| `summary.pass_rate` | number | Recommended | Pass rate percentage |
| `summary.fail_rate` | number | Recommended | Fail rate percentage |
| `summary.new_failures` | string[] | Recommended | Newly failing tasks this round (pass last round, fail this round) |
| `summary.fixed_issues` | string[] | Recommended | Failures fixed this round (fail last round, pass this round) |

## Usage

### With verify_manifest.py

```bash
# Generate results.json after evaluation
python3 evaluate.py > results.json

# Pass it to the verify script to check the Change Manifest predictions
python3 AHE.skill/scripts/verify_manifest.py \
  --workspace . \
  --results results.json
```

### With generate_manifest.py

The `expected_fixes` and `at_risk_regressions` fields in the Change Manifest produced by
generate_manifest.py should be verified against the `passed` / `failed` lists of results.json.

## Example

```json
{
  "version": "1.0",
  "iteration": 1,
  "timestamp": "2026-05-21T00:00:00+08:00",
  "total": 5,
  "passed": ["T-001", "T-003", "T-005"],
  "failed": ["T-002", "T-004"],
  "traces": {
    "T-002": "traces/task-002-tool-error.json",
    "T-004": "traces/task-004-memory-oom.json"
  },
  "summary": {
    "pass_rate": 60.0,
    "fail_rate": 40.0,
    "new_failures": ["T-002", "T-004"],
    "fixed_issues": []
  }
}
```
