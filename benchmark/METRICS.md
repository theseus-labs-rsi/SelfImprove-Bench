# Metric policies and historical exports

`benchmark.metrics.calculate_metrics()` is the single numeric entry point for
the unified runner and `benchmark.aggregate_results.parse_unified()`.
Its output records `metric_policy_version = "unified_all_attempts_v2"`.

Under this policy:

- Every attempted case remains in the score denominator. Explicit errors,
  skipped cases, and invalid or unavailable numeric scores contribute zero.
  This applies to the supplied training/holdout run records; unexecuted cases
  are not invented.
- Evolution gain uses the configured positive integer `evolution_window` over
  non-replay training records. A missing window in an older unified result
  defaults to one. Export columns `first_absolute_score` and
  `last_absolute_score` are the corresponding window means.
- Retention compares the original and replay execution scores. Replay does not
  enter the evolution windows, but its token and time costs enter the complete
  training cost totals. Holdout costs are outside this training-cost scope.
- Missing, null, negative, nonfinite, boolean, or otherwise invalid cost values
  make that cost total unknown (`null`). Token counts must be whole numbers;
  durations may be fractional. Known zero remains zero. An empty training
  schedule has zero observed training cost and undefined gain.
- Cost per gain is defined only for strictly positive gain and a known cost
  total. Zero or negative gain produces `null`, while the observed cost total
  remains available.
- Holdout export categories retain attempted-case weights, including when
  multiple configured groups map to the same category.

The exporter recomputes unified metrics from the saved run records instead of
trusting possibly stale embedded summaries. It does not modify the input file.
`metric_policy_version` describes the recomputation, while
`source_metric_policy_version` preserves the saved embedded version when one
exists. An absent source version is `null`; this does not claim that an old
experiment originally used the current policy. Different configured windows
are kept in separate aggregate groups. JSON, CSV, and Markdown exports show the
policy and window.

Legacy overall-control-loop summaries contain endpoint averages rather than
the per-case records required by the unified policy. They remain readable under
the explicit `legacy_overall_control_loop_v1` policy: initial/final endpoint
gain, available round-level stability, partial held-in token accounting, and
the existing low-confidence filesystem or peer time estimates. Missing raw
cases cannot be reconstructed as failed or successful attempts. Legacy and
unified policies are kept in separate aggregate groups and must not be treated
as equivalent cost measurements.

The version records the metric definition; it cannot repair missing historical
usage. A caller must retain unknown usage in each run record for that uncertainty
to reach both the runner summary and exports.

Checkpoint recovery archives interrupted attempts separately under
`abandoned_attempts/` and reruns them from the last committed state. The metrics
above describe the resulting canonical run records, not a billing total that
includes every discarded retry. Keep those archived attempts for experiment
cost audits; their unreported usage cannot be inferred from the resumed run.
