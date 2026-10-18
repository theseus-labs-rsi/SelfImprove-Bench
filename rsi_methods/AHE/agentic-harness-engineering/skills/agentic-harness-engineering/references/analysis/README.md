# Analysis — experience observability standard

> Corresponds to the second layer of the AHE paper: Experience Observability.

## Goal

Distill large volumes of raw trajectory / trace data into consumable analysis reports so the
evolution agent can see where the failure patterns are.

## Report structure

analysis/
├── README.md              # this file
├── overview-template.md   # overview report template
└── detail/
    └── task_xxx.md        # per-task detailed analysis

## Overview report contents
- This round's evaluation summary (pass/fail counts, success rate)
- Failure pattern clustering
- Root-cause distribution
- Components recommended for priority fixing

## Detail report contents
- Task description
- Trace excerpt
- Failure cause analysis
- Root cause localized to a specific component
- Fix suggestions
