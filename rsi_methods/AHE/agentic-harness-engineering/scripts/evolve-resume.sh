#!/usr/bin/env bash
set -euo pipefail

# # # Resume the gpt54 experiment from round 1, skipping evaluation (reuse existing rollout results)
# exec "$(dirname "$0")/evolve.sh" \
#     --experiment xxx \
#     --start-iteration 13 \
#     --skip-eval \
#     configs/experiments/exp-003-gpt54-32.yaml


exec "$(dirname "$0")/evolve.sh" \
    --experiment xxx \
    --start-iteration 13 \
    --skip-eval \
    xxx
    

    # --skip-eval \
# exec "$(dirname "$0")/evolve.sh" \
    # --experiment xxx \
    # --start-iteration 1 \
    # --skip-eval \
    # xxx