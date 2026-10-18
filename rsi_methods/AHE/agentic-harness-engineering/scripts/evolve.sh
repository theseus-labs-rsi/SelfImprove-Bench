#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: ./scripts/evolve.sh [options] <config_file>

Start an Agentic Harness Engineering evolution experiment in a background tmux session.

Arguments:
  config_file              config file path (relative to Agentic Harness Engineering/ or absolute)

Options:
  --experiment NAME        resume an existing experiment (directory name under experiments/)
  --start-iteration N      start from round N (default 1)
  --skip-eval              skip evaluation and use existing results directly
  --session NAME           custom tmux session name
  --batch                  batch mode: start every experiment under configs/experiments/
  --attach                 automatically attach to the tmux session after launch
  -h, --help               show help

Examples:
  # Start a single experiment
  ./scripts/evolve.sh configs/experiments/exp-simple-code-gpt54.yaml

  # Resume an interrupted experiment from round 16
  ./scripts/evolve.sh --experiment 2026-03-13__18-02-54__gpt54 --start-iteration 16 configs/experiments/exp-simple-code-gpt54.yaml

  # Start all experiments in batch
  ./scripts/evolve.sh --batch

  # Attach automatically after launch
  ./scripts/evolve.sh --attach configs/experiments/exp-simple-code-gpt54.yaml

Managing tmux sessions:
  tmux ls                          # list all sessions
  tmux attach -t <session>         # enter a session
  Ctrl-b d                         # detach from the session (return to background)
  tmux kill-session -t <session>   # terminate a session
EOF
}

EXPERIMENT=""
START_ITER=""
SKIP_EVAL=""
SESSION_NAME=""
BATCH_MODE=false
AUTO_ATTACH=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            EXPERIMENT="$2"; shift 2 ;;
        --start-iteration)
            START_ITER="$2"; shift 2 ;;
        --skip-eval)
            SKIP_EVAL="--skip-eval"; shift ;;
        --session)
            SESSION_NAME="$2"; shift 2 ;;
        --batch)
            BATCH_MODE=true; shift ;;
        --attach)
            AUTO_ATTACH=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            echo "Unknown option: $1"; usage; exit 1 ;;
        *)
            CONFIG_FILE="$1"; shift ;;
    esac
done

if ! command -v tmux &>/dev/null; then
    echo "Error: tmux is not installed. Run: brew install tmux"
    exit 1
fi

# --- Batch mode ---
if $BATCH_MODE; then
    CONFIGS_DIR="$PROJECT_ROOT/configs/experiments"
    if [[ ! -d "$CONFIGS_DIR" ]]; then
        echo "Error: config directory not found: $CONFIGS_DIR"
        exit 1
    fi

    count=0
    for cfg in "$CONFIGS_DIR"/*.yaml; do
        name=$(basename "$cfg" .yaml | sed 's/^exp-[0-9]*-//')
        ts=$(date +%Y%m%d-%H%M)
        sess="ahe-${name}-${ts}"

        if tmux has-session -t "$sess" 2>/dev/null; then
            echo "[skip] session '$sess' already exists"
            continue
        fi

        cmd="export PATH="$HOME/.local/bin:$PATH" && cd '$PROJECT_ROOT' && uv run python evolve.py --config '$cfg'"
        tmux new-session -d -s "$sess" "$cmd"
        echo "[start] $name -> tmux session '$sess'"
        ((count++))
    done

    echo ""
    echo "Started $count experiments."
    echo "List:   tmux ls"
    echo "Enter:  tmux attach -t <session>"
    exit 0
fi

# --- Single-experiment mode ---
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: specify a config file"
    usage
    exit 1
fi

if [[ ! "$CONFIG_FILE" = /* ]]; then
    if [[ -f "$SCRIPT_DIR/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
    elif [[ -f "$PROJECT_ROOT/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
    fi
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config file does not exist: $CONFIG_FILE"
    exit 1
fi

if [[ -z "$SESSION_NAME" ]]; then
    cfg_basename=$(basename "$CONFIG_FILE" .yaml | sed 's/^exp-[0-9]*-//')
    ts=$(date +%Y%m%d-%H%M)
    SESSION_NAME="ahe-${cfg_basename}-${ts}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "  enter: tmux attach -t $SESSION_NAME"
    echo "  terminate: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

CMD="export PATH="$HOME/.local/bin:$PATH" && cd '$PROJECT_ROOT' && uv run python evolve.py --config '$CONFIG_FILE'"

if [[ -n "$EXPERIMENT" ]]; then
    CMD="$CMD --experiment '$EXPERIMENT'"
fi
if [[ -n "$START_ITER" ]]; then
    CMD="$CMD --start-iteration $START_ITER"
fi
if [[ -n "$SKIP_EVAL" ]]; then
    CMD="$CMD $SKIP_EVAL"
fi

# Keep the session alive by appending a shell, so the final output can be inspected
CMD="$CMD; echo ''; echo '=== experiment finished ==='; echo 'Press Enter to close this session'; read"

tmux new-session -d -s "$SESSION_NAME" "$CMD"

echo "============================================"
echo "  Agentic Harness Engineering experiment started in a background tmux session"
echo "============================================"
echo "  Session:  $SESSION_NAME"
echo "  Config:   $(basename "$CONFIG_FILE")"
[[ -n "$EXPERIMENT" ]] && echo "  resuming experiment: $EXPERIMENT"
[[ -n "$START_ITER" ]] && echo "  starting round: $START_ITER"
echo ""
echo "  enter session:  tmux attach -t $SESSION_NAME"
echo "  detach to background: Ctrl-b d"
echo "  list all:       tmux ls"
echo "  terminate:      tmux kill-session -t $SESSION_NAME"
echo "============================================"

if $AUTO_ATTACH; then
    exec tmux attach -t "$SESSION_NAME"
fi
