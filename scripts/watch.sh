#!/usr/bin/env bash
set -euo pipefail
#
# Watch a trained SAC model in the Drake Meshcat 3D visualizer.
#
# Usage:
#   ./scripts/watch.sh nominal 500k          # Nominal SAC @ 500k steps, nominal physics
#   ./scripts/watch.sh nominal 750k          # Nominal SAC @ 750k steps
#   ./scripts/watch.sh nominal 1m            # Nominal SAC @ 1M steps (final)
#   ./scripts/watch.sh robust 500k           # Robust SAC @ 500k steps
#   ./scripts/watch.sh robust 750k           # Robust SAC @ 750k steps
#   ./scripts/watch.sh robust 1m             # Robust SAC @ 1M steps (final)
#   ./scripts/watch.sh fsm                   # FSM baseline (no RL)
#
#   ./scripts/watch.sh nominal 1m --randomize    # Test with randomized physics
#   ./scripts/watch.sh nominal 1m --episodes 3   # Watch 3 episodes
#   ./scripts/watch.sh nominal 1m --record-path logs/demo.html  # Save recording
#

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"

MODEL_TYPE="${1:-}"
STEPS="${2:-}"
shift 2 2>/dev/null || true

if [[ -z "$MODEL_TYPE" ]]; then
    echo "Usage: ./scripts/watch.sh <nominal|robust|fsm> [500k|750k|1m] [extra flags...]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/watch.sh nominal 1m                    # Best nominal model, nominal physics"
    echo "  ./scripts/watch.sh robust 1m --randomize         # Best robust model, randomized physics"
    echo "  ./scripts/watch.sh nominal 500k                  # Nominal @ 500k checkpoint"
    echo "  ./scripts/watch.sh fsm                           # FSM baseline"
    echo "  ./scripts/watch.sh nominal 1m --episodes 3       # Watch 3 episodes"
    echo "  ./scripts/watch.sh robust 1m --record-path logs/demo.html"
    exit 1
fi

EXTRA_ARGS=("$@")

if [[ "$MODEL_TYPE" == "fsm" ]]; then
    echo "Launching FSM baseline in Meshcat..."
    exec python -m src.evaluate --fsm-only --render --realtime --episodes 1 "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

if [[ -z "$STEPS" ]]; then
    echo "Error: specify step count (500k, 750k, or 1m)"
    exit 1
fi

case "$MODEL_TYPE" in
    nominal) DATA_DIR="data/sac_nominal_1m" ;;
    robust)  DATA_DIR="data/sac_robust_1m" ;;
    *)
        echo "Error: unknown model type '$MODEL_TYPE'. Use: nominal, robust, or fsm"
        exit 1
        ;;
esac

case "$STEPS" in
    500k)  MODEL_PATH="$DATA_DIR/checkpoints/sac_residual_500000_steps.zip" ;;
    750k)  MODEL_PATH="$DATA_DIR/checkpoints/sac_residual_750000_steps.zip" ;;
    1m|1M) MODEL_PATH="$DATA_DIR/sac_final.zip" ;;
    *)
        echo "Error: unknown step count '$STEPS'. Use: 500k, 750k, or 1m"
        exit 1
        ;;
esac

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: model not found at $MODEL_PATH"
    exit 1
fi

echo "Launching $MODEL_TYPE SAC @ $STEPS in Meshcat..."
echo "  Model: $MODEL_PATH"
echo ""
exec python -m src.evaluate --model "$MODEL_PATH" --render --realtime --episodes 1 "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
