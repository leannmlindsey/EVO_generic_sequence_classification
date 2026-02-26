#!/bin/bash
# Evo Inference - Interactive/Local Script
#
# Use this script for running inference interactively
# without SLURM (e.g., on a local workstation with GPU).
#
# Expected environment variables (set by wrapper script or manually):
#   INPUT_LIST      - Text file with one CSV path per line (optional)
#   INPUT_CSVS      - Space-separated list of CSV paths (optional)
#   MODEL_DIR       - Directory containing trained model artifacts
#   OUTPUT_DIR      - Output directory for predictions (optional)
#   MODEL_NAME      - Evo model name (optional, default: evo-1-8k-base)
#   BATCH_SIZE      - Batch size (optional, default: 8)
#   MAX_LENGTH      - Maximum sequence length (optional, default: 8192)
#   POOLING         - Pooling strategy (optional, default: mean)
#   LAYER_IDX       - Layer index (optional, default: None for final layer)
#   THRESHOLD       - Classification threshold (optional, default: 0.5)
#   RUN_NN          - Run neural network inference (optional, default: false)
#   RUN_LP          - Run linear probe inference (optional, default: false)
#   SAVE_EMBEDDINGS - Save extracted embeddings (optional, default: false)
#   SAVE_METRICS    - Calculate and save metrics (optional, default: false)
#   DEVICE          - Device for inference (optional, default: cuda:0)

set -e  # Exit on error

echo "=========================================="
echo "Evo Inference (Interactive Mode)"
echo "=========================================="
echo "Start time: $(date)"
echo "=========================================="

# Validate required environment variable
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR environment variable not set"
    echo ""
    echo "Usage: export MODEL_DIR=/path/to/model && export INPUT_LIST=/path/to/files.txt && bash $0"
    echo ""
    echo "Or source the wrapper script first:"
    echo "  source wrapper_run_inference.sh"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR"
    exit 1
fi

# Print environment info
echo "Python: $(which python)"
if python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q "True"; then
    echo "CUDA available: True"
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo ""
    echo "GPU Information:"
    nvidia-smi
else
    echo "CUDA available: False (will use CPU - this will be slow!)"
fi
echo "=========================================="

# Set defaults
MODEL_NAME="${MODEL_NAME:-evo-1-8k-base}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
POOLING="${POOLING:-mean}"
THRESHOLD="${THRESHOLD:-0.5}"
DEVICE="${DEVICE:-cuda:0}"

# Construct output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="./results/inference/${MODEL_NAME}_${POOLING}_${TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  MODEL_DIR: $MODEL_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  MODEL_NAME: $MODEL_NAME"
echo "  DEVICE: $DEVICE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  MAX_LENGTH: $MAX_LENGTH"
echo "  POOLING: $POOLING"
echo "  THRESHOLD: $THRESHOLD"
echo "  RUN_NN: ${RUN_NN:-false}"
echo "  RUN_LP: ${RUN_LP:-false}"
echo "=========================================="

# Build command arguments
CMD_ARGS=(
    --model_dir "$MODEL_DIR"
    --output_dir "$OUTPUT_DIR"
    --model_name "$MODEL_NAME"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --pooling "$POOLING"
    --threshold "$THRESHOLD"
)

# Add input source
if [ -n "$INPUT_LIST" ] && [ -f "$INPUT_LIST" ]; then
    CMD_ARGS+=(--input_list "$INPUT_LIST")
elif [ -n "$INPUT_CSVS" ]; then
    CMD_ARGS+=(--input_csvs $INPUT_CSVS)
else
    echo "ERROR: Neither INPUT_LIST nor INPUT_CSVS specified"
    exit 1
fi

# Add layer_idx if specified
if [ -n "$LAYER_IDX" ]; then
    CMD_ARGS+=(--layer_idx "$LAYER_IDX")
fi

# Add inference flags
if [ "$RUN_NN" = "true" ]; then
    CMD_ARGS+=(--run_nn)
fi

if [ "$RUN_LP" = "true" ]; then
    CMD_ARGS+=(--run_lp)
fi

if [ "$SAVE_EMBEDDINGS" = "true" ]; then
    CMD_ARGS+=(--save_embeddings)
fi

if [ "$SAVE_METRICS" = "true" ]; then
    CMD_ARGS+=(--save_metrics)
fi

# Set PYTHONPATH to include the evo repo (in case not installed in editable mode)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO_REPO_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${EVO_REPO_DIR}:${PYTHONPATH}"

# Run batch inference
echo "Running: python scripts/batch_inference.py ${CMD_ARGS[*]}"
echo ""

python "${EVO_REPO_DIR}/scripts/batch_inference.py" "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
