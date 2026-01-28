#!/bin/bash
# Evo Embedding Analysis - Interactive/Local Script
#
# Use this script for running embedding analysis interactively
# without SLURM (e.g., on a local workstation with GPU).
#
# Expected environment variables (set by wrapper script or manually):
#   CSV_DIR         - Directory containing train.csv, dev.csv/val.csv, test.csv
#   OUTPUT_DIR      - Output directory for results (optional)
#   MODEL_NAME      - Evo model name (optional, default: evo-1-8k-base)
#   BATCH_SIZE      - Batch size (optional, default: 8)
#   MAX_LENGTH      - Maximum sequence length (optional, default: 8192)
#   POOLING         - Pooling strategy (optional, default: mean)
#   LAYER_IDX       - Layer index (optional, default: None for final layer)
#   NN_EPOCHS       - Neural network epochs (optional, default: 100)
#   NN_HIDDEN_DIM   - Neural network hidden dim (optional, default: 256)
#   NN_LR           - Neural network learning rate (optional, default: 0.001)
#   SEED            - Random seed for reproducibility (optional, default: 42)
#   INCLUDE_RANDOM_BASELINE - Include random baseline (optional, default: false)
#   SKIP_NN         - Skip neural network training (optional, default: false)
#   CACHE_EMBEDDINGS - Cache embeddings to disk (optional, default: false)
#   DEVICE          - Device for inference (optional, default: cuda:0)

set -e  # Exit on error

echo "=========================================="
echo "Evo Embedding Analysis (Interactive Mode)"
echo "=========================================="
echo "Start time: $(date)"
echo "=========================================="

# Validate required environment variable
if [ -z "$CSV_DIR" ]; then
    echo "ERROR: CSV_DIR environment variable not set"
    echo ""
    echo "Usage: export CSV_DIR=/path/to/csv/data && bash $0"
    echo ""
    echo "Or source the wrapper script first:"
    echo "  source wrapper_run_embedding_analysis.sh"
    exit 1
fi

if [ ! -d "$CSV_DIR" ]; then
    echo "ERROR: CSV_DIR does not exist: $CSV_DIR"
    exit 1
fi

# Check for required CSV files
for FILE in train.csv test.csv; do
    if [ ! -f "$CSV_DIR/$FILE" ]; then
        echo "ERROR: Required file not found: $CSV_DIR/$FILE"
        exit 1
    fi
done

# Check for dev.csv or val.csv
if [ ! -f "$CSV_DIR/dev.csv" ] && [ ! -f "$CSV_DIR/val.csv" ]; then
    echo "ERROR: Neither dev.csv nor val.csv found in $CSV_DIR"
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
NN_EPOCHS="${NN_EPOCHS:-100}"
NN_HIDDEN_DIM="${NN_HIDDEN_DIM:-256}"
NN_LR="${NN_LR:-0.001}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"

# Construct output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DATASET_NAME=$(basename ${CSV_DIR})
    OUTPUT_DIR="./results/embeddings/${DATASET_NAME}/${MODEL_NAME}_len${MAX_LENGTH}_${POOLING}_seed${SEED}_${TIMESTAMP}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  CSV_DIR: $CSV_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  MODEL_NAME: $MODEL_NAME"
echo "  DEVICE: $DEVICE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  MAX_LENGTH: $MAX_LENGTH"
echo "  POOLING: $POOLING"
echo "  LAYER_IDX: ${LAYER_IDX:-'None (final layer)'}"
echo "  NN_EPOCHS: $NN_EPOCHS"
echo "  NN_HIDDEN_DIM: $NN_HIDDEN_DIM"
echo "  NN_LR: $NN_LR"
echo "  SEED: $SEED"
echo "=========================================="

# Build command arguments
CMD_ARGS=(
    --csv_dir "$CSV_DIR"
    --output_dir "$OUTPUT_DIR"
    --model_name "$MODEL_NAME"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --pooling "$POOLING"
    --nn_epochs "$NN_EPOCHS"
    --nn_hidden_dim "$NN_HIDDEN_DIM"
    --nn_lr "$NN_LR"
    --seed "$SEED"
)

# Add layer_idx if specified
if [ -n "$LAYER_IDX" ]; then
    CMD_ARGS+=(--layer_idx "$LAYER_IDX")
fi

# Add optional flags
if [ "$INCLUDE_RANDOM_BASELINE" = "true" ]; then
    CMD_ARGS+=(--include_random_baseline)
fi

if [ "$SKIP_NN" = "true" ]; then
    CMD_ARGS+=(--skip_nn)
fi

if [ "$CACHE_EMBEDDINGS" = "true" ]; then
    CMD_ARGS+=(--cache_embeddings)
fi

# Set PYTHONPATH to include the evo repo (in case not installed in editable mode)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVO_REPO_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${EVO_REPO_DIR}:${PYTHONPATH}"

# Run embedding analysis
echo "Running: python -m evo.embedding_analysis ${CMD_ARGS[*]}"
echo ""

python -m evo.embedding_analysis "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
