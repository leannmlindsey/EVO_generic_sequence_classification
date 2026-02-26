#!/bin/bash
#SBATCH --job-name=evo_inference
#SBATCH --output=evo_inference_%j.out
#SBATCH --error=evo_inference_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Evo Inference SLURM Script
#
# Runs batch inference using trained classifiers on new DNA sequences.
#
# Expected environment variables (set by wrapper script):
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

echo "=========================================="
echo "Evo Inference"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules (adjust for your cluster)
# module load cuda/11.8
# module load anaconda3

# Activate conda environment (adjust for your setup)
# source activate evo

# Print environment info
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Validate required environment variable
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR environment variable not set"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR"
    exit 1
fi

# Set defaults
MODEL_NAME="${MODEL_NAME:-evo-1-8k-base}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
POOLING="${POOLING:-mean}"
THRESHOLD="${THRESHOLD:-0.5}"

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
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
