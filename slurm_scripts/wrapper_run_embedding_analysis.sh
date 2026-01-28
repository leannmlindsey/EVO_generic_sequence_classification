#!/bin/bash
# Evo Embedding Analysis - Configuration Wrapper
#
# This script sets configuration variables and submits the embedding
# analysis job to SLURM (or runs interactively).
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_embedding_analysis.sh
#
# For interactive mode (no SLURM):
#   bash wrapper_run_embedding_analysis.sh --interactive

set -e  # Exit on error

# ============================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================

# Required: Path to directory containing train.csv, dev.csv/val.csv, test.csv
# Each CSV should have columns: 'sequence' (DNA), 'label' (0 or 1)
export CSV_DIR="/path/to/your/csv/data"

# Optional: Output directory (default: ./results/embeddings/$(basename $CSV_DIR))
# Leave empty to use default with dataset name
export OUTPUT_DIR=""

# Model configuration
export MODEL_NAME="evo-1-8k-base"  # Options: evo-1.5-8k-base, evo-1-8k-base, evo-1-131k-base, evo-1-8k-crispr, evo-1-8k-transposon

# Processing configuration
export BATCH_SIZE="8"              # Reduce if running out of GPU memory
export MAX_LENGTH="8192"           # Maximum sequence length (8192 for 8k models, 131072 for 131k model)
export POOLING="mean"              # Pooling strategy: mean, first, last

# Layer selection (leave empty for final layer, which is recommended)
# Use negative indexing: -1 = last layer, -2 = second-to-last, etc.
# export LAYER_IDX="-1"

# Neural network training configuration
export NN_EPOCHS="100"
export NN_HIDDEN_DIM="256"
export NN_LR="0.001"

# Random seed for reproducibility
export SEED="42"

# Optional flags (set to "true" to enable)
export INCLUDE_RANDOM_BASELINE="false"  # Compare with random embeddings
export SKIP_NN="false"                  # Only run linear probe, skip NN
export CACHE_EMBEDDINGS="true"          # Cache embeddings for reuse

# Device for inference
# NOTE: If CUDA_VISIBLE_DEVICES is set, use cuda:0 since that becomes the first visible GPU
# Example: export CUDA_VISIBLE_DEVICES=3 means GPU 3 becomes cuda:0
export DEVICE="cuda:0"

# SLURM configuration (edit for your cluster)
SLURM_PARTITION="gpu"
SLURM_TIME="04:00:00"
SLURM_MEM="64G"
SLURM_GPUS="1"
SLURM_CPUS="8"

# ============================================================
# END CONFIGURATION
# ============================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate CSV_DIR
if [ "$CSV_DIR" = "/path/to/your/csv/data" ]; then
    echo "ERROR: Please edit CSV_DIR in this script"
    echo "       Set it to the directory containing your train.csv, dev.csv/val.csv, and test.csv files"
    exit 1
fi

if [ ! -d "$CSV_DIR" ]; then
    echo "ERROR: CSV_DIR does not exist: $CSV_DIR"
    exit 1
fi

# Check for required CSV files
MISSING_FILES=()
if [ ! -f "$CSV_DIR/train.csv" ]; then
    MISSING_FILES+=("train.csv")
fi
if [ ! -f "$CSV_DIR/test.csv" ]; then
    MISSING_FILES+=("test.csv")
fi
if [ ! -f "$CSV_DIR/dev.csv" ] && [ ! -f "$CSV_DIR/val.csv" ]; then
    MISSING_FILES+=("dev.csv or val.csv")
fi

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "ERROR: Missing required files in $CSV_DIR:"
    for FILE in "${MISSING_FILES[@]}"; do
        echo "  - $FILE"
    done
    exit 1
fi

echo "=========================================="
echo "Evo Embedding Analysis - Job Submission"
echo "=========================================="
echo "CSV_DIR: $CSV_DIR"
echo "MODEL_NAME: $MODEL_NAME"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "POOLING: $POOLING"
echo "SEED: $SEED"
echo "=========================================="

# Check for interactive mode
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    echo "Running in interactive mode..."
    echo ""
    bash "$SCRIPT_DIR/run_embedding_analysis_interactive.sh"
    exit $?
fi

# Check if SLURM is available
if ! command -v sbatch &> /dev/null; then
    echo "SLURM (sbatch) not found. Running in interactive mode instead..."
    echo ""
    bash "$SCRIPT_DIR/run_embedding_analysis_interactive.sh"
    exit $?
fi

# Submit SLURM job
echo "Submitting SLURM job..."
echo ""

JOB_ID=$(sbatch \
    --partition="$SLURM_PARTITION" \
    --time="$SLURM_TIME" \
    --mem="$SLURM_MEM" \
    --gres=gpu:"$SLURM_GPUS" \
    --cpus-per-task="$SLURM_CPUS" \
    --export=ALL \
    "$SCRIPT_DIR/run_embedding_analysis.sh" \
    | awk '{print $4}')

echo "=========================================="
echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor with: squeue -j $JOB_ID"
echo "View output:  tail -f evo_embedding_${JOB_ID}.out"
echo "View errors:  tail -f evo_embedding_${JOB_ID}.err"
echo "=========================================="
