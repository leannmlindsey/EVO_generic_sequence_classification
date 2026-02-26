#!/bin/bash
# Evo Inference - Configuration Wrapper
#
# This script sets configuration variables and submits the inference
# job to SLURM (or runs interactively).
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_inference.sh
#
# For interactive mode (no SLURM):
#   bash wrapper_run_inference.sh --interactive

set -e  # Exit on error

# ============================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================

# Required: Text file with one input CSV path per line
# OR: Set INPUT_CSVS to a space-separated list of CSV paths
export INPUT_LIST="/path/to/input_files.txt"
# export INPUT_CSVS="/path/to/file1.csv /path/to/file2.csv"

# Required: Directory containing trained model artifacts
# (linear_probe.pkl, linear_probe_scaler.pkl, three_layer_nn_pretrained.pt, three_layer_nn_scaler.pkl)
export MODEL_DIR="/path/to/trained/model"

# Required: Output directory for predictions
export OUTPUT_DIR=""  # Leave empty for auto-generated path

# What to run (set to "true" to enable)
export RUN_NN="true"
export RUN_LP="true"

# Evo model configuration (must match what was used during training)
export MODEL_NAME="evo-1-8k-base"  # Options: evo-1.5-8k-base, evo-1-8k-base, evo-1-131k-base, evo-1-8k-crispr, evo-1-8k-transposon
export BATCH_SIZE="8"              # Reduce if running out of GPU memory
export MAX_LENGTH="8192"           # Maximum sequence length
export POOLING="mean"              # Pooling strategy: mean, first, last

# Layer selection (leave empty for final layer, which is recommended)
# export LAYER_IDX="-1"

# Classification threshold
export THRESHOLD="0.5"

# Optional flags
export SAVE_EMBEDDINGS="false"     # Save extracted embeddings for reuse
export SAVE_METRICS="true"         # Calculate metrics (requires "label" column)

# Device for inference
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

# Validate inputs
if [ "$INPUT_LIST" = "/path/to/input_files.txt" ] && [ -z "$INPUT_CSVS" ]; then
    echo "ERROR: Please edit INPUT_LIST or INPUT_CSVS in this script"
    echo "       INPUT_LIST: text file with one CSV path per line"
    echo "       INPUT_CSVS: space-separated list of CSV paths"
    exit 1
fi

if [ -n "$INPUT_LIST" ] && [ "$INPUT_LIST" != "/path/to/input_files.txt" ] && [ ! -f "$INPUT_LIST" ]; then
    echo "ERROR: INPUT_LIST file not found: $INPUT_LIST"
    exit 1
fi

if [ "$MODEL_DIR" = "/path/to/trained/model" ]; then
    echo "ERROR: Please edit MODEL_DIR in this script"
    echo "       Set it to the directory containing trained model artifacts"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR"
    exit 1
fi

# Check for required model artifacts
if [ "$RUN_LP" = "true" ]; then
    if [ ! -f "$MODEL_DIR/linear_probe.pkl" ] || [ ! -f "$MODEL_DIR/linear_probe_scaler.pkl" ]; then
        echo "ERROR: LP artifacts not found in $MODEL_DIR"
        echo "  Expected: linear_probe.pkl, linear_probe_scaler.pkl"
        exit 1
    fi
fi

if [ "$RUN_NN" = "true" ]; then
    if [ ! -f "$MODEL_DIR/three_layer_nn_pretrained.pt" ] || [ ! -f "$MODEL_DIR/three_layer_nn_scaler.pkl" ]; then
        echo "ERROR: NN artifacts not found in $MODEL_DIR"
        echo "  Expected: three_layer_nn_pretrained.pt, three_layer_nn_scaler.pkl"
        exit 1
    fi
fi

echo "=========================================="
echo "Evo Inference - Job Submission"
echo "=========================================="
echo "MODEL_DIR: $MODEL_DIR"
echo "MODEL_NAME: $MODEL_NAME"
echo "RUN_NN: $RUN_NN"
echo "RUN_LP: $RUN_LP"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "POOLING: $POOLING"
echo "=========================================="

# Check for interactive mode
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    echo "Running in interactive mode..."
    echo ""
    bash "$SCRIPT_DIR/run_inference_interactive.sh"
    exit $?
fi

# Check if SLURM is available
if ! command -v sbatch &> /dev/null; then
    echo "SLURM (sbatch) not found. Running in interactive mode instead..."
    echo ""
    bash "$SCRIPT_DIR/run_inference_interactive.sh"
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
    "$SCRIPT_DIR/run_inference.sh" \
    | awk '{print $4}')

echo "=========================================="
echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor with: squeue -j $JOB_ID"
echo "View output:  tail -f evo_inference_${JOB_ID}.out"
echo "View errors:  tail -f evo_inference_${JOB_ID}.err"
echo "=========================================="
