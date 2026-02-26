#!/usr/bin/env python
"""
Linear Probe Inference for Evo

Loads a trained linear probe classifier and scaler, extracts embeddings
from new DNA sequences, and runs classification.

Usage:
    # With pre-extracted embeddings (no GPU needed):
    python scripts/lp_inference.py \
        --input_csv new_data.csv \
        --classifier_path model_dir/linear_probe.pkl \
        --scaler_path model_dir/linear_probe_scaler.pkl \
        --embeddings_path embeddings/embeddings.npz \
        --output_csv predictions.csv

    # Extract embeddings on-the-fly (requires GPU):
    python scripts/lp_inference.py \
        --input_csv new_data.csv \
        --classifier_path model_dir/linear_probe.pkl \
        --scaler_path model_dir/linear_probe_scaler.pkl \
        --model_name evo-1-8k-base \
        --output_csv predictions.csv
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Run linear probe inference on new DNA sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        '--input_csv', type=str, required=True,
        help='Path to CSV file with a "sequence" column (and optional "label" column)',
    )
    parser.add_argument(
        '--classifier_path', type=str, required=True,
        help='Path to trained linear probe classifier (.pkl)',
    )
    parser.add_argument(
        '--scaler_path', type=str, required=True,
        help='Path to trained scaler (.pkl)',
    )

    # Output
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help='Path for output predictions CSV (default: <input>_lp_predictions.csv)',
    )

    # Embedding source (pre-extracted or on-the-fly)
    parser.add_argument(
        '--embeddings_path', type=str, default=None,
        help='Path to pre-extracted embeddings (.npz). Skips model loading if provided.',
    )

    # Model config (only needed if --embeddings_path not provided)
    parser.add_argument(
        '--model_name', type=str, default='evo-1-8k-base',
        help='Evo model name (only used if embeddings_path not provided)',
    )
    parser.add_argument(
        '--pooling', type=str, default='mean',
        choices=['mean', 'first', 'last'],
        help='Pooling strategy (only used if embeddings_path not provided)',
    )
    parser.add_argument(
        '--layer_idx', type=int, default=None,
        help='Layer index for intermediate embeddings (only used if embeddings_path not provided)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for embedding extraction',
    )
    parser.add_argument(
        '--max_length', type=int, default=8192,
        help='Maximum sequence length',
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for model inference',
    )

    # Classification options
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Classification threshold for positive class',
    )
    parser.add_argument(
        '--save_metrics', action='store_true',
        help='Calculate and save metrics (requires "label" column in input CSV)',
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"ERROR: Input CSV not found: {args.input_csv}")
        sys.exit(1)

    for path_arg, name in [(args.classifier_path, 'classifier'), (args.scaler_path, 'scaler')]:
        if not Path(path_arg).exists():
            print(f"ERROR: {name} file not found: {path_arg}")
            sys.exit(1)

    # Set default output path
    if args.output_csv is None:
        args.output_csv = str(input_path.parent / f'{input_path.stem}_lp_predictions.csv')

    # Read input CSV
    print(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if 'sequence' not in df.columns:
        print(f"ERROR: CSV must have a 'sequence' column. Found: {df.columns.tolist()}")
        sys.exit(1)

    sequences = df['sequence'].tolist()
    has_labels = 'label' in df.columns
    labels = df['label'].astype(int).values if has_labels else None

    print(f"Loaded {len(sequences)} sequences")

    # Load classifier and scaler
    print(f"Loading classifier: {args.classifier_path}")
    with open(args.classifier_path, 'rb') as f:
        classifier = pickle.load(f)

    print(f"Loading scaler: {args.scaler_path}")
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Get embeddings
    if args.embeddings_path is not None:
        # Load pre-extracted embeddings
        print(f"Loading pre-extracted embeddings: {args.embeddings_path}")
        data = np.load(args.embeddings_path)
        embeddings = data['embeddings']
        if embeddings.shape[0] != len(sequences):
            print(f"WARNING: Embeddings count ({embeddings.shape[0]}) != sequences count ({len(sequences)})")
        print(f"Loaded embeddings: {embeddings.shape}")
    else:
        # Extract embeddings on-the-fly
        from evo.embedding_analysis import EvoEmbeddingExtractor, extract_embeddings

        print(f"\nInitializing Evo model: {args.model_name}")
        extractor = EvoEmbeddingExtractor(
            model_name=args.model_name,
            device=args.device,
            layer_idx=args.layer_idx,
            pooling=args.pooling,
        )

        dummy_labels = [0] * len(sequences)
        print(f"Extracting embeddings (batch_size={args.batch_size})...")
        start_time = time.time()
        embeddings, _ = extract_embeddings(
            extractor, sequences, dummy_labels,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print(f"Extraction complete: {embeddings.shape} in {time.time() - start_time:.1f}s")
        extractor.restore_model()

    # Run inference
    print("\nRunning linear probe inference...")
    scaled = scaler.transform(embeddings)
    probabilities = classifier.predict_proba(scaled)
    prob_0 = probabilities[:, 0]
    prob_1 = probabilities[:, 1]
    pred_labels = (prob_1 >= args.threshold).astype(int)

    # Build output DataFrame
    output_df = pd.DataFrame({
        'prob_0': prob_0,
        'prob_1': prob_1,
        'pred_label': pred_labels,
    })

    if has_labels:
        output_df.insert(0, 'label', labels)

    # Save predictions
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # Print summary
    print(f"\nPrediction summary:")
    print(f"  Total: {len(pred_labels)}")
    print(f"  Predicted 0: {(pred_labels == 0).sum()}")
    print(f"  Predicted 1: {(pred_labels == 1).sum()}")

    # Calculate and save metrics if requested
    if args.save_metrics:
        if not has_labels:
            print("WARNING: --save_metrics requires 'label' column in input CSV. Skipping.")
        else:
            from evo.embedding_analysis import calculate_metrics

            metrics = calculate_metrics(labels, pred_labels, prob_1)
            print(f"\nMetrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            metrics_path = output_path.parent / f'{output_path.stem}_metrics.json'
            # Convert numpy values to Python floats for JSON serialization
            metrics_json = {k: float(v) for k, v in metrics.items()}
            with open(metrics_path, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print(f"Saved metrics to {metrics_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
