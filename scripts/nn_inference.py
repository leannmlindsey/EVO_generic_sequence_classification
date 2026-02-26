#!/usr/bin/env python
"""
Neural Network Inference for Evo

Loads a trained 3-layer neural network checkpoint and scaler, extracts
embeddings from new DNA sequences, and runs classification.

Usage:
    # With pre-extracted embeddings (no GPU model load needed):
    python scripts/nn_inference.py \
        --input_csv new_data.csv \
        --checkpoint_path model_dir/three_layer_nn_pretrained.pt \
        --scaler_path model_dir/three_layer_nn_scaler.pkl \
        --embeddings_path embeddings/embeddings.npz \
        --output_csv predictions.csv

    # Extract embeddings on-the-fly (requires GPU):
    python scripts/nn_inference.py \
        --input_csv new_data.csv \
        --checkpoint_path model_dir/three_layer_nn_pretrained.pt \
        --scaler_path model_dir/three_layer_nn_scaler.pkl \
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
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Run neural network inference on new DNA sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        '--input_csv', type=str, required=True,
        help='Path to CSV file with a "sequence" column (and optional "label" column)',
    )
    parser.add_argument(
        '--checkpoint_path', type=str, required=True,
        help='Path to trained NN checkpoint (.pt)',
    )
    parser.add_argument(
        '--scaler_path', type=str, required=True,
        help='Path to trained scaler (.pkl)',
    )

    # Output
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help='Path for output predictions CSV (default: <input>_nn_predictions.csv)',
    )

    # Embedding source (pre-extracted or on-the-fly)
    parser.add_argument(
        '--embeddings_path', type=str, default=None,
        help='Path to pre-extracted embeddings (.npz). Skips Evo model loading if provided.',
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
        help='Device for inference',
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
    parser.add_argument(
        '--save_embeddings', action='store_true',
        help='Save extracted embeddings to output directory for reuse with other scripts',
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"ERROR: Input CSV not found: {args.input_csv}")
        sys.exit(1)

    for path_arg, name in [(args.checkpoint_path, 'checkpoint'), (args.scaler_path, 'scaler')]:
        if not Path(path_arg).exists():
            print(f"ERROR: {name} file not found: {path_arg}")
            sys.exit(1)

    # Set default output path
    if args.output_csv is None:
        args.output_csv = str(input_path.parent / f'{input_path.stem}_nn_predictions.csv')

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

    # Load NN checkpoint and scaler
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)

    # Support both old format (state_dict only) and new format (dict with metadata)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        state_dict = checkpoint['model_state_dict']
    else:
        # Legacy format: bare state_dict, infer dimensions from weights
        state_dict = checkpoint
        input_dim = state_dict['fc1.weight'].shape[1]
        hidden_dim = state_dict['fc1.weight'].shape[0]

    print(f"  input_dim={input_dim}, hidden_dim={hidden_dim}")

    print(f"Loading scaler: {args.scaler_path}")
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Initialize NN model
    from evo.embedding_analysis import ThreeLayerNN

    nn_device = args.device if torch.cuda.is_available() else 'cpu'
    model = ThreeLayerNN(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.to(nn_device)
    model.eval()
    print(f"Loaded neural network on {nn_device}")

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

        # Save embeddings if requested
        if args.save_embeddings:
            output_dir = Path(args.output_csv).parent
            emb_path = output_dir / 'embeddings.npz'
            save_dict = {'embeddings': embeddings}
            if has_labels:
                save_dict['labels'] = labels
            np.savez(emb_path, **save_dict)
            print(f"Saved embeddings to {emb_path}")

    # Run inference
    print("\nRunning neural network inference...")
    scaled = scaler.transform(embeddings)
    X = torch.tensor(scaled, dtype=torch.float32).to(nn_device)

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        pred_labels = outputs.argmax(dim=1).cpu().numpy()

    prob_0 = probs[:, 0]
    prob_1 = probs[:, 1]

    # Apply custom threshold if not default
    if args.threshold != 0.5:
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
            metrics_json = {k: float(v) for k, v in metrics.items()}
            with open(metrics_path, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print(f"Saved metrics to {metrics_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
