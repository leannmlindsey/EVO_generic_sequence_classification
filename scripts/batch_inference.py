#!/usr/bin/env python
"""
Batch Inference for Evo (NN + LP)

Loads the Evo model ONCE and processes multiple input CSV files.
Extracts embeddings once per file, then runs NN and/or LP inference.

Usage:
    python scripts/batch_inference.py \
        --input_list files.txt \
        --model_dir trained_model/ \
        --output_dir results/ \
        --run_nn --run_lp \
        --model_name evo-1-8k-base

    The --input_list file should contain one CSV path per line.
    Alternatively, use --input_csvs to pass CSV paths directly.
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


def run_lp_inference(embeddings, classifier, scaler, threshold=0.5):
    """Run linear probe inference on embeddings."""
    scaled = scaler.transform(embeddings)
    probabilities = classifier.predict_proba(scaled)
    prob_0 = probabilities[:, 0]
    prob_1 = probabilities[:, 1]
    pred_labels = (prob_1 >= threshold).astype(int)
    return prob_0, prob_1, pred_labels


def run_nn_inference(embeddings, model, scaler, device, threshold=0.5):
    """Run neural network inference on embeddings."""
    scaled = scaler.transform(embeddings)
    X = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        pred_labels = outputs.argmax(dim=1).cpu().numpy()

    prob_0 = probs[:, 0]
    prob_1 = probs[:, 1]

    if threshold != 0.5:
        pred_labels = (prob_1 >= threshold).astype(int)

    return prob_0, prob_1, pred_labels


def main():
    parser = argparse.ArgumentParser(
        description='Batch inference: load Evo model once, run NN+LP on multiple files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input specification
    parser.add_argument(
        '--input_list', type=str, default=None,
        help='Text file with one CSV path per line',
    )
    parser.add_argument(
        '--input_csvs', type=str, nargs='*', default=None,
        help='CSV file paths to process',
    )

    # Model artifacts directory
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Directory containing trained model artifacts (pkl, pt files)',
    )

    # Output
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory for output predictions',
    )

    # What to run
    parser.add_argument(
        '--run_nn', action='store_true',
        help='Run neural network inference',
    )
    parser.add_argument(
        '--run_lp', action='store_true',
        help='Run linear probe inference',
    )

    # Evo model config
    parser.add_argument(
        '--model_name', type=str, default='evo-1-8k-base',
        help='Evo model name for embedding extraction',
    )
    parser.add_argument(
        '--pooling', type=str, default='mean',
        choices=['mean', 'first', 'last'],
        help='Pooling strategy',
    )
    parser.add_argument(
        '--layer_idx', type=int, default=None,
        help='Layer index for intermediate embeddings',
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
        help='Classification threshold',
    )
    parser.add_argument(
        '--save_embeddings', action='store_true',
        help='Save extracted embeddings for each file',
    )
    parser.add_argument(
        '--save_metrics', action='store_true',
        help='Calculate and save metrics (requires "label" column)',
    )

    args = parser.parse_args()

    if not args.run_nn and not args.run_lp:
        print("ERROR: Specify at least one of --run_nn or --run_lp")
        sys.exit(1)

    # Collect input files
    csv_files = []
    if args.input_list:
        list_path = Path(args.input_list)
        if not list_path.exists():
            print(f"ERROR: Input list not found: {args.input_list}")
            sys.exit(1)
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    csv_files.append(line)

    if args.input_csvs:
        csv_files.extend(args.input_csvs)

    if not csv_files:
        print("ERROR: No input files specified. Use --input_list or --input_csvs.")
        sys.exit(1)

    # Validate input files
    valid_files = []
    for csv_path in csv_files:
        if Path(csv_path).exists():
            valid_files.append(csv_path)
        else:
            print(f"WARNING: File not found, skipping: {csv_path}")
    csv_files = valid_files

    if not csv_files:
        print("ERROR: No valid input files found.")
        sys.exit(1)

    print(f"Will process {len(csv_files)} file(s)")

    # Create output directory
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classifier artifacts
    lp_classifier = None
    lp_scaler = None
    nn_model = None
    nn_scaler = None
    nn_device = args.device if torch.cuda.is_available() else 'cpu'

    if args.run_lp:
        lp_clf_path = model_dir / 'linear_probe.pkl'
        lp_scl_path = model_dir / 'linear_probe_scaler.pkl'
        if not lp_clf_path.exists() or not lp_scl_path.exists():
            print(f"ERROR: LP artifacts not found in {model_dir}")
            print(f"  Expected: linear_probe.pkl, linear_probe_scaler.pkl")
            sys.exit(1)

        print(f"Loading linear probe classifier: {lp_clf_path}")
        with open(lp_clf_path, 'rb') as f:
            lp_classifier = pickle.load(f)
        with open(lp_scl_path, 'rb') as f:
            lp_scaler = pickle.load(f)

    if args.run_nn:
        nn_ckpt_path = model_dir / 'three_layer_nn_pretrained.pt'
        nn_scl_path = model_dir / 'three_layer_nn_scaler.pkl'
        if not nn_ckpt_path.exists() or not nn_scl_path.exists():
            print(f"ERROR: NN artifacts not found in {model_dir}")
            print(f"  Expected: three_layer_nn_pretrained.pt, three_layer_nn_scaler.pkl")
            sys.exit(1)

        print(f"Loading NN checkpoint: {nn_ckpt_path}")
        checkpoint = torch.load(nn_ckpt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            input_dim = checkpoint['input_dim']
            hidden_dim = checkpoint['hidden_dim']
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            input_dim = state_dict['fc1.weight'].shape[1]
            hidden_dim = state_dict['fc1.weight'].shape[0]

        from evo.embedding_analysis import ThreeLayerNN
        nn_model = ThreeLayerNN(input_dim=input_dim, hidden_dim=hidden_dim)
        nn_model.load_state_dict(state_dict)
        nn_model.to(nn_device)
        nn_model.eval()

        with open(nn_scl_path, 'rb') as f:
            nn_scaler = pickle.load(f)

    # Load Evo model ONCE for embedding extraction
    from evo.embedding_analysis import EvoEmbeddingExtractor, extract_embeddings

    print(f"\nInitializing Evo model: {args.model_name}")
    extractor = EvoEmbeddingExtractor(
        model_name=args.model_name,
        device=args.device,
        layer_idx=args.layer_idx,
        pooling=args.pooling,
    )

    # Process each file
    all_metrics = {}
    total_start = time.time()

    for file_idx, csv_path in enumerate(csv_files):
        print(f"\n{'='*60}")
        print(f"Processing file {file_idx + 1}/{len(csv_files)}: {csv_path}")
        print(f"{'='*60}")

        # Read CSV
        df = pd.read_csv(csv_path)
        if 'sequence' not in df.columns:
            print(f"  SKIP: No 'sequence' column in {csv_path}")
            continue

        sequences = df['sequence'].tolist()
        has_labels = 'label' in df.columns
        labels = df['label'].astype(int).values if has_labels else None
        file_stem = Path(csv_path).stem

        print(f"  Sequences: {len(sequences)}")

        # Extract embeddings
        dummy_labels = [0] * len(sequences)
        start_time = time.time()
        embeddings, _ = extract_embeddings(
            extractor, sequences, dummy_labels,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        emb_time = time.time() - start_time
        print(f"  Embeddings: {embeddings.shape} ({emb_time:.1f}s)")

        # Save embeddings if requested
        if args.save_embeddings:
            emb_path = output_dir / f'{file_stem}_embeddings.npz'
            save_dict = {'embeddings': embeddings}
            if has_labels:
                save_dict['labels'] = labels
            np.savez(emb_path, **save_dict)
            print(f"  Saved embeddings to {emb_path}")

        file_metrics = {}

        # Run LP inference
        if args.run_lp and lp_classifier is not None:
            prob_0, prob_1, pred_labels = run_lp_inference(
                embeddings, lp_classifier, lp_scaler, args.threshold,
            )

            lp_df = pd.DataFrame({
                'prob_0': prob_0, 'prob_1': prob_1, 'pred_label': pred_labels,
            })
            if has_labels:
                lp_df.insert(0, 'label', labels)

            lp_path = output_dir / f'{file_stem}_lp_predictions.csv'
            lp_df.to_csv(lp_path, index=False)
            print(f"  LP predictions saved to {lp_path}")

            if args.save_metrics and has_labels:
                from evo.embedding_analysis import calculate_metrics
                lp_metrics = calculate_metrics(labels, pred_labels, prob_1)
                file_metrics['linear_probe'] = {k: float(v) for k, v in lp_metrics.items()}
                print(f"  LP - Acc: {lp_metrics['accuracy']:.4f}, F1: {lp_metrics['f1']:.4f}, MCC: {lp_metrics['mcc']:.4f}")

        # Run NN inference
        if args.run_nn and nn_model is not None:
            prob_0, prob_1, pred_labels = run_nn_inference(
                embeddings, nn_model, nn_scaler, nn_device, args.threshold,
            )

            nn_df = pd.DataFrame({
                'prob_0': prob_0, 'prob_1': prob_1, 'pred_label': pred_labels,
            })
            if has_labels:
                nn_df.insert(0, 'label', labels)

            nn_path = output_dir / f'{file_stem}_nn_predictions.csv'
            nn_df.to_csv(nn_path, index=False)
            print(f"  NN predictions saved to {nn_path}")

            if args.save_metrics and has_labels:
                from evo.embedding_analysis import calculate_metrics
                nn_metrics = calculate_metrics(labels, pred_labels, prob_1)
                file_metrics['neural_network'] = {k: float(v) for k, v in nn_metrics.items()}
                print(f"  NN - Acc: {nn_metrics['accuracy']:.4f}, F1: {nn_metrics['f1']:.4f}, MCC: {nn_metrics['mcc']:.4f}")

        if file_metrics:
            all_metrics[csv_path] = file_metrics

    # Cleanup
    extractor.restore_model()

    total_time = time.time() - total_start

    # Save aggregate metrics
    if all_metrics:
        metrics_path = output_dir / 'batch_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved batch metrics to {metrics_path}")

    print(f"\n{'='*60}")
    print(f"Batch inference complete!")
    print(f"  Files processed: {len(csv_files)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
