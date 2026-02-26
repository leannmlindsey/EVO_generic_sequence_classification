#!/usr/bin/env python
"""
Standalone Embedding Extraction for Evo

Extracts embeddings from DNA sequences using a trained Evo model.
Uses EvoEmbeddingExtractor from evo.embedding_analysis.

Usage:
    python scripts/embedding_extraction.py \
        --input_csv data.csv \
        --output_dir ./embeddings \
        --model_name evo-1-8k-base \
        --pooling mean
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Extract embeddings from DNA sequences using Evo model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--input_csv', type=str, required=True,
        help='Path to CSV file with a "sequence" column (and optional "label" column)',
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save extracted embeddings',
    )
    parser.add_argument(
        '--model_name', type=str, default='evo-1-8k-base',
        help='Evo model name',
    )
    parser.add_argument(
        '--pooling', type=str, default='mean',
        choices=['mean', 'first', 'last'],
        help='Pooling strategy for sequence embeddings',
    )
    parser.add_argument(
        '--layer_idx', type=int, default=None,
        help='Layer index for intermediate embeddings (None = final layer)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for embedding extraction',
    )
    parser.add_argument(
        '--max_length', type=int, default=8192,
        help='Maximum sequence length (longer sequences are truncated)',
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for model inference',
    )
    parser.add_argument(
        '--output_name', type=str, default='embeddings',
        help='Base name for output files (produces <name>.npz and <name>_metadata.json)',
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"ERROR: Input CSV not found: {args.input_csv}")
        sys.exit(1)

    # Read input CSV
    print(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if 'sequence' not in df.columns:
        print(f"ERROR: CSV must have a 'sequence' column. Found: {df.columns.tolist()}")
        sys.exit(1)

    sequences = df['sequence'].tolist()
    has_labels = 'label' in df.columns
    labels = df['label'].astype(int).tolist() if has_labels else [0] * len(sequences)

    print(f"Loaded {len(sequences)} sequences")
    if has_labels:
        unique_labels = sorted(set(labels))
        print(f"Labels found: {unique_labels} (counts: {[labels.count(l) for l in unique_labels]})")
    else:
        print("No 'label' column found - using dummy labels")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import and initialize extractor (deferred to avoid slow import if args are invalid)
    from evo.embedding_analysis import EvoEmbeddingExtractor, extract_embeddings

    print(f"\nInitializing Evo model: {args.model_name}")
    extractor = EvoEmbeddingExtractor(
        model_name=args.model_name,
        device=args.device,
        layer_idx=args.layer_idx,
        pooling=args.pooling,
    )

    # Extract embeddings
    print(f"\nExtracting embeddings (batch_size={args.batch_size}, max_length={args.max_length})...")
    start_time = time.time()

    embeddings, labels_array = extract_embeddings(
        extractor,
        sequences,
        labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    elapsed = time.time() - start_time
    print(f"Extraction complete: {embeddings.shape} in {elapsed:.1f}s")

    # Save embeddings
    npz_path = output_dir / f'{args.output_name}.npz'
    save_dict = {'embeddings': embeddings}
    if has_labels:
        save_dict['labels'] = labels_array
    np.savez(npz_path, **save_dict)
    print(f"Saved embeddings to {npz_path}")

    # Save metadata
    metadata = {
        'input_csv': str(input_path.resolve()),
        'num_sequences': len(sequences),
        'embedding_dim': int(embeddings.shape[1]),
        'model_name': args.model_name,
        'pooling': args.pooling,
        'layer_idx': args.layer_idx,
        'max_length': args.max_length,
        'has_labels': has_labels,
        'extraction_time_seconds': round(elapsed, 2),
    }

    metadata_path = output_dir / f'{args.output_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Cleanup
    extractor.restore_model()
    print("\nDone.")


if __name__ == '__main__':
    main()
