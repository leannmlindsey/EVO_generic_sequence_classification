#!/usr/bin/env python
"""
Standalone Metrics Calculation

Calculates classification metrics from any CSV with 'label' and 'pred_label' columns.
Supports single file, multiple files, and directory scanning.

Usage:
    # Single file:
    python scripts/calculate_metrics.py --input predictions.csv

    # Multiple files:
    python scripts/calculate_metrics.py --input pred1.csv pred2.csv

    # Directory (scans for *predictions*.csv):
    python scripts/calculate_metrics.py --input_dir results/

    # Save metrics to JSON:
    python scripts/calculate_metrics.py --input predictions.csv --output_json metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    # Sensitivity and specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0

    return metrics


def process_file(csv_path, verbose=True):
    """Process a single predictions CSV and return metrics."""
    df = pd.read_csv(csv_path)

    # Check required columns
    if 'label' not in df.columns:
        # Also accept 'true_label' for backward compatibility
        if 'true_label' in df.columns:
            df['label'] = df['true_label']
        else:
            print(f"  SKIP: No 'label' or 'true_label' column in {csv_path}")
            return None

    if 'pred_label' not in df.columns:
        if 'predicted_label' in df.columns:
            df['pred_label'] = df['predicted_label']
        else:
            print(f"  SKIP: No 'pred_label' or 'predicted_label' column in {csv_path}")
            return None

    y_true = df['label'].astype(int).values
    y_pred = df['pred_label'].astype(int).values

    # Check for probability columns
    y_prob = None
    if 'prob_1' in df.columns:
        y_prob = df['prob_1'].values
    elif 'probability' in df.columns:
        y_prob = df['probability'].values

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics['num_samples'] = len(y_true)

    if verbose:
        print(f"\n  File: {csv_path}")
        print(f"  Samples: {metrics['num_samples']}")
        for key in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'sensitivity', 'specificity', 'fpr', 'fnr', 'auc']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print(f"  Confusion matrix: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Calculate classification metrics from prediction CSVs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--input', type=str, nargs='*', default=None,
        help='One or more prediction CSV files',
    )
    parser.add_argument(
        '--input_dir', type=str, default=None,
        help='Directory to scan for prediction CSVs (matches *predictions*.csv)',
    )
    parser.add_argument(
        '--output_json', type=str, default=None,
        help='Path to save metrics as JSON',
    )
    parser.add_argument(
        '--aggregate', action='store_true',
        help='Compute aggregate metrics across all files',
    )

    args = parser.parse_args()

    # Collect input files
    csv_files = []

    if args.input:
        for path_str in args.input:
            path = Path(path_str)
            if not path.exists():
                print(f"WARNING: File not found: {path_str}")
            else:
                csv_files.append(path)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"ERROR: Directory not found: {args.input_dir}")
            sys.exit(1)
        found = sorted(input_dir.glob('*predictions*.csv'))
        if not found:
            # Also try recursive search
            found = sorted(input_dir.rglob('*predictions*.csv'))
        csv_files.extend(found)
        print(f"Found {len(found)} prediction files in {args.input_dir}")

    if not csv_files:
        print("ERROR: No input files specified. Use --input or --input_dir.")
        parser.print_help()
        sys.exit(1)

    # Process files
    all_results = {}
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    print(f"Processing {len(csv_files)} file(s)...")

    for csv_path in csv_files:
        metrics = process_file(csv_path)
        if metrics is not None:
            all_results[str(csv_path)] = metrics

            # Collect for aggregate
            if args.aggregate:
                df = pd.read_csv(csv_path)
                label_col = 'label' if 'label' in df.columns else 'true_label'
                pred_col = 'pred_label' if 'pred_label' in df.columns else 'predicted_label'
                all_y_true.extend(df[label_col].astype(int).tolist())
                all_y_pred.extend(df[pred_col].astype(int).tolist())
                if 'prob_1' in df.columns:
                    all_y_prob.extend(df['prob_1'].tolist())
                elif 'probability' in df.columns:
                    all_y_prob.extend(df['probability'].tolist())

    # Aggregate metrics
    if args.aggregate and len(all_y_true) > 0:
        print("\n" + "=" * 50)
        print("AGGREGATE METRICS (all files combined)")
        print("=" * 50)
        y_prob = np.array(all_y_prob) if all_y_prob else None
        agg_metrics = compute_metrics(
            np.array(all_y_true), np.array(all_y_pred), y_prob,
        )
        agg_metrics['num_samples'] = len(all_y_true)
        agg_metrics['num_files'] = len(all_results)

        for key in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'sensitivity', 'specificity', 'fpr', 'fnr', 'auc']:
            if key in agg_metrics:
                print(f"  {key}: {agg_metrics[key]:.4f}")
        print(f"  Total samples: {agg_metrics['num_samples']}")
        print(f"  Files: {agg_metrics['num_files']}")

        all_results['aggregate'] = agg_metrics

    # Save to JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = {}
        for file_key, metrics in all_results.items():
            serializable[file_key] = {k: convert(v) for k, v in metrics.items()}

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved metrics to {output_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
