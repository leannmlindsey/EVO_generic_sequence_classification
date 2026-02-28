#!/usr/bin/env python3
"""
Rebuild embedding_analysis_results.json from existing output files.

Use this when the main embedding_analysis.py run completed successfully
(CSV, npz, pkl files exist) but the final JSON save failed (e.g., due to NaN).

Usage:
    python scripts/rebuild_results_json.py --output_dir /path/to/results

    With random baseline:
    python scripts/rebuild_results_json.py --output_dir /path/to/results --include_random_baseline
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

try:
    from sklearn.metrics import silhouette_score
except ImportError:
    silhouette_score = None


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0

    return metrics


def has_nan(arr):
    """Check if array contains NaN or Inf."""
    return np.isnan(arr).any() or np.isinf(arr).any()


def calc_silhouette(embeddings, labels):
    """Calculate silhouette score."""
    if silhouette_score is None:
        return None
    if len(np.unique(labels)) < 2:
        return None
    if has_nan(embeddings):
        print("  WARNING: Embeddings contain NaN/Inf, skipping silhouette")
        return None
    return silhouette_score(embeddings, labels)


def sanitize_for_json(obj):
    """Convert numpy types and NaN/Inf to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild embedding_analysis_results.json from existing output files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory containing the embedding analysis output files',
    )
    parser.add_argument(
        '--include_random_baseline', action='store_true',
        help='Include random baseline metrics (requires embeddings_random.npz)',
    )
    parser.add_argument(
        '--model_name', type=str, default='evo-1-8k-base',
        help='Model name (for metadata only)',
    )
    parser.add_argument(
        '--pooling', type=str, default='mean',
        help='Pooling strategy used (for metadata only)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Seed used (for metadata only)',
    )
    parser.add_argument(
        '--csv_dir', type=str, default=None,
        help='Original CSV directory (to get sequence counts for metadata)',
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if not output_dir.is_dir():
        print(f"ERROR: Output directory not found: {args.output_dir}")
        sys.exit(1)

    results = {
        'model_name': args.model_name,
        'pooling': args.pooling,
        'seed': args.seed,
    }

    # Get sequence counts from CSV dir if provided
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        for split in ['train', 'val', 'dev', 'test']:
            csv_path = csv_dir / f'{split}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                key = 'num_val' if split in ('val', 'dev') else f'num_{split}'
                results[key] = len(df)

    # Load pretrained embeddings for silhouette scores
    pretrained_emb_path = output_dir / 'embeddings_pretrained.npz'
    if pretrained_emb_path.exists():
        print(f"Loading pretrained embeddings: {pretrained_emb_path}")
        cached = np.load(pretrained_emb_path)
        train_embeddings = cached['train_embeddings']
        test_embeddings = cached['test_embeddings']

        # We need labels - try to get them from the predictions CSV
        pred_path = output_dir / 'test_predictions_pretrained.csv'
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            test_labels = pred_df['true_label'].astype(int).values
        else:
            test_labels = None

        # Try to get train labels from csv_dir
        train_labels = None
        if args.csv_dir:
            train_csv = Path(args.csv_dir) / 'train.csv'
            if train_csv.exists():
                train_df = pd.read_csv(train_csv)
                if 'label' in train_df.columns:
                    train_labels = train_df['label'].astype(int).values

        # Silhouette scores
        train_sil = calc_silhouette(train_embeddings, train_labels) if train_labels is not None else None
        test_sil = calc_silhouette(test_embeddings, test_labels) if test_labels is not None else None

        results['pretrained'] = {}
        if train_sil is not None:
            results['pretrained']['train_silhouette'] = train_sil
            print(f"  Train silhouette: {train_sil:.4f}")
        if test_sil is not None:
            results['pretrained']['test_silhouette'] = test_sil
            print(f"  Test silhouette: {test_sil:.4f}")
    else:
        print(f"WARNING: No pretrained embeddings found at {pretrained_emb_path}")
        results['pretrained'] = {}
        test_labels = None

    # Reconstruct LP metrics from predictions CSV + test labels
    # LP doesn't save its own predictions CSV, so recompute from the classifier if available
    # For now, we check if we can load the classifier and recompute
    lp_clf_path = output_dir / 'linear_probe.pkl'
    lp_scaler_path = output_dir / 'linear_probe_scaler.pkl'
    if lp_clf_path.exists() and lp_scaler_path.exists() and pretrained_emb_path.exists() and test_labels is not None:
        import pickle
        print("\nRecomputing linear probe metrics...")
        with open(lp_clf_path, 'rb') as f:
            clf = pickle.load(f)
        with open(lp_scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        test_scaled = scaler.transform(test_embeddings)
        lp_preds = clf.predict(test_scaled)
        lp_probs = clf.predict_proba(test_scaled)[:, 1]
        lp_metrics = calculate_metrics(test_labels, lp_preds, lp_probs)
        results['pretrained']['linear_probe'] = lp_metrics

        print("  Linear Probe Results:")
        for k, v in lp_metrics.items():
            print(f"    {k}: {v:.4f}")
    else:
        print("WARNING: Cannot recompute LP metrics (missing classifier, scaler, embeddings, or labels)")

    # Reconstruct NN metrics from test_predictions_pretrained.csv
    pred_path = output_dir / 'test_predictions_pretrained.csv'
    if pred_path.exists():
        print("\nRecomputing NN metrics from test_predictions_pretrained.csv...")
        pred_df = pd.read_csv(pred_path)
        y_true = pred_df['true_label'].astype(int).values
        y_pred = pred_df['predicted_label'].astype(int).values
        y_prob = pred_df['probability'].values
        nn_metrics = calculate_metrics(y_true, y_pred, y_prob)
        results['pretrained']['neural_network'] = nn_metrics

        print("  Neural Network Results:")
        for k, v in nn_metrics.items():
            print(f"    {k}: {v:.4f}")
    else:
        print(f"WARNING: No predictions file found at {pred_path}")

    # Random baseline
    if args.include_random_baseline:
        random_emb_path = output_dir / 'embeddings_random.npz'
        if not random_emb_path.exists():
            print(f"\nWARNING: Random embeddings not found at {random_emb_path}")
        else:
            print(f"\nLoading random embeddings: {random_emb_path}")
            cached_random = np.load(random_emb_path)
            random_train = cached_random['train_embeddings']
            random_test = cached_random['test_embeddings']

            # Check for NaN in random embeddings
            random_has_nan = has_nan(random_train) or has_nan(random_test)
            if random_has_nan:
                print("  WARNING: Random embeddings contain NaN/Inf!")
                print("  This likely means they were generated before the Savanna-style init fix.")
                print("  Delete embeddings_random.npz and re-extract to get valid random baseline.")
                print("  Skipping random baseline metrics.")
                results['random'] = {'error': 'embeddings contain NaN - re-extract needed'}
            else:
                # Random silhouette
                results['random'] = {}
                if train_labels is not None:
                    random_train_sil = calc_silhouette(random_train, train_labels)
                    if random_train_sil is not None:
                        results['random']['train_silhouette'] = random_train_sil
                        print(f"  Random train silhouette: {random_train_sil:.4f}")
                if test_labels is not None:
                    random_test_sil = calc_silhouette(random_test, test_labels)
                    if random_test_sil is not None:
                        results['random']['test_silhouette'] = random_test_sil
                        print(f"  Random test silhouette: {random_test_sil:.4f}")

                # Random LP - recompute by training on random embeddings
                if test_labels is not None and train_labels is not None:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler

                    print("\n  Training random LP...")
                    rscaler = StandardScaler()
                    rtrain_scaled = rscaler.fit_transform(random_train)
                    rtest_scaled = rscaler.transform(random_test)
                    rclf = LogisticRegression(max_iter=1000, random_state=42)
                    rclf.fit(rtrain_scaled, train_labels)
                    rlp_preds = rclf.predict(rtest_scaled)
                    rlp_probs = rclf.predict_proba(rtest_scaled)[:, 1]
                    rlp_metrics = calculate_metrics(test_labels, rlp_preds, rlp_probs)
                    results['random']['linear_probe'] = rlp_metrics

                    print("  Random Linear Probe Results:")
                    for k, v in rlp_metrics.items():
                        print(f"    {k}: {v:.4f}")

                # Random NN - need to retrain, which is expensive
                # Check if there's a random predictions CSV
                random_pred_path = output_dir / 'test_predictions_random.csv'
                if random_pred_path.exists():
                    print("\n  Recomputing random NN metrics from test_predictions_random.csv...")
                    rpred_df = pd.read_csv(random_pred_path)
                    ry_true = rpred_df['true_label'].astype(int).values
                    ry_pred = rpred_df['predicted_label'].astype(int).values
                    ry_prob = rpred_df['probability'].values
                    rnn_metrics = calculate_metrics(ry_true, ry_pred, ry_prob)
                    results['random']['neural_network'] = rnn_metrics

                    print("  Random Neural Network Results:")
                    for k, v in rnn_metrics.items():
                        print(f"    {k}: {v:.4f}")
                elif test_labels is not None and train_labels is not None:
                    # Retrain random NN (fast, no GPU needed for small data)
                    print("\n  Retraining random NN on random embeddings...")
                    random_val = cached_random['val_embeddings']
                    # Get val labels
                    val_labels = None
                    if args.csv_dir:
                        for vname in ['val.csv', 'dev.csv']:
                            vpath = Path(args.csv_dir) / vname
                            if vpath.exists():
                                vdf = pd.read_csv(vpath)
                                if 'label' in vdf.columns:
                                    val_labels = vdf['label'].astype(int).values
                                break

                    if val_labels is not None:
                        from evo.embedding_analysis import train_three_layer_nn
                        rnn_results = train_three_layer_nn(
                            random_train, train_labels,
                            random_val, val_labels,
                            random_test, test_labels,
                            device='cpu',
                        )
                        rnn_metrics = {k: v for k, v in rnn_results.items()
                                       if k not in ['predictions', 'probabilities', 'model', 'scaler']}
                        results['random']['neural_network'] = rnn_metrics

                        print("  Random Neural Network Results:")
                        for k, v in rnn_metrics.items():
                            print(f"    {k}: {v:.4f}")

                # Embedding power
                embedding_power = {}
                if 'test_silhouette' in results.get('pretrained', {}) and 'test_silhouette' in results.get('random', {}):
                    sil_power = results['pretrained']['test_silhouette'] - results['random']['test_silhouette']
                    embedding_power['silhouette_score'] = sil_power

                for classifier_key, prefix in [('linear_probe', 'linear_probe'), ('neural_network', 'nn')]:
                    pretrained_metrics = results.get('pretrained', {}).get(classifier_key, {})
                    random_metrics = results.get('random', {}).get(classifier_key, {})
                    for metric in ['accuracy', 'f1', 'mcc', 'auc']:
                        if metric in pretrained_metrics and metric in random_metrics:
                            power = pretrained_metrics[metric] - random_metrics[metric]
                            embedding_power[f'{prefix}_{metric}'] = power

                if embedding_power:
                    results['embedding_power'] = embedding_power
                    print("\n  Embedding Power (pretrained - random):")
                    for k, v in embedding_power.items():
                        print(f"    {k}: {v:+.4f}")

    # Save JSON
    results_path = output_dir / 'embedding_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(sanitize_for_json(results), f, indent=2)
    print(f"\nSaved results to {results_path}")
    print("Done.")


if __name__ == '__main__':
    main()
