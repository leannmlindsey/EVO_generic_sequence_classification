"""
Embedding Analysis Module for Evo

Extracts embeddings from DNA sequences using the Evo model and performs
downstream classification analysis (linear probe, neural network).

Based on approaches from:
- GitHub Issue #32: https://github.com/evo-design/evo/issues/32
- GitHub Issue #93: https://github.com/evo-design/evo/issues/93
"""

import argparse
import json
import os
import pkgutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA

try:
    from sklearn.metrics import silhouette_score
except ImportError:
    silhouette_score = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evo import Evo
from evo.scoring import prepare_batch
from stripedhyena.model import StripedHyena
from stripedhyena.tokenizer import CharLevelTokenizer
from stripedhyena.utils import dotdict


def create_random_evo_model(model_name: str, device: str, seed: int = 42):
    """
    Create a randomly initialized Evo model (same architecture, no pretrained weights).

    This is useful for baseline comparison to measure the value of pretrained weights.

    Parameters
    ----------
    model_name : str
        Name of the Evo model (used to determine config)
    device : str
        Device to load model on
    seed : int
        Random seed for reproducible initialization

    Returns
    -------
    model : StripedHyena
        Randomly initialized model
    tokenizer : CharLevelTokenizer
        Tokenizer (same as pretrained)
    """
    print("\n" + "=" * 60)
    print("Creating Randomly Initialized Baseline Model")
    print("=" * 60)

    # Set seed for reproducible random initialization
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine config path based on model name (same logic as in evo/models.py)
    if model_name in ['evo-1-8k-base', 'evo-1-8k-crispr', 'evo-1-8k-transposon', 'evo-1.5-8k-base']:
        config_path = 'configs/evo-1-8k-base_inference.yml'
    elif model_name == 'evo-1-131k-base':
        config_path = 'configs/evo-1-131k-base_inference.yml'
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load config from evo package
    print(f"Loading config from: {config_path}")
    import evo
    config = yaml.safe_load(pkgutil.get_data('evo', config_path))
    global_config = dotdict(config, Loader=yaml.FullLoader)

    # Create model with random weights (no pretrained weights loaded)
    print("Initializing StripedHyena with random weights...")
    model = StripedHyena(global_config)
    model.to_bfloat16_except_poles_residues()

    if device is not None:
        model = model.to(device)
    model.eval()

    # Use same tokenizer as pretrained
    tokenizer = CharLevelTokenizer(512)

    print("Random model initialized successfully")

    return model, tokenizer


class IdentityUnembed(nn.Module):
    """
    Identity function to replace model.unembed for extracting embeddings.

    When model.unembed is replaced with this class, the model returns
    hidden states instead of logits. This approach is documented in
    GitHub issues #32 and #93.
    """
    def unembed(self, u):
        return u


class EvoEmbeddingExtractor:
    """
    Wrapper for Evo model that extracts embeddings from DNA sequences.

    Supports two modes:
    1. Final layer embeddings via unembed monkey-patching (default, recommended)
    2. Intermediate layer embeddings via forward hooks (optional)

    Parameters
    ----------
    model_name : str
        Name of the Evo model to load (e.g., 'evo-1-8k-base', 'evo-1-131k-base')
    device : str
        Device to run inference on (e.g., 'cuda:0', 'cpu')
    layer_idx : int, optional
        Layer index for intermediate embedding extraction via hooks.
        If None (default), uses the final layer via unembed patching.
        Use negative indexing (e.g., -1 for last layer, -2 for second-to-last).
    pooling : str
        Pooling strategy: 'mean', 'first', 'last'
    """

    def __init__(
        self,
        model_name: str = 'evo-1-8k-base',
        device: str = 'cuda:0',
        layer_idx: Optional[int] = None,
        pooling: str = 'mean',
    ):
        self.model_name = model_name
        self.device = device
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.hidden_states = None
        self._hook_handle = None

        # Load model
        print(f"Loading Evo model: {model_name}")
        evo_model = Evo(model_name, device=device)
        self.model = evo_model.model
        self.tokenizer = evo_model.tokenizer
        self.model.eval()

        # Get hidden dimension from model config or default
        self.hidden_dim = 4096  # Evo models use 4096

        # Setup embedding extraction method
        if layer_idx is None:
            # Use unembed patching for final layer (recommended approach)
            self._original_unembed = self.model.unembed
            self.model.unembed = IdentityUnembed()
            self._use_hooks = False
            print("Using unembed patching for final layer embeddings")
        else:
            # Use forward hooks for intermediate layers
            self._setup_hook(layer_idx)
            self._use_hooks = True
            print(f"Using forward hooks for layer {layer_idx} embeddings")

    def _setup_hook(self, layer_idx: int):
        """Register forward hook on specified layer."""
        # StripedHyena uses 'blocks' attribute for layers
        if hasattr(self.model, 'blocks'):
            layers = self.model.blocks
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise AttributeError(
                "Could not find 'blocks' or 'layers' attribute on model. "
                "Model structure may have changed."
            )

        num_layers = len(layers)

        # Handle negative indexing
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx

        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range. Model has {num_layers} layers."
            )

        target_layer = layers[layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        print(f"Registered hook on layer {layer_idx} (of {num_layers} total)")

    def _hook_fn(self, module, input, output):
        """Capture hidden states from forward hook."""
        if isinstance(output, tuple):
            self.hidden_states = output[0].detach()
        else:
            self.hidden_states = output.detach()

    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        seq_lengths: List[int],
        prepend_bos: bool = True,
    ) -> torch.Tensor:
        """
        Pool hidden states to get fixed-size embeddings.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape (batch_size, seq_len, hidden_dim)
        seq_lengths : List[int]
            Original sequence lengths (before padding, excluding BOS)
        prepend_bos : bool
            Whether BOS token was prepended during tokenization

        Returns
        -------
        torch.Tensor
            Shape (batch_size, hidden_dim)
        """
        batch_size = hidden_states.shape[0]
        pooled = []

        for i in range(batch_size):
            seq_len = seq_lengths[i]
            # Account for BOS token if prepended
            start_idx = 1 if prepend_bos else 0
            end_idx = start_idx + seq_len

            # Get embeddings for actual sequence (excluding padding and BOS)
            seq_hidden = hidden_states[i, start_idx:end_idx, :]

            if self.pooling == 'mean':
                pooled.append(seq_hidden.mean(dim=0))
            elif self.pooling == 'first':
                pooled.append(seq_hidden[0, :])
            elif self.pooling == 'last':
                pooled.append(seq_hidden[-1, :])
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return torch.stack(pooled, dim=0)

    def extract_batch(
        self,
        sequences: List[str],
        prepend_bos: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of sequences.

        Parameters
        ----------
        sequences : List[str]
            DNA sequences
        prepend_bos : bool
            Whether to prepend BOS token

        Returns
        -------
        np.ndarray
            Shape (batch_size, hidden_dim)
        """
        # Prepare batch
        input_ids, seq_lengths = prepare_batch(
            sequences,
            self.tokenizer,
            prepend_bos=prepend_bos,
            device=self.device,
        )

        # Forward pass
        with torch.inference_mode():
            output, _ = self.model(input_ids)

        if self._use_hooks:
            # Get hidden states captured by hook
            hidden_states = self.hidden_states
        else:
            # Output is already hidden states (due to unembed patching)
            hidden_states = output

        # Pool embeddings
        embeddings = self._pool_embeddings(hidden_states, seq_lengths, prepend_bos)

        return embeddings.float().cpu().numpy()

    def restore_model(self):
        """Restore original model state (remove hooks/patches)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        if hasattr(self, '_original_unembed'):
            self.model.unembed = self._original_unembed

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.restore_model()


def extract_embeddings(
    extractor,  # EvoEmbeddingExtractor or RandomEvoEmbeddingExtractor
    sequences: List[str],
    labels: List[int],
    batch_size: int = 8,
    max_length: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings for a list of sequences.

    Parameters
    ----------
    extractor : EvoEmbeddingExtractor or RandomEvoEmbeddingExtractor
        Initialized embedding extractor
    sequences : List[str]
        DNA sequences
    labels : List[int]
        Class labels
    batch_size : int
        Batch size for processing
    max_length : int, optional
        Maximum sequence length (truncate longer sequences)
    show_progress : bool
        Whether to show progress

    Returns
    -------
    embeddings : np.ndarray
        Shape (num_sequences, hidden_dim)
    labels : np.ndarray
        Shape (num_sequences,)
    """
    all_embeddings = []

    # Optionally truncate sequences
    if max_length is not None:
        sequences = [seq[:max_length] for seq in sequences]

    num_batches = (len(sequences) + batch_size - 1) // batch_size

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]

        if show_progress:
            batch_num = i // batch_size + 1
            print(f"Processing batch {batch_num}/{num_batches}", end='\r')

        embeddings = extractor.extract_batch(batch_seqs)
        all_embeddings.append(embeddings)

    if show_progress:
        print()  # Newline after progress

    return np.vstack(all_embeddings), np.array(labels)


class RandomEvoEmbeddingExtractor:
    """
    Wrapper for randomly initialized Evo model that extracts embeddings.

    This is identical to EvoEmbeddingExtractor but uses a randomly initialized
    model instead of pretrained weights, for baseline comparison.
    """

    def __init__(
        self,
        model_name: str = 'evo-1-8k-base',
        device: str = 'cuda:0',
        pooling: str = 'mean',
        seed: int = 42,
    ):
        self.model_name = model_name
        self.device = device
        self.pooling = pooling

        # Create randomly initialized model
        self.model, self.tokenizer = create_random_evo_model(model_name, device, seed)
        self.hidden_dim = 4096

        # Use unembed patching for final layer (same as pretrained)
        self._original_unembed = self.model.unembed
        self.model.unembed = IdentityUnembed()
        print("Using unembed patching for final layer embeddings (random model)")

    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        seq_lengths: List[int],
        prepend_bos: bool = True,
    ) -> torch.Tensor:
        """Pool hidden states to get fixed-size embeddings."""
        batch_size = hidden_states.shape[0]
        pooled = []

        for i in range(batch_size):
            seq_len = seq_lengths[i]
            start_idx = 1 if prepend_bos else 0
            end_idx = start_idx + seq_len
            seq_hidden = hidden_states[i, start_idx:end_idx, :]

            if self.pooling == 'mean':
                pooled.append(seq_hidden.mean(dim=0))
            elif self.pooling == 'first':
                pooled.append(seq_hidden[0, :])
            elif self.pooling == 'last':
                pooled.append(seq_hidden[-1, :])
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return torch.stack(pooled, dim=0)

    def extract_batch(
        self,
        sequences: List[str],
        prepend_bos: bool = True,
    ) -> np.ndarray:
        """Extract embeddings for a batch of sequences."""
        input_ids, seq_lengths = prepare_batch(
            sequences,
            self.tokenizer,
            prepend_bos=prepend_bos,
            device=self.device,
        )

        with torch.inference_mode():
            output, _ = self.model(input_ids)

        hidden_states = output
        embeddings = self._pool_embeddings(hidden_states, seq_lengths, prepend_bos)

        return embeddings.float().cpu().numpy()

    def restore_model(self):
        """Restore original model state."""
        if hasattr(self, '_original_unembed'):
            self.model.unembed = self._original_unembed

    def __del__(self):
        """Cleanup on deletion."""
        self.restore_model()


def train_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """
    Train a logistic regression classifier on embeddings.

    Parameters
    ----------
    train_embeddings : np.ndarray
        Training embeddings
    train_labels : np.ndarray
        Training labels
    test_embeddings : np.ndarray
        Test embeddings
    test_labels : np.ndarray
        Test labels
    max_iter : int
        Maximum iterations for logistic regression

    Returns
    -------
    dict
        Dictionary containing metrics and predictions
    """
    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Train logistic regression
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(train_scaled, train_labels)

    # Predict
    test_preds = clf.predict(test_scaled)
    test_probs = clf.predict_proba(test_scaled)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(test_labels, test_preds, test_probs)
    metrics['predictions'] = test_preds
    metrics['probabilities'] = test_probs
    metrics['scaler'] = scaler
    metrics['classifier'] = clf

    return metrics


class ThreeLayerNN(nn.Module):
    """
    Three-layer neural network for classification.

    Architecture: input -> hidden (ReLU, dropout) -> hidden/2 (ReLU, dropout) -> 2
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_three_layer_nn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    device: str = 'cuda:0',
) -> Dict[str, Any]:
    """
    Train a 3-layer neural network classifier with early stopping.

    Parameters
    ----------
    train_embeddings : np.ndarray
        Training embeddings
    train_labels : np.ndarray
        Training labels
    val_embeddings : np.ndarray
        Validation embeddings
    val_labels : np.ndarray
        Validation labels
    test_embeddings : np.ndarray
        Test embeddings
    test_labels : np.ndarray
        Test labels
    hidden_dim : int
        Hidden layer dimension
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Learning rate
    patience : int
        Early stopping patience
    device : str
        Device for training

    Returns
    -------
    dict
        Dictionary containing metrics, predictions, and model
    """
    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(train_scaled, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_X = torch.tensor(val_scaled, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_labels, dtype=torch.long).to(device)

    test_X = torch.tensor(test_scaled, dtype=torch.float32).to(device)

    # Initialize model
    input_dim = train_embeddings.shape[1]
    model = ThreeLayerNN(input_dim, hidden_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test predictions
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X)
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()

    # Calculate metrics
    metrics = calculate_metrics(test_labels, test_preds, test_probs)
    metrics['predictions'] = test_preds
    metrics['probabilities'] = test_probs
    metrics['model'] = model
    metrics['scaler'] = scaler

    return metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Predicted probabilities for positive class

    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    # Calculate sensitivity and specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC requires probabilities
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0

    return metrics


def calculate_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calculate silhouette score for cluster quality.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings
    labels : np.ndarray
        Class labels

    Returns
    -------
    float
        Silhouette score (-1 to 1, higher is better)
    """
    if silhouette_score is None:
        print("Warning: silhouette_score not available")
        return 0.0

    if len(np.unique(labels)) < 2:
        print("Warning: Need at least 2 classes for silhouette score")
        return 0.0

    return silhouette_score(embeddings, labels)


def create_pca_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "PCA Visualization",
) -> None:
    """
    Create a 2D PCA visualization of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings
    labels : np.ndarray
        Class labels
    output_path : str
        Path to save the plot
    title : str
        Plot title
    """
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.6,
            s=50,
        )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved PCA visualization to {output_path}")


def load_csv_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Load sequences and labels from a CSV file.

    Expected columns: 'sequence' (DNA), 'label' (0 or 1)

    Parameters
    ----------
    csv_path : str
        Path to CSV file

    Returns
    -------
    sequences : List[str]
        DNA sequences
    labels : List[int]
        Class labels
    """
    df = pd.read_csv(csv_path)

    # Check for required columns
    if 'sequence' not in df.columns:
        raise ValueError(f"CSV must have 'sequence' column. Found: {df.columns.tolist()}")
    if 'label' not in df.columns:
        raise ValueError(f"CSV must have 'label' column. Found: {df.columns.tolist()}")

    sequences = df['sequence'].tolist()
    labels = df['label'].astype(int).tolist()

    return sequences, labels


def main():
    parser = argparse.ArgumentParser(
        description='Embedding analysis for Evo model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        '--csv_dir',
        type=str,
        required=True,
        help='Directory containing train.csv, dev.csv/val.csv, test.csv',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./embedding_analysis_results',
        help='Output directory for results',
    )

    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='evo-1-8k-base',
        choices=['evo-1.5-8k-base', 'evo-1-8k-base', 'evo-1-131k-base',
                 'evo-1-8k-crispr', 'evo-1-8k-transposon'],
        help='Evo model name',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for inference',
    )
    parser.add_argument(
        '--layer_idx',
        type=int,
        default=None,
        help='Layer index for intermediate embeddings (None = final layer)',
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default='mean',
        choices=['mean', 'first', 'last'],
        help='Pooling strategy',
    )

    # Processing arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for embedding extraction',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=8192,
        help='Maximum sequence length',
    )

    # NN training arguments
    parser.add_argument(
        '--nn_epochs',
        type=int,
        default=100,
        help='Maximum epochs for neural network training',
    )
    parser.add_argument(
        '--nn_hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension for neural network',
    )
    parser.add_argument(
        '--nn_lr',
        type=float,
        default=1e-3,
        help='Learning rate for neural network',
    )

    # Options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--include_random_baseline',
        action='store_true',
        help='Include random embedding baseline comparison (uses randomly initialized model)',
    )
    parser.add_argument(
        '--skip_nn',
        action='store_true',
        help='Skip neural network training (only run linear probe)',
    )
    parser.add_argument(
        '--cache_embeddings',
        action='store_true',
        help='Cache embeddings to disk',
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"\nRandom seed: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find data files
    csv_dir = Path(args.csv_dir)
    train_csv = csv_dir / 'train.csv'
    test_csv = csv_dir / 'test.csv'

    # Try dev.csv or val.csv for validation
    val_csv = csv_dir / 'dev.csv'
    if not val_csv.exists():
        val_csv = csv_dir / 'val.csv'

    # Validate files exist
    for csv_file in [train_csv, val_csv, test_csv]:
        if not csv_file.exists():
            raise FileNotFoundError(f"Required CSV file not found: {csv_file}")

    print(f"Loading data from {csv_dir}")
    train_seqs, train_labels = load_csv_data(train_csv)
    val_seqs, val_labels = load_csv_data(val_csv)
    test_seqs, test_labels = load_csv_data(test_csv)

    print(f"  Train: {len(train_seqs)} sequences")
    print(f"  Val: {len(val_seqs)} sequences")
    print(f"  Test: {len(test_seqs)} sequences")

    # Initialize results
    results = {
        'model_name': args.model_name,
        'pooling': args.pooling,
        'layer_idx': args.layer_idx,
        'seed': args.seed,
        'num_train': len(train_seqs),
        'num_val': len(val_seqs),
        'num_test': len(test_seqs),
    }

    # Check for cached embeddings
    cache_path = output_dir / 'embeddings_pretrained.npz'

    if args.cache_embeddings and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        cached = np.load(cache_path)
        train_embeddings = cached['train_embeddings']
        val_embeddings = cached['val_embeddings']
        test_embeddings = cached['test_embeddings']
    else:
        # Initialize embedding extractor
        extractor = EvoEmbeddingExtractor(
            model_name=args.model_name,
            device=args.device,
            layer_idx=args.layer_idx,
            pooling=args.pooling,
        )

        # Extract embeddings
        print("\nExtracting training embeddings...")
        train_embeddings, _ = extract_embeddings(
            extractor, train_seqs, train_labels,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        print("\nExtracting validation embeddings...")
        val_embeddings, _ = extract_embeddings(
            extractor, val_seqs, val_labels,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        print("\nExtracting test embeddings...")
        test_embeddings, _ = extract_embeddings(
            extractor, test_seqs, test_labels,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # Cleanup
        extractor.restore_model()

        # Cache embeddings
        if args.cache_embeddings:
            print(f"Caching embeddings to {cache_path}")
            np.savez(
                cache_path,
                train_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                test_embeddings=test_embeddings,
            )

    print(f"\nEmbedding shape: {train_embeddings.shape}")

    # Calculate silhouette scores
    print("\nCalculating silhouette scores...")
    train_silhouette = calculate_silhouette(train_embeddings, np.array(train_labels))
    test_silhouette = calculate_silhouette(test_embeddings, np.array(test_labels))
    print(f"  Train silhouette: {train_silhouette:.4f}")
    print(f"  Test silhouette: {test_silhouette:.4f}")

    results['pretrained'] = {
        'train_silhouette': train_silhouette,
        'test_silhouette': test_silhouette,
    }

    # Train linear probe
    print("\nTraining linear probe...")
    linear_results = train_linear_probe(
        train_embeddings, np.array(train_labels),
        test_embeddings, np.array(test_labels),
    )

    print("\nLinear Probe Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'sensitivity', 'specificity']:
        if metric in linear_results:
            print(f"  {metric}: {linear_results[metric]:.4f}")

    results['pretrained']['linear_probe'] = {
        k: v for k, v in linear_results.items()
        if k not in ['predictions', 'probabilities', 'scaler', 'classifier']
    }

    # Train neural network
    if not args.skip_nn:
        print("\nTraining 3-layer neural network...")
        nn_results = train_three_layer_nn(
            train_embeddings, np.array(train_labels),
            val_embeddings, np.array(val_labels),
            test_embeddings, np.array(test_labels),
            hidden_dim=args.nn_hidden_dim,
            epochs=args.nn_epochs,
            learning_rate=args.nn_lr,
            device=args.device,
        )

        print("\nNeural Network Results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'sensitivity', 'specificity']:
            if metric in nn_results:
                print(f"  {metric}: {nn_results[metric]:.4f}")

        results['pretrained']['neural_network'] = {
            k: v for k, v in nn_results.items()
            if k not in ['predictions', 'probabilities', 'model', 'scaler']
        }

        # Save NN model
        nn_model_path = output_dir / 'three_layer_nn_pretrained.pt'
        torch.save(nn_results['model'].state_dict(), nn_model_path)
        print(f"Saved neural network to {nn_model_path}")

        # Save predictions
        pred_df = pd.DataFrame({
            'true_label': test_labels,
            'predicted_label': nn_results['predictions'],
            'probability': nn_results['probabilities'],
        })
        pred_path = output_dir / 'test_predictions_pretrained.csv'
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

    # Create PCA visualization
    print("\nCreating PCA visualization...")
    pca_path = output_dir / 'pca_visualization_pretrained.png'
    create_pca_visualization(
        test_embeddings,
        np.array(test_labels),
        str(pca_path),
        title=f"Evo Embeddings PCA ({args.model_name})",
    )

    # Random baseline comparison (using randomly initialized model)
    if args.include_random_baseline:
        print("\n" + "#" * 60)
        print("# RANDOM BASELINE MODEL ANALYSIS")
        print("#" * 60)

        # Check for cached random embeddings
        cache_path_random = output_dir / 'embeddings_random.npz'

        if args.cache_embeddings and cache_path_random.exists():
            print(f"Loading cached random embeddings from {cache_path_random}")
            cached_random = np.load(cache_path_random)
            random_train = cached_random['train_embeddings']
            random_val = cached_random['val_embeddings']
            random_test = cached_random['test_embeddings']
        else:
            # Initialize random embedding extractor (uses seed + 1000 for different random init)
            random_extractor = RandomEvoEmbeddingExtractor(
                model_name=args.model_name,
                device=args.device,
                pooling=args.pooling,
                seed=args.seed + 1000,
            )

            # Extract embeddings from random model
            print("\nExtracting training embeddings (random model)...")
            random_train, _ = extract_embeddings(
                random_extractor, train_seqs, train_labels,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            print("\nExtracting validation embeddings (random model)...")
            random_val, _ = extract_embeddings(
                random_extractor, val_seqs, val_labels,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            print("\nExtracting test embeddings (random model)...")
            random_test, _ = extract_embeddings(
                random_extractor, test_seqs, test_labels,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            # Cleanup
            random_extractor.restore_model()
            del random_extractor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Cache random embeddings
            if args.cache_embeddings:
                print(f"Caching random embeddings to {cache_path_random}")
                np.savez(
                    cache_path_random,
                    train_embeddings=random_train,
                    val_embeddings=random_val,
                    test_embeddings=random_test,
                )

        print(f"\nRandom embedding shape: {random_train.shape}")

        # Random silhouette
        random_train_silhouette = calculate_silhouette(random_train, np.array(train_labels))
        random_test_silhouette = calculate_silhouette(random_test, np.array(test_labels))
        print(f"\nRandom silhouette scores:")
        print(f"  Train: {random_train_silhouette:.4f}")
        print(f"  Test: {random_test_silhouette:.4f}")

        results['random'] = {
            'train_silhouette': random_train_silhouette,
            'test_silhouette': random_test_silhouette,
        }

        # Random linear probe
        print("\nTraining linear probe on random embeddings...")
        random_linear_results = train_linear_probe(
            random_train, np.array(train_labels),
            random_test, np.array(test_labels),
        )

        print("\nRandom Linear Probe Results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']:
            if metric in random_linear_results:
                print(f"  {metric}: {random_linear_results[metric]:.4f}")

        results['random']['linear_probe'] = {
            k: v for k, v in random_linear_results.items()
            if k not in ['predictions', 'probabilities', 'scaler', 'classifier']
        }

        # Random NN
        if not args.skip_nn:
            print("\nTraining neural network on random embeddings...")
            random_nn_results = train_three_layer_nn(
                random_train, np.array(train_labels),
                random_val, np.array(val_labels),
                random_test, np.array(test_labels),
                hidden_dim=args.nn_hidden_dim,
                epochs=args.nn_epochs,
                learning_rate=args.nn_lr,
                device=args.device,
            )

            print("\nRandom Neural Network Results:")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']:
                if metric in random_nn_results:
                    print(f"  {metric}: {random_nn_results[metric]:.4f}")

            results['random']['neural_network'] = {
                k: v for k, v in random_nn_results.items()
                if k not in ['predictions', 'probabilities', 'model', 'scaler']
            }

        # Random PCA
        pca_random_path = output_dir / 'pca_visualization_random.png'
        create_pca_visualization(
            random_test,
            np.array(test_labels),
            str(pca_random_path),
            title="Random Model Embeddings PCA (Baseline)",
        )

        # Compute embedding power (pretrained - random)
        print("\n" + "=" * 60)
        print("EMBEDDING POWER (Pretrained - Random)")
        print("=" * 60)

        embedding_power = {}

        # Silhouette power
        silhouette_power = test_silhouette - random_test_silhouette
        embedding_power['silhouette_score'] = silhouette_power
        print(f"\nSilhouette Score: {test_silhouette:.4f} - {random_test_silhouette:.4f} = {silhouette_power:+.4f}")

        # Linear probe power
        print("\nLinear Probe:")
        for metric in ['accuracy', 'f1', 'mcc', 'auc']:
            if metric in linear_results and metric in random_linear_results:
                pretrained_val = linear_results[metric]
                random_val_metric = random_linear_results[metric]
                power = pretrained_val - random_val_metric
                embedding_power[f'linear_probe_{metric}'] = power
                print(f"  {metric}: {pretrained_val:.4f} - {random_val_metric:.4f} = {power:+.4f}")

        # NN power
        if not args.skip_nn:
            print("\n3-Layer NN:")
            for metric in ['accuracy', 'f1', 'mcc', 'auc']:
                if metric in nn_results and metric in random_nn_results:
                    pretrained_val = nn_results[metric]
                    random_val_metric = random_nn_results[metric]
                    power = pretrained_val - random_val_metric
                    embedding_power[f'nn_{metric}'] = power
                    print(f"  {metric}: {pretrained_val:.4f} - {random_val_metric:.4f} = {power:+.4f}")

        results['embedding_power'] = embedding_power

    # Save results
    results_path = output_dir / 'embedding_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
