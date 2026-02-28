#!/usr/bin/env python3
"""
Quick smoke test: create random Evo model with Savanna-style init,
run one sequence, check for NaN. Takes ~30 seconds.
"""

import sys
import torch
import numpy as np

# Use the actual create_random_evo_model from the fixed code
from evo.embedding_analysis import create_random_evo_model, EvoEmbeddingExtractor
from evo.scoring import prepare_batch
from stripedhyena.tokenizer import CharLevelTokenizer

print("Creating random model with Savanna-style init...")
model, tokenizer = create_random_evo_model('evo-1-8k-base', device='cuda:0', seed=1042)

# Check poles
print("\nPole values after Savanna init:")
for name, param in model.named_parameters():
    if name.endswith('.poles'):
        has_pos = bool((param > 0).any())
        print(f"  {name}: min={param.min().item():.4f}, max={param.max().item():.4f}, has_positive={has_pos}")
        if has_pos:
            print("  WARNING: Positive poles found! This will cause NaN.")
        break

# Smoke test: run one sequence
print("\nRunning smoke test...")
test_seq = "ATCGATCGATCGATCGATCGATCGATCGATCG"
input_ids, seq_lengths = prepare_batch([test_seq], tokenizer, prepend_bos=True, device='cuda:0')

# Need to patch unembed for embedding extraction (same as EvoEmbeddingExtractor does)
from evo.embedding_analysis import IdentityUnembed
original_unembed = model.unembed
model.unembed = IdentityUnembed()

with torch.inference_mode():
    output, _ = model(input_ids)

model.unembed = original_unembed

nan_count = torch.isnan(output).sum().item()
inf_count = torch.isinf(output).sum().item()
print(f"  Output shape: {output.shape}")
print(f"  NaN count: {nan_count}")
print(f"  Inf count: {inf_count}")

if nan_count > 0 or inf_count > 0:
    print("\n  FAIL: Random model produces NaN/Inf!")
    print("  The poles initialization needs adjustment.")
    sys.exit(1)
else:
    mean_val = output.float().mean().item()
    std_val = output.float().std().item()
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std:  {std_val:.6f}")
    print("\n  PASS: No NaN/Inf. Safe to run full extraction.")
    sys.exit(0)
