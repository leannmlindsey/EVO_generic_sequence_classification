#!/usr/bin/env python3
"""
Quick CPU-only test to verify apply_savanna_style_init matches all parameters
in the Evo1 StripedHyena model. No GPU needed.
"""

import yaml
import pkgutil
import torch
import math
from stripedhyena.model import StripedHyena
from stripedhyena.utils import dotdict

config = yaml.safe_load(pkgutil.get_data('evo', 'configs/evo-1-8k-base_inference.yml'))
model = StripedHyena(dotdict(config, Loader=yaml.FullLoader))
sd = model.state_dict()

# Simulate what apply_savanna_style_init will match
matched = {}
unmatched = []

for key, tensor in sd.items():
    if '_extra_state' in key or not isinstance(tensor, torch.Tensor):
        continue

    if 'log_poles' in key or (key.endswith('.poles') and 'log_poles' not in key):
        matched[key] = 'poles'
    elif 'residues' in key:
        matched[key] = 'residues'
    elif key.endswith('.D'):
        matched[key] = 'D (zeros)'
    elif '.filter.h' in key or (key.endswith('.h') and 'filter' in key):
        matched[key] = 'filter.h'
    elif 'short_filter_weight' in key:
        matched[key] = 'short_filter (uniform)'
    elif 'short_filter_bias' in key:
        matched[key] = 'short_filter_bias (zeros)'
    elif 'norm' in key and ('weight' in key or 'scale' in key):
        matched[key] = 'norm (ones)'
    elif key.endswith('.bias'):
        matched[key] = 'bias (zeros)'
    elif ('out_filter_dense' in key or 'out_proj' in key) and 'weight' in key:
        matched[key] = 'output proj (wang_init)'
    elif 'weight' in key and tensor.dim() >= 2:
        matched[key] = 'weight (small_init)'
    else:
        unmatched.append((key, tuple(tensor.shape)))

print(f"Matched: {len(matched)} parameters")
print(f"Unmatched: {len(unmatched)} parameters")
print()
if unmatched:
    print("UNMATCHED (these keep constructor defaults):")
    for key, shape in unmatched:
        print(f"  {key}  shape={shape}")
else:
    print("ALL parameters matched - init is complete")
