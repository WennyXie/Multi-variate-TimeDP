#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect NPZ file for nuScenes trajectory data.
Verifies shapes, dtypes, and data quality.
"""

import numpy as np
import argparse
from pathlib import Path


def inspect_npz(npz_path: str):
    """Inspect NPZ file and print detailed statistics."""
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    
    print("\n" + "="*70)
    print("NPZ FILE CONTENTS")
    print("="*70)
    
    for key in data.files:
        arr = data[key]
        print(f"\n{key}:")
        print(f"  shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        
        if arr.ndim > 0 and arr.size > 0:
            print(f"  min: {arr.min():.6f}")
            print(f"  max: {arr.max():.6f}")
            
            if np.issubdtype(arr.dtype, np.floating):
                print(f"  mean: {arr.mean():.6f}")
                print(f"  std: {arr.std():.6f}")
                
                if np.isnan(arr).any():
                    n_nan = np.isnan(arr).sum()
                    print(f"  ⚠️ WARNING: contains {n_nan} NaN values!")
                else:
                    print(f"  NaN: None ✓")
                    
                if np.isinf(arr).any():
                    n_inf = np.isinf(arr).sum()
                    print(f"  ⚠️ WARNING: contains {n_inf} Inf values!")
                else:
                    print(f"  Inf: None ✓")
    
    # Check expected format
    print("\n" + "="*70)
    print("FORMAT ANALYSIS")
    print("="*70)
    
    if 'X_train' in data.files:
        X_train = data['X_train']
        print(f"\nX_train shape: {X_train.shape}")
        
        if X_train.ndim == 3:
            N, dim1, dim2 = X_train.shape
            
            # TimeCraft expects (N, F, T) where F=6, T=32
            if dim1 == 32 and dim2 == 6:
                print("  Current format: (N, T, F) - NEEDS TRANSPOSE to (N, F, T)")
                print(f"  After transpose: ({N}, {dim2}, {dim1}) = (N, F=6, T=32)")
            elif dim1 == 6 and dim2 == 32:
                print("  Current format: (N, F, T) - CORRECT ✓")
            else:
                print(f"  Unknown format: dim1={dim1}, dim2={dim2}")
    
    if 'X_test' in data.files:
        X_test = data['X_test']
        print(f"\nX_test shape: {X_test.shape}")
        
        if X_test.ndim == 3:
            N, dim1, dim2 = X_test.shape
            if dim1 == 32 and dim2 == 6:
                print("  Current format: (N, T, F) - NEEDS TRANSPOSE to (N, F, T)")
            elif dim1 == 6 and dim2 == 32:
                print("  Current format: (N, F, T) - CORRECT ✓")
    
    # Per-channel statistics
    print("\n" + "="*70)
    print("PER-CHANNEL STATISTICS (from X_train)")
    print("="*70)
    
    if 'X_train' in data.files:
        X_train = data['X_train']
        feature_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
        
        # Determine if we need to transpose
        if X_train.shape[1] == 32 and X_train.shape[2] == 6:
            # (N, T, F) format - transpose for analysis
            X_train_FT = X_train.transpose(0, 2, 1)  # (N, F, T)
        else:
            X_train_FT = X_train
        
        print(f"\nShape for analysis: {X_train_FT.shape} (N, F, T)")
        
        for i, name in enumerate(feature_names):
            channel_data = X_train_FT[:, i, :]
            print(f"\n  {name} (channel {i}):")
            print(f"    mean: {channel_data.mean():.6f}")
            print(f"    std:  {channel_data.std():.6f}")
            print(f"    min:  {channel_data.min():.6f}")
            print(f"    max:  {channel_data.max():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Inspect NPZ file')
    parser.add_argument('--npz_path', type=str, 
                        default='data/nuscenes_traj_40scenes_T32_stride2_seed0/dataset.npz',
                        help='Path to NPZ file')
    args = parser.parse_args()
    
    inspect_npz(args.npz_path)


if __name__ == '__main__':
    main()

