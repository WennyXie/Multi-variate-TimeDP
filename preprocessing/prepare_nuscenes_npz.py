#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare nuScenes trajectory data for TimeDP/TimeCraft.
Converts dataset.npz to train.npy/test.npy with shape (N, F, T).

TimeCraft/TarDiff expects (N, F, T) format:
  N = number of samples
  F = number of features (6 for our case)
  T = sequence length (32)
"""

import numpy as np
import argparse
from pathlib import Path


def prepare_nuscenes_data(
    npz_path: str,
    output_dir: str = None,
    n_prompt: int = 32,
    prompt_seed: int = 42
):
    """
    Convert dataset.npz to train.npy/test.npy/prompt_pool.npy.
    
    Args:
        npz_path: Path to dataset.npz
        output_dir: Output directory (defaults to same as npz_path parent)
        n_prompt: Number of samples for prompt pool
        prompt_seed: Random seed for prompt pool selection
    """
    npz_path = Path(npz_path)
    if output_dir is None:
        output_dir = npz_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    
    X_train = data['X_train']  # Currently (N, T, F) = (N, 32, 6)
    X_test = data['X_test']    # Currently (N, T, F) = (N, 32, 6)
    mean = data['mean']
    std = data['std']
    
    print(f"\nOriginal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Determine if transpose is needed
    # We expect (N, T=32, F=6) and need to convert to (N, F=6, T=32)
    if X_train.shape[1] == 32 and X_train.shape[2] == 6:
        print("\nTransposing from (N, T, F) to (N, F, T)...")
        X_train = X_train.transpose(0, 2, 1)  # (N, F, T)
        X_test = X_test.transpose(0, 2, 1)    # (N, F, T)
    elif X_train.shape[1] == 6 and X_train.shape[2] == 32:
        print("\nData already in (N, F, T) format, no transpose needed.")
    else:
        raise ValueError(f"Unexpected shape: {X_train.shape}")
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Validate: no NaN/Inf
    assert not np.isnan(X_train).any(), "X_train contains NaN!"
    assert not np.isnan(X_test).any(), "X_test contains NaN!"
    assert not np.isinf(X_train).any(), "X_train contains Inf!"
    assert not np.isinf(X_test).any(), "X_test contains Inf!"
    print("\n✓ No NaN or Inf values")
    
    # Check normalization (Boston train should be ~0 mean, ~1 std)
    train_mean = X_train.mean()
    train_std = X_train.std()
    print(f"\nTrain data statistics (overall):")
    print(f"  mean: {train_mean:.4f} (expected ~0)")
    print(f"  std:  {train_std:.4f} (expected ~1)")
    
    # Test data may deviate (different domain)
    test_mean = X_test.mean()
    test_std = X_test.std()
    print(f"\nTest data statistics (Singapore, may deviate):")
    print(f"  mean: {test_mean:.4f}")
    print(f"  std:  {test_std:.4f}")
    
    # Save train.npy
    train_path = output_dir / 'train.npy'
    np.save(train_path, X_train.astype(np.float32))
    print(f"\nSaved: {train_path} - shape {X_train.shape}")
    
    # Save test.npy
    test_path = output_dir / 'test.npy'
    np.save(test_path, X_test.astype(np.float32))
    print(f"Saved: {test_path} - shape {X_test.shape}")
    
    # Create prompt pool from test (Singapore) data
    np.random.seed(prompt_seed)
    n_test = len(X_test)
    n_prompt = min(n_prompt, n_test)
    prompt_indices = np.random.choice(n_test, n_prompt, replace=False)
    prompt_pool = X_test[prompt_indices]
    
    prompt_path = output_dir / 'prompt_pool.npy'
    np.save(prompt_path, prompt_pool.astype(np.float32))
    print(f"Saved: {prompt_path} - shape {prompt_pool.shape}")
    
    # Save normalization stats
    stats_path = output_dir / 'norm_stats.npz'
    np.savez(stats_path, mean=mean, std=std)
    print(f"Saved: {stats_path}")
    
    print("\n" + "="*60)
    print("PREPARATION COMPLETE!")
    print("="*60)
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Prompt pool samples: {len(prompt_pool)}")
    print(f"Shape: (N, F={X_train.shape[1]}, T={X_train.shape[2]})")
    
    return {
        'train_path': str(train_path),
        'test_path': str(test_path),
        'prompt_path': str(prompt_path),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_prompt': len(prompt_pool),
        'shape': X_train.shape
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare nuScenes data for TimeDP')
    parser.add_argument('--npz_path', type=str, 
                        default='data/nuscenes_traj_40scenes_T32_stride2_seed0/dataset.npz',
                        help='Path to dataset.npz')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to same as npz_path parent)')
    parser.add_argument('--n_prompt', type=int, default=32,
                        help='Number of samples for prompt pool')
    parser.add_argument('--prompt_seed', type=int, default=42,
                        help='Random seed for prompt pool selection')
    
    args = parser.parse_args()
    
    prepare_nuscenes_data(
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        n_prompt=args.n_prompt,
        prompt_seed=args.prompt_seed
    )


if __name__ == '__main__':
    main()

