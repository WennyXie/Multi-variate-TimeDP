#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Few-shot Evaluation for nuScenes Multivariate TimeDP

Compares:
1. No-prompt (m=0): Generate without domain prompt
2. K-shot prompt: Generate with K Singapore samples as prompt

Metrics:
- Per-channel (v, a, curvature) mean/std gap
- Wasserstein distance per channel
- 6x6 correlation matrix Frobenius error
"""

import os
os.environ['DATA_ROOT'] = '/u/pyt9dp/TimeCraft/TimeDP/data'

import sys
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from ldm.util import instantiate_from_config


def wasserstein_1d(p, q, n_bins=50):
    """Compute 1D Wasserstein distance between two samples."""
    # Use scipy's wasserstein_distance
    return stats.wasserstein_distance(p.flatten(), q.flatten())


def correlation_matrix(data):
    """Compute correlation matrix for multivariate data.
    
    Args:
        data: (N, F, T) array
        
    Returns:
        (F, F) correlation matrix
    """
    N, F, T = data.shape
    # Reshape to (N*T, F) for correlation
    flat = data.transpose(0, 2, 1).reshape(-1, F)
    return np.corrcoef(flat.T)


def frobenius_error(A, B):
    """Frobenius norm of difference between two matrices."""
    return np.linalg.norm(A - B, 'fro')


def compute_metrics(real_data, gen_data, feature_names):
    """
    Compute comparison metrics between real and generated data.
    
    Args:
        real_data: (N, F, T) real samples
        gen_data: (N, F, T) generated samples
        feature_names: list of feature names
        
    Returns:
        dict of metrics
    """
    metrics = {}
    
    # Per-channel statistics
    for i, name in enumerate(feature_names):
        real_ch = real_data[:, i, :]
        gen_ch = gen_data[:, i, :]
        
        # Mean/Std gap
        mean_gap = abs(real_ch.mean() - gen_ch.mean())
        std_gap = abs(real_ch.std() - gen_ch.std())
        
        # Wasserstein distance
        wd = wasserstein_1d(real_ch, gen_ch)
        
        metrics[f'{name}_mean_gap'] = mean_gap
        metrics[f'{name}_std_gap'] = std_gap
        metrics[f'{name}_wasserstein'] = wd
    
    # Correlation matrix error
    real_corr = correlation_matrix(real_data)
    gen_corr = correlation_matrix(gen_data)
    corr_frob_error = frobenius_error(real_corr, gen_corr)
    metrics['corr_frobenius_error'] = corr_frob_error
    
    return metrics, real_corr, gen_corr


def load_model_from_checkpoint(config_path, ckpt_path):
    """Load model from config and checkpoint."""
    config = OmegaConf.load(config_path)
    
    # Set model params
    config.model.params.unet_config.params.seq_len = 32
    config.model.params.seq_len = 32
    config.model.params.channels = 6
    
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt_path)
    model = model.cuda()
    model.eval()
    
    return model


def sample_no_prompt(model, n_samples=32, dummy_data=None):
    """
    Generate samples without prompt by setting mask=0 (no attention to prototypes).
    
    We still need to provide conditioning structure but with mask=-inf 
    to effectively disable prompt influence.
    """
    with torch.no_grad():
        if dummy_data is not None:
            # Use dummy data to get conditioning structure, then zero out mask
            x = torch.tensor(dummy_data[:n_samples]).float().cuda()
            c, mask = model.get_learned_conditioning(x, return_mask=True)
            
            # Set mask to very negative values (like -inf) to disable all prototypes
            # This effectively makes the model ignore the prompt
            mask_zero = torch.full_like(mask, -1e4)  # Large negative = no attention
            
            samples, _ = model.sample_log(
                cond=c,
                batch_size=n_samples,
                ddim=False,
                cfg_scale=1,
                mask=mask_zero
            )
        else:
            # Fallback: try unconditional
            samples, _ = model.sample_log(
                cond=None, 
                batch_size=n_samples, 
                ddim=False, 
                cfg_scale=1,
                mask=None
            )
        
        gen_data = model.decode_first_stage(samples).detach().cpu().numpy()
    
    return gen_data  # (N, F, T)


def sample_with_prompt(model, prompt_data, n_samples=32):
    """
    Generate samples with K-shot prompt.
    
    Args:
        model: trained model
        prompt_data: (K, F, T) prompt samples from Singapore
        n_samples: number of samples to generate
    """
    K = len(prompt_data)
    
    with torch.no_grad():
        # Get conditioning from prompt data
        x = torch.tensor(prompt_data).float().cuda()  # (K, F, T)
        c, mask = model.get_learned_conditioning(x, return_mask=True)
        
        # Repeat to get n_samples
        repeats = (n_samples + K - 1) // K
        cond = torch.repeat_interleave(c, repeats, dim=0)[:n_samples]
        
        if mask is not None:
            mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)[:n_samples]
        else:
            mask_repeat = None
        
        # Sample
        samples, _ = model.sample_log(
            cond=cond,
            batch_size=n_samples,
            ddim=False,
            cfg_scale=1,
            mask=mask_repeat
        )
        gen_data = model.decode_first_stage(samples).detach().cpu().numpy()
    
    return gen_data  # (N, F, T)


def main():
    print("\n" + "="*70)
    print("NUSCENES FEW-SHOT EVALUATION")
    print("="*70)
    
    # Paths
    data_dir = Path('data/nuscenes_traj_40scenes_T32_stride2_seed0')
    
    # Find latest checkpoint from any nuscenes run
    # Use planA config
    config_path = 'configs/nuscenes_planA_timedp.yaml'
    
    # Find checkpoint from planA logs
    ckpt_dirs = [
        Path('logs/nuscenes_planA'),
    ]
    
    ckpt_pattern = []
    for ckpt_dir in ckpt_dirs:
        if ckpt_dir.exists():
            ckpt_pattern.extend(list(ckpt_dir.glob('*/checkpoints/*.ckpt')))
    
    if not ckpt_pattern:
        print("No checkpoint found! Run training first.")
        return
    
    # Get best checkpoint (not last.ckpt)
    ckpt_files = [p for p in ckpt_pattern if 'last' not in p.name]
    if ckpt_files:
        ckpt_path = sorted(ckpt_files)[-1]
    else:
        ckpt_path = ckpt_pattern[0]
    
    print(f"\nUsing checkpoint: {ckpt_path}")
    print(f"Config: {config_path}")
    
    # Load data
    test_data = np.load(data_dir / 'test.npy')  # (N, F, T) Singapore
    norm_stats = np.load(data_dir / 'norm_stats.npz')
    mean, std = norm_stats['mean'], norm_stats['std']
    
    feature_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
    
    print(f"\nTest data (Singapore): {test_data.shape}")
    
    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(config_path, str(ckpt_path))
    print("Model loaded!")
    
    # Parameters
    n_samples = 48  # Number of samples to generate
    K = 4  # Few-shot K
    
    # Select K random prompts from Singapore test data
    np.random.seed(123)
    prompt_indices = np.random.choice(len(test_data), K, replace=False)
    prompt_data = test_data[prompt_indices]
    
    print(f"\nPrompt samples (K={K}): indices {prompt_indices}")
    
    # ==========================================
    # 1. No-prompt sampling (m=0, disable prototypes)
    # ==========================================
    print("\n" + "-"*50)
    print("1. NO-PROMPT SAMPLING (mask=-inf, prototypes disabled)")
    print("-"*50)
    
    # Use test data as dummy to get conditioning structure
    gen_no_prompt = sample_no_prompt(model, n_samples, dummy_data=test_data)
    print(f"Generated shape: {gen_no_prompt.shape}")
    
    metrics_no_prompt, _, _ = compute_metrics(test_data, gen_no_prompt, feature_names)
    
    print("\nMetrics (No-prompt vs Singapore real):")
    for key, val in metrics_no_prompt.items():
        print(f"  {key}: {val:.4f}")
    
    # ==========================================
    # 2. K-shot prompt sampling
    # ==========================================
    print("\n" + "-"*50)
    print(f"2. K-SHOT PROMPT SAMPLING (K={K})")
    print("-"*50)
    
    gen_kshot = sample_with_prompt(model, prompt_data, n_samples)
    print(f"Generated shape: {gen_kshot.shape}")
    
    metrics_kshot, real_corr, gen_corr = compute_metrics(test_data, gen_kshot, feature_names)
    
    print(f"\nMetrics (K={K}-shot vs Singapore real):")
    for key, val in metrics_kshot.items():
        print(f"  {key}: {val:.4f}")
    
    # ==========================================
    # 3. Comparison Summary
    # ==========================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Focus on key features: v, a, curvature
    key_features = ['v', 'a', 'curvature']
    
    print("\n{:<15} {:>15} {:>15} {:>10}".format(
        "Metric", "No-Prompt", "K-shot", "Better?"
    ))
    print("-" * 55)
    
    improvements = 0
    for feat in key_features:
        for metric_type in ['wasserstein', 'mean_gap', 'std_gap']:
            key = f'{feat}_{metric_type}'
            no_p = metrics_no_prompt[key]
            kshot = metrics_kshot[key]
            better = "✓ K-shot" if kshot < no_p else "✗ No-prompt"
            if kshot < no_p:
                improvements += 1
            print(f"{key:<15} {no_p:>15.4f} {kshot:>15.4f} {better:>10}")
    
    # Correlation matrix error
    corr_np = metrics_no_prompt['corr_frobenius_error']
    corr_ks = metrics_kshot['corr_frobenius_error']
    better = "✓ K-shot" if corr_ks < corr_np else "✗ No-prompt"
    if corr_ks < corr_np:
        improvements += 1
    print(f"{'corr_frob':<15} {corr_np:>15.4f} {corr_ks:>15.4f} {better:>10}")
    
    print("\n" + "="*70)
    total_metrics = len(key_features) * 3 + 1  # 3 types per feature + corr
    print(f"K-shot better on {improvements}/{total_metrics} metrics")
    
    if improvements > total_metrics // 2:
        print("✓ PROMPT MECHANISM EFFECTIVE: K-shot outperforms no-prompt!")
    else:
        print("⚠ Mixed results - may need more training or data")
    print("="*70)
    
    # Save results
    results = {
        'no_prompt': metrics_no_prompt,
        'kshot': metrics_kshot,
        'K': K,
        'n_samples': n_samples
    }
    
    save_path = data_dir / 'fewshot_eval_results.npz'
    np.savez(save_path, **results)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()

