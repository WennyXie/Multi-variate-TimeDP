#!/usr/bin/env python3
"""
Plan A Evaluation Script for nuScenes multivariate TimeDP.

Supports:
- Seen domains (D0, D1, D2) evaluation with disjoint prompts/eval windows
- Unseen domain (D3) evaluation with prompt_pool/eval_pool
- Ablation modes: no_prompt, k_shot, minus_pam, shuffled
- TimeDP paper metrics: MMD, KL, MDD
- Our metrics: Wasserstein, corr_frob

Task 1: Centered-PIT diagnostics (--run_pit_diagnostics)
Task 2: Correlation heatmaps (--save_corr_heatmaps)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Add TimeDP root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# Import eval modules for Task 1 and Task 2
from evaluation.eval_modules.pit_diagnostics import (
    run_pit_diagnostics, 
    apply_pit_clip,
    CenteredPITInverter
)
from evaluation.eval_modules.plot_corr_heatmaps import run_corr_heatmaps


def compute_mmd(x, y, kernel='rbf', gamma=None):
    """Compute Maximum Mean Discrepancy."""
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    
    xx = np.exp(-gamma * np.sum((x[:, None] - x[None, :]) ** 2, axis=-1))
    yy = np.exp(-gamma * np.sum((y[:, None] - y[None, :]) ** 2, axis=-1))
    xy = np.exp(-gamma * np.sum((x[:, None] - y[None, :]) ** 2, axis=-1))
    
    mmd = np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
    return float(max(0, mmd))


def compute_kl(real, gen, n_bins=50):
    """Compute KL divergence per channel."""
    kl_list = []
    for ch in range(real.shape[-1]):
        r = real[..., ch].flatten()
        g = gen[..., ch].flatten()
        
        # Use common bin edges
        min_val = min(r.min(), g.min())
        max_val = max(r.max(), g.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        hist_r, _ = np.histogram(r, bins=bins, density=True)
        hist_g, _ = np.histogram(g, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist_r = hist_r + eps
        hist_g = hist_g + eps
        
        # Normalize
        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        
        kl = np.sum(hist_r * np.log(hist_r / hist_g))
        kl_list.append(float(kl))
    
    return float(np.mean(kl_list)), kl_list


def compute_mdd(real, gen):
    """Compute Marginal Distribution Distance (mean of Wasserstein per channel)."""
    mdd_list = []
    for ch in range(real.shape[-1]):
        r = real[..., ch].flatten()
        g = gen[..., ch].flatten()
        wass = stats.wasserstein_distance(r, g)
        mdd_list.append(float(wass))
    return float(np.mean(mdd_list)), mdd_list


def compute_wasserstein(real, gen, channel_idx):
    """Compute Wasserstein distance for specific channel."""
    r = real[..., channel_idx].flatten()
    g = gen[..., channel_idx].flatten()
    return float(stats.wasserstein_distance(r, g))


def _spearman_corr_matrix(X):
    """Compute Spearman (rank) correlation matrix for 2D array X of shape (N, D).
    
    Why Spearman: Spearman correlation is invariant to monotonic per-channel
    transforms. This is crucial when comparing results across different
    normalization schemes (zscore vs centered_pit) because Spearman only
    considers relative ordering of values, not their absolute scale or
    distribution shape. This makes the metric fair across normalizations.
    
    Args:
        X: array of shape (N, D) where N is samples, D is channels
        
    Returns:
        C: array of shape (D, D) - Spearman correlation matrix
    """
    N, D = X.shape
    
    # Convert each column to ranks (1-based)
    # Using scipy.stats.rankdata for proper tie handling (average method)
    try:
        from scipy.stats import rankdata
        ranks = np.zeros_like(X, dtype=np.float64)
        for j in range(D):
            ranks[:, j] = rankdata(X[:, j], method='average')
    except ImportError:
        # Fallback: numpy-based ranking (approximate tie handling via argsort)
        # Note: This gives ties the rank of their first occurrence, not average
        ranks = np.zeros_like(X, dtype=np.float64)
        for j in range(D):
            order = np.argsort(X[:, j])
            ranks[order, j] = np.arange(1, N + 1)
    
    # Compute Pearson correlation on ranks = Spearman correlation
    C = np.corrcoef(ranks.T)
    return C


def compute_corr_frob(real, gen, return_per_timestep=False, diagnostics=False, eps=1e-6):
    """Compute MASKED per-timestep Spearman correlation Frobenius distance, then average.
    
    Why masked: Near-constant channels (std < eps) produce NaN correlations. Instead of
    replacing NaNs with 0 (which distorts the metric), we mask out low-variance channels
    and compute correlation only on valid channels. This prevents a few bad channels from
    dominating the metric.
    
    Why per-timestep: Flattening across time mixes samples from different temporal
    positions, which can obscure time-varying correlation structure.
    
    Why Spearman: Spearman correlation is invariant to monotonic per-channel transforms
    (e.g., zscore vs centered_pit), making the metric fair across normalizations.
    
    Args:
        real: array of shape (N, T, D) - real denormalized data
        gen: array of shape (N, T, D) - generated denormalized data
        return_per_timestep: if True, also return the per-timestep distances
        diagnostics: if True, return detailed diagnostic dict
        eps: threshold for near-constant detection (default 1e-6)
        
    Returns:
        float: mean Frobenius distance over valid timesteps
        (optional) np.ndarray: per-timestep distances if return_per_timestep=True
        (optional) dict: diagnostics if diagnostics=True
    """
    N, T, D = real.shape
    assert gen.shape == real.shape, f"Shape mismatch: real {real.shape} vs gen {gen.shape}"
    
    frob_distances = []       # Only valid timesteps
    valid_count_per_t = []    # Number of valid channels per timestep
    d_t_all = []              # Distance for all timesteps (NaN for skipped)
    skipped_timesteps = 0
    
    for t in range(T):
        real_t = real[:, t, :]  # (N, D)
        gen_t = gen[:, t, :]    # (N, D)
        
        # Compute per-channel std
        std_real = np.std(real_t, axis=0)  # (D,)
        std_gen = np.std(gen_t, axis=0)    # (D,)
        
        # Valid channels: both real and gen have std > eps
        valid_mask = (std_real > eps) & (std_gen > eps)
        n_valid = valid_mask.sum()
        valid_count_per_t.append(int(n_valid))
        
        # Skip timestep if < 2 valid channels (need at least 2 for meaningful correlation)
        if n_valid < 2:
            skipped_timesteps += 1
            d_t_all.append(np.nan)
            continue
        
        # Extract only valid channels
        real_valid = real_t[:, valid_mask]  # (N, n_valid)
        gen_valid = gen_t[:, valid_mask]    # (N, n_valid)
        
        # Compute Spearman correlation on valid channels only
        corr_real = _spearman_corr_matrix(real_valid)  # (n_valid, n_valid)
        corr_gen = _spearman_corr_matrix(gen_valid)    # (n_valid, n_valid)
        
        # Frobenius distance (should be NaN-free now due to masking)
        d_t = np.linalg.norm(corr_real - corr_gen, 'fro')
        frob_distances.append(d_t)
        d_t_all.append(d_t)
    
    # Compute mean over valid timesteps only
    if len(frob_distances) == 0:
        mean_frob = 0.0  # Edge case: all timesteps skipped
    else:
        mean_frob = float(np.mean(frob_distances))
    
    kept_T = T - skipped_timesteps
    avg_valid_channels = np.mean(valid_count_per_t) if valid_count_per_t else 0
    
    # Build return values
    if diagnostics:
        diag_dict = {
            'd_t': d_t_all,  # NaN for skipped timesteps
            'valid_count_per_t': valid_count_per_t,
            'skipped_timesteps': skipped_timesteps,
            'kept_timesteps': kept_T,
            'total_timesteps': T,
            'avg_valid_channels': float(avg_valid_channels),
            'corr_frob_masked': mean_frob,
            'eps': eps,
        }
        
        if return_per_timestep:
            return mean_frob, np.array(d_t_all), diag_dict
        return mean_frob, diag_dict
    
    if return_per_timestep:
        return mean_frob, np.array(d_t_all)
    return mean_frob


def _spearman_cross_corr_matrix(X, Y):
    """Compute Spearman cross-correlation matrix between X and Y.
    
    Args:
        X: array of shape (N, D1)
        Y: array of shape (N, D2)
        
    Returns:
        C: array of shape (D1, D2) where C[i,j] = spearman_corr(X[:,i], Y[:,j])
    """
    N, D1 = X.shape
    D2 = Y.shape[1]
    
    # Rank transform
    try:
        from scipy.stats import rankdata
        ranks_X = np.zeros_like(X, dtype=np.float64)
        ranks_Y = np.zeros_like(Y, dtype=np.float64)
        for j in range(D1):
            ranks_X[:, j] = rankdata(X[:, j], method='average')
        for j in range(D2):
            ranks_Y[:, j] = rankdata(Y[:, j], method='average')
    except ImportError:
        ranks_X = np.zeros_like(X, dtype=np.float64)
        ranks_Y = np.zeros_like(Y, dtype=np.float64)
        for j in range(D1):
            order = np.argsort(X[:, j])
            ranks_X[order, j] = np.arange(1, N + 1)
        for j in range(D2):
            order = np.argsort(Y[:, j])
            ranks_Y[order, j] = np.arange(1, N + 1)
    
    # Compute cross-correlation: C[i,j] = corr(ranks_X[:,i], ranks_Y[:,j])
    # Standardize
    ranks_X = (ranks_X - ranks_X.mean(axis=0)) / (ranks_X.std(axis=0) + 1e-10)
    ranks_Y = (ranks_Y - ranks_Y.mean(axis=0)) / (ranks_Y.std(axis=0) + 1e-10)
    
    # Cross-correlation matrix
    C = (ranks_X.T @ ranks_Y) / N
    return C


def compute_lagged_corr_frob(real, gen, L=4, eps=1e-6, diagnostics=False):
    """Compute lagged cross-correlation Frobenius distance (masked Spearman).
    
    For each lag ℓ in {0..L}, computes cross-correlation between timestep t and t+ℓ,
    then measures the Frobenius distance between real and generated correlation structures.
    
    This metric captures temporal dependencies: how well the generated data preserves
    the correlation between different time points (autocorrelation structure).
    
    Args:
        real: array of shape (N, T, D) - real denormalized data
        gen: array of shape (N, T, D) - generated denormalized data
        L: maximum lag to compute (default 4)
        eps: threshold for near-constant detection (default 1e-6)
        diagnostics: if True, return detailed diagnostic dict
        
    Returns:
        dict: { 'L': L, 'D_lags': [D_0, ..., D_L], 'kept_pairs': [...] }
        (optional) detailed diagnostics if diagnostics=True
    """
    N, T, D = real.shape
    assert gen.shape == real.shape, f"Shape mismatch: real {real.shape} vs gen {gen.shape}"
    
    D_lags = []
    kept_pairs_per_lag = []
    all_d_t_lag = []  # For diagnostics
    
    for lag in range(L + 1):
        frob_distances = []
        d_t_this_lag = []
        
        for t in range(T - lag):
            # X = timestep t, Y = timestep t+lag
            Xr = real[:, t, :]        # (N, D)
            Yr = real[:, t + lag, :]  # (N, D)
            Xg = gen[:, t, :]         # (N, D)
            Yg = gen[:, t + lag, :]   # (N, D)
            
            # Compute std for masking
            std_Xr = np.std(Xr, axis=0)  # (D,)
            std_Yr = np.std(Yr, axis=0)
            std_Xg = np.std(Xg, axis=0)
            std_Yg = np.std(Yg, axis=0)
            
            # Valid channels: both real and gen have std > eps
            valid_i = (std_Xr > eps) & (std_Xg > eps)  # for X dimension
            valid_j = (std_Yr > eps) & (std_Yg > eps)  # for Y dimension
            
            # Skip if not enough valid channels
            if valid_i.sum() < 1 or valid_j.sum() < 1:
                d_t_this_lag.append(np.nan)
                continue
            
            # Extract valid channels
            Xr_valid = Xr[:, valid_i]
            Yr_valid = Yr[:, valid_j]
            Xg_valid = Xg[:, valid_i]
            Yg_valid = Yg[:, valid_j]
            
            # Compute cross-correlation matrices
            C_real = _spearman_cross_corr_matrix(Xr_valid, Yr_valid)
            C_gen = _spearman_cross_corr_matrix(Xg_valid, Yg_valid)
            
            # Frobenius distance
            d_t = np.linalg.norm(C_real - C_gen, 'fro')
            frob_distances.append(d_t)
            d_t_this_lag.append(d_t)
        
        # Mean over kept timesteps
        if len(frob_distances) == 0:
            D_lag = 0.0
        else:
            D_lag = float(np.mean(frob_distances))
        
        D_lags.append(D_lag)
        kept_pairs_per_lag.append(len(frob_distances))
        all_d_t_lag.append(d_t_this_lag)
    
    result = {
        'L': L,
        'D_lags': D_lags,
        'kept_pairs': kept_pairs_per_lag,
    }
    
    if diagnostics:
        result['d_t_per_lag'] = all_d_t_lag
        result['total_pairs_per_lag'] = [T - lag for lag in range(L + 1)]
    
    return result


def save_corr_diagnostics(diag_dict, output_dir, normalizer, domain, k, seed=0):
    """Save correlation diagnostics to JSON file.
    
    Args:
        diag_dict: diagnostics dict from compute_corr_frob(..., diagnostics=True)
        output_dir: directory to save the file
        normalizer: normalization method name (e.g., 'zscore', 'centered_pit')
        domain: domain identifier (e.g., 'D0', 'D1', 'D2', 'D3_unseen')
        k: number of prompts
        seed: random seed
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"debug_corr_{normalizer}_{domain}_K{k}_seed{seed}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(diag_dict, f, indent=2)
    
    print(f"  [Diagnostics] Saved to {filepath}")
    print(f"    corr_frob (masked): {diag_dict['corr_frob_masked']:.4f}")
    print(f"    Timesteps: {diag_dict['kept_timesteps']}/{diag_dict['total_timesteps']} kept (skipped {diag_dict['skipped_timesteps']})")
    print(f"    Avg valid channels: {diag_dict['avg_valid_channels']:.1f}/{len(diag_dict['valid_count_per_t']) and 6}")


def compute_stats(data, name):
    """Compute sanity statistics."""
    return {
        f'{name}_mean': float(np.mean(data)),
        f'{name}_std': float(np.std(data)),
        f'{name}_min': float(np.min(data)),
        f'{name}_max': float(np.max(data)),
    }


def sample_no_prompt(model, n_samples=128, batch_size=32):
    """Sample with no prompt (m=0)."""
    all_samples = []
    
    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        
        # Create dummy context
        dummy = torch.zeros(bs, 6, 32).to(model.device)
        c, mask = model.get_learned_conditioning(dummy, return_mask=True)
        
        # Zero out the mask (no prompt effect)
        mask_zero = torch.full_like(mask, -1e9)
        
        samples = model.sample_log(
            cond=c,
            batch_size=bs,
            ddim=False,
            cfg_scale=1.0,
            mask=mask_zero
        )[0]
        
        # samples: (B, C, T) -> (B, T, C)
        samples = samples.permute(0, 2, 1).cpu().numpy()
        all_samples.append(samples)
    
    return np.concatenate(all_samples, axis=0)


def check_nan_inf(arr, name="array"):
    """Check for NaN/Inf in array and return diagnostics."""
    total = arr.size
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    finite_count = np.isfinite(arr).sum()
    
    nan_frac = nan_count / total
    inf_frac = inf_count / total
    finite_frac = finite_count / total
    
    return {
        'name': name,
        'shape': arr.shape,
        'nan_frac': float(nan_frac),
        'inf_frac': float(inf_frac),
        'finite_frac': float(finite_frac),
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'has_nan': nan_count > 0,
        'has_inf': inf_count > 0
    }


def check_mask_all_masked(mask, threshold=-1e8):
    """
    Check if any sample has all prototypes masked (all values <= threshold).
    
    Args:
        mask: (B, num_prototypes) mask tensor
        threshold: values <= this are considered masked
        
    Returns:
        dict with diagnostic info
    """
    # Check per-sample: is every prototype masked?
    all_masked_per_sample = (mask <= threshold).all(dim=1)  # (B,)
    all_masked_indices = torch.where(all_masked_per_sample)[0].cpu().numpy()
    
    return {
        'num_all_masked': len(all_masked_indices),
        'total_samples': mask.shape[0],
        'all_masked_indices': all_masked_indices.tolist(),
        'has_all_masked': len(all_masked_indices) > 0
    }


def sample_k_shot(model, prompt_data, n_samples=128, batch_size=32, 
                  debug=False, eval_mask_fallback=False):
    """Sample with K-shot prompts (cyclic reuse).
    
    Args:
        model: the diffusion model
        prompt_data: (K, D, T) prompt data
        n_samples: number of samples to generate
        batch_size: batch size for generation
        debug: if True, print detailed diagnostics
        eval_mask_fallback: if True, set all-masked samples' mask to zeros
    """
    K = len(prompt_data)
    all_samples = []
    all_indices_used = []
    
    # Pre-compute conditioning for all prompts
    prompt_tensor = torch.from_numpy(prompt_data).float().to(model.device)
    c_all, mask_all = model.get_learned_conditioning(prompt_tensor, return_mask=True)
    
    if debug:
        print(f"\n[DEBUG] sample_k_shot diagnostics:")
        print(f"  prompt_data shape: {prompt_data.shape}")
        print(f"  c_all shape: {c_all.shape}")
        print(f"  mask_all shape: {mask_all.shape}")
        print(f"  mask_all min: {mask_all.min().item():.2f}, max: {mask_all.max().item():.2f}")
        
        # Check for all-masked prompts
        mask_diag = check_mask_all_masked(mask_all)
        print(f"  All-masked prompts: {mask_diag['num_all_masked']}/{mask_diag['total_samples']}")
        if mask_diag['has_all_masked']:
            print(f"    WARNING: Indices with all-masked: {mask_diag['all_masked_indices'][:10]}...")
    
    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        
        # Cyclic reuse of prompts
        indices = [(i + j) % K for j in range(bs)]
        all_indices_used.extend(indices)
        c = c_all[indices]
        mask = mask_all[indices]
        
        # Check and optionally fix all-masked samples
        if eval_mask_fallback:
            all_masked_per_sample = (mask <= -1e8).all(dim=1)
            if all_masked_per_sample.any():
                # Set mask to zeros for all-masked samples (uniform attention)
                mask[all_masked_per_sample] = 0.0
                if debug:
                    print(f"  [FALLBACK] Fixed {all_masked_per_sample.sum().item()} all-masked samples in batch {i//batch_size}")
        
        samples = model.sample_log(
            cond=c,
            batch_size=bs,
            ddim=False,
            cfg_scale=1.0,
            mask=mask
        )[0]
        
        samples = samples.permute(0, 2, 1).cpu().numpy()
        all_samples.append(samples)
    
    result = np.concatenate(all_samples, axis=0)
    
    # Final NaN/Inf check
    diag = check_nan_inf(result, "generated_samples")
    if debug or diag['has_nan'] or diag['has_inf']:
        print(f"\n[SAMPLE CHECK] Generated samples diagnostics:")
        print(f"  Shape: {diag['shape']}")
        print(f"  NaN frac: {diag['nan_frac']:.6f} ({diag['nan_count']} values)")
        print(f"  Inf frac: {diag['inf_frac']:.6f} ({diag['inf_count']} values)")
        print(f"  Finite frac: {diag['finite_frac']:.6f}")
        
        if diag['has_nan'] or diag['has_inf']:
            print(f"  WARNING: Invalid values detected!")
    
    return result, {'nan_inf_diag': diag, 'indices_used': all_indices_used}


def sample_shuffled(model, prompt_data, n_samples=128, batch_size=32):
    """Sample with shuffled prompts (permute prototype dimension)."""
    K = len(prompt_data)
    all_samples = []
    
    prompt_tensor = torch.from_numpy(prompt_data).float().to(model.device)
    c_all, mask_all = model.get_learned_conditioning(prompt_tensor, return_mask=True)
    
    # Shuffle the prototype dimension
    perm = torch.randperm(c_all.shape[1])
    c_shuffled = c_all[:, perm, :]
    mask_shuffled = mask_all[:, perm]
    
    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        
        indices = [(i + j) % K for j in range(bs)]
        c = c_shuffled[indices]
        mask = mask_shuffled[indices]
        
        samples = model.sample_log(
            cond=c,
            batch_size=bs,
            ddim=False,
            cfg_scale=1.0,
            mask=mask
        )[0]
        
        samples = samples.permute(0, 2, 1).cpu().numpy()
        all_samples.append(samples)
    
    return np.concatenate(all_samples, axis=0)


def sample_minus_pam(model, prompt_data, n_samples=128, batch_size=32):
    """Sample with -PAM ablation (no sparsification, use dense prompt)."""
    K = len(prompt_data)
    all_samples = []
    
    prompt_tensor = torch.from_numpy(prompt_data).float().to(model.device)
    
    # Get raw conditioning without PAM sparsification
    # This requires accessing the encoder differently
    c_all, mask_all = model.get_learned_conditioning(prompt_tensor, return_mask=True)
    
    # For -PAM: don't mask negatives to -inf, keep original mask
    # Here we use the mask as-is without the -inf treatment
    # This is a simplified version - full implementation would modify the model
    mask_dense = torch.zeros_like(mask_all)  # No masking effect
    
    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        
        indices = [(i + j) % K for j in range(bs)]
        c = c_all[indices]
        mask = mask_dense[indices]
        
        samples = model.sample_log(
            cond=c,
            batch_size=bs,
            ddim=False,
            cfg_scale=1.0,
            mask=mask
        )[0]
        
        samples = samples.permute(0, 2, 1).cpu().numpy()
        all_samples.append(samples)
    
    return np.concatenate(all_samples, axis=0)


def evaluate_domain(
    model, 
    dm, 
    domain_id, 
    prompt_data,
    eval_data,
    K_values=[1, 4, 8, 16],
    N_gen=None,
    modes=['no_prompt', 'k_shot', 'shuffled', 'minus_pam']
):
    """Evaluate on a single domain with all modes and K values."""
    results = []
    
    # Match N_gen to eval_data size if not specified
    if N_gen is None:
        N_gen = len(eval_data)
        print(f"  [evaluate_domain] N_gen auto-set to {N_gen} (matched to eval_data)")
    
    # Denormalize evaluation data
    eval_denorm = dm.inverse_transform(eval_data, data_name='all')
    
    for mode in modes:
        for K in K_values:
            if mode == 'no_prompt' and K != K_values[0]:
                continue  # Only run no_prompt once
            
            print(f"  Domain {domain_id}, Mode: {mode}, K={K}")
            
            # Sample prompts
            if K > len(prompt_data):
                K_actual = len(prompt_data)
                print(f"    Warning: K={K} > available prompts ({len(prompt_data)}), using K={K_actual}")
            else:
                K_actual = K
            
            np.random.seed(42 + domain_id * 100 + K)
            prompt_indices = np.random.choice(len(prompt_data), K_actual, replace=False)
            prompts = prompt_data[prompt_indices]
            
            # Generate samples
            if mode == 'no_prompt':
                gen_norm = sample_no_prompt(model, N_gen)
            elif mode == 'k_shot':
                gen_norm, _ = sample_k_shot(model, prompts, N_gen)
            elif mode == 'shuffled':
                gen_norm = sample_shuffled(model, prompts, N_gen)
            elif mode == 'minus_pam':
                gen_norm = sample_minus_pam(model, prompts, N_gen)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Denormalize generated samples
            gen_denorm = dm.inverse_transform(gen_norm, data_name='all')
            
            # Compute metrics
            mmd = compute_mmd(eval_denorm, gen_denorm)
            kl_avg, kl_per_ch = compute_kl(eval_denorm, gen_denorm)
            mdd_avg, mdd_per_ch = compute_mdd(eval_denorm, gen_denorm)
            
            # Channel indices: 0=dx, 1=dy, 2=v, 3=a, 4=yaw_rate, 5=curvature
            v_wass = compute_wasserstein(eval_denorm, gen_denorm, 2)
            a_wass = compute_wasserstein(eval_denorm, gen_denorm, 3)
            curv_wass = compute_wasserstein(eval_denorm, gen_denorm, 5)
            corr_frob = compute_corr_frob(eval_denorm, gen_denorm)
            
            # Sanity stats
            sanity = {}
            for ch_name, ch_idx in [('v', 2), ('a', 3), ('curvature', 5)]:
                sanity.update(compute_stats(eval_denorm[..., ch_idx], f'real_{ch_name}'))
                sanity.update(compute_stats(gen_denorm[..., ch_idx], f'gen_{ch_name}'))
            
            result = {
                'domain': domain_id,
                'mode': mode,
                'K': K if mode != 'no_prompt' else 0,
                'N_gen': N_gen,
                'mmd': mmd,
                'kl': kl_avg,
                'mdd': mdd_avg,
                'v_wass': v_wass,
                'a_wass': a_wass,
                'curvature_wass': curv_wass,
                'corr_frob': corr_frob,
                **sanity
            }
            results.append(result)
            
            print(f"    MMD={mmd:.4f}, KL={kl_avg:.4f}, MDD={mdd_avg:.4f}")
            print(f"    v_wass={v_wass:.4f}, a_wass={a_wass:.4f}, curv_wass={curv_wass:.4f}")
            print(f"    corr_frob={corr_frob:.4f}")
    
    return results


def detect_normalizer_type(data_dir: Path) -> str:
    """Detect normalizer type from data directory name or files."""
    dir_name = data_dir.name.lower()
    
    if 'centered_pit' in dir_name or 'pit' in dir_name:
        return 'centered_pit'
    elif 'zscore_winsor' in dir_name or 'winsor' in dir_name:
        return 'zscore_winsor'
    elif 'zscore' in dir_name:
        return 'zscore'
    
    # Fallback: check for pit_params.json
    if (data_dir / 'pit_params.json').exists():
        return 'centered_pit'
    elif (data_dir / 'norm_stats.npz').exists():
        return 'zscore'
    
    return 'zscore'  # Default


def load_normalizer_params(data_dir: Path, normalizer_type: str):
    """Load normalizer parameters based on type."""
    if normalizer_type == 'centered_pit':
        pit_path = data_dir / 'pit_params.json'
        if pit_path.exists():
            with open(pit_path) as f:
                return json.load(f)
        return None
    else:
        # zscore or zscore_winsor
        stats_path = data_dir / 'norm_stats.npz'
        if stats_path.exists():
            stats = np.load(stats_path)
            return {'mean': stats['mean'], 'std': stats['std']}
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--config', type=str, default='configs/nuscenes_planA_timedp.yaml')
    parser.add_argument('--data_dir', type=str, default='data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--N_gen', type=int, default=128)
    parser.add_argument('--K_values', type=str, default='1,4,8,16')
    parser.add_argument('--modes', type=str, default='no_prompt,k_shot,shuffled')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--domains', type=str, default=None, help='Comma-separated domain IDs to evaluate (e.g., "0,1,2" for seen only, or "0,1,2,3" for all). If None, evaluates all domains.')
    
    # Task 1: PIT diagnostics
    parser.add_argument('--run_pit_diagnostics', action='store_true', default=False,
                        help='Run centered-PIT diagnostics (Task 1)')
    parser.add_argument('--pit_clip_output', action='store_true', default=False,
                        help='[EXPERIMENTAL] Clamp generated normalized outputs to [-1,1] before inverse-ECDF (centered_pit only)')
    
    # Task 2: Correlation heatmaps
    parser.add_argument('--save_corr_heatmaps', action='store_true', default=False,
                        help='Generate and save correlation heatmaps (Task 2)')
    parser.add_argument('--compute_lag1', action='store_true', default=True,
                        help='Compute lag-1 cross-correlations in heatmaps')
    parser.add_argument('--compute_noise_floor', action='store_true', default=False,
                        help='Compute bootstrap noise floor for correlation metrics')
    parser.add_argument('--n_bootstrap', type=int, default=50,
                        help='Number of bootstrap iterations for noise floor estimation')
    
    # Normalizer override (auto-detected by default)
    parser.add_argument('--normalizer', type=str, default=None,
                        choices=['zscore', 'zscore_winsor', 'centered_pit'],
                        help='Override normalizer type (auto-detected from data_dir if not specified)')
    
    # Debug flags
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug output for sampling')
    parser.add_argument('--eval_mask_fallback', action='store_true', default=False,
                        help='[DEBUG] Set mask to zeros for all-masked samples to fix NaN from PAM')
    
    # Save samples for UMAP visualization
    parser.add_argument('--save_samples', action='store_true', default=False,
                        help='Save generated and real samples as .npy files for UMAP visualization')
    
    args = parser.parse_args()
    
    # Parse K values and modes
    K_values = [int(k) for k in args.K_values.split(',')]
    modes = args.modes.split(',')
    
    # Parse domains
    if args.domains is None:
        # Default: evaluate all domains (0,1,2,3)
        eval_seen_domains = True
        eval_unseen_domain = True
        domain_list = [0, 1, 2, 3]
    else:
        domain_list = [int(d) for d in args.domains.split(',')]
        eval_seen_domains = any(d in domain_list for d in [0, 1, 2])
        eval_unseen_domain = 3 in domain_list
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path(args.data_dir) / 'fewshot_eval'
    else:
        args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect normalizer type
    data_dir = Path(args.data_dir)
    if args.normalizer is not None:
        normalizer_type = args.normalizer
    else:
        normalizer_type = detect_normalizer_type(data_dir)
    print(f"Normalizer type: {normalizer_type}")
    
    # Load normalizer parameters
    norm_params = load_normalizer_params(data_dir, normalizer_type)
    pit_params = norm_params if normalizer_type == 'centered_pit' else None
    zscore_mean = norm_params['mean'] if normalizer_type in ['zscore', 'zscore_winsor'] and norm_params else None
    zscore_std = norm_params['std'] if normalizer_type in ['zscore', 'zscore_winsor'] and norm_params else None
    
    # Load config and model
    print(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)
    
    print(f"Loading checkpoint from {args.ckpt}")
    model = instantiate_from_config(config.model)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading data from {args.data_dir}")
    npz = np.load(Path(args.data_dir) / 'dataset.npz')
    
    X_train = npz['X_train']  # (N, F, T)
    domain_train = npz['domain_train']
    X_unseen_prompt = npz['X_unseen_prompt_pool']
    X_unseen_eval = npz['X_unseen_eval_pool']
    
    # Get raw data for computing mean/std
    X_train_raw = npz['X_train_raw_NTD']  # (N, T, D)
    mean = X_train_raw.reshape(-1, 6).mean(axis=0).astype(np.float32)
    std = X_train_raw.reshape(-1, 6).std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    
    print(f"Train: {X_train.shape}, domains: {np.unique(domain_train)}")
    print(f"Unseen prompt: {X_unseen_prompt.shape}, eval: {X_unseen_eval.shape}")
    
    # Create simple datamodule-like object for denormalization
    class SimpleDM:
        def __init__(self, mean, std, normalizer_type='zscore', pit_params=None):
            self.mean = mean
            self.std = std
            self.normalizer_type = normalizer_type
            self.pit_params = pit_params
            self.pit_inverter = None
            
            if normalizer_type == 'centered_pit' and pit_params is not None:
                self.pit_inverter = CenteredPITInverter(pit_params)
        
        def inverse_transform(self, x, data_name='all'):
            # x: (N, T, D) or (N, D, T)
            if self.normalizer_type == 'centered_pit' and self.pit_inverter is not None:
                # For centered_pit, use inverse ECDF
                if x.shape[-1] == 6:
                    return self.pit_inverter.inverse_transform(x)
                else:
                    # (N, D, T) -> (N, T, D), denorm, keep as (N, T, D)
                    x_t = x.transpose(0, 2, 1)
                    return self.pit_inverter.inverse_transform(x_t)
            else:
                # For zscore/zscore_winsor, use mean/std
                if x.shape[-1] == 6:
                    return x * self.std + self.mean
                else:
                    # (N, D, T) -> transpose, denorm, keep
                    x_t = x.transpose(0, 2, 1)
                    return x_t * self.std + self.mean
    
    dm = SimpleDM(mean, std, normalizer_type, pit_params)
    
    all_results = []
    
    # Evaluate seen domains (0, 1, 2)
    if eval_seen_domains:
        seen_domains_to_eval = [d for d in [0, 1, 2] if d in domain_list]
        for domain_id in seen_domains_to_eval:
            print(f"\n=== Evaluating Seen Domain {domain_id} ===")
            
            # Get data for this domain
            domain_mask = domain_train == domain_id
            domain_data = X_train[domain_mask]  # (N, F, T)
            
            # Split into prompt and eval pools (disjoint)
            n = len(domain_data)
            n_prompt = n // 2
            np.random.seed(42 + domain_id)
            perm = np.random.permutation(n)
            prompt_pool = domain_data[perm[:n_prompt]]
            eval_pool = domain_data[perm[n_prompt:]]
            
            print(f"  Prompt pool: {len(prompt_pool)}, Eval pool: {len(eval_pool)}")
            
            # Need to transpose for evaluation: (N, F, T) -> (N, T, F) for metrics
            prompt_pool_t = prompt_pool.transpose(0, 2, 1)
            eval_pool_t = eval_pool.transpose(0, 2, 1)
            
            results = evaluate_domain(
                model, dm, domain_id, 
                prompt_pool, eval_pool_t,
                K_values=K_values, N_gen=None, modes=modes  # N_gen=None -> auto-match to eval_pool
            )
            all_results.extend(results)
    
    # Evaluate unseen domain (3)
    # Store generated samples for diagnostics/heatmaps
    last_gen_norm = None
    last_real_norm = None
    last_domain_id = None
    last_K = None
    
    if eval_unseen_domain:
        print(f"\n=== Evaluating Unseen Domain 3 ===")
        
        # Transpose for evaluation
        X_unseen_prompt_t = X_unseen_prompt.transpose(0, 2, 1)
        X_unseen_eval_t = X_unseen_eval.transpose(0, 2, 1)
        
        print(f"  Prompt pool: {len(X_unseen_prompt)}, Eval pool: {len(X_unseen_eval)}")
        
        # Match N_gen to eval pool size to avoid shape mismatch
        N_gen_matched = len(X_unseen_eval_t)
        
        # Denormalize eval data once
        eval_denorm = dm.inverse_transform(X_unseen_eval_t, data_name='all')
        
        # Loop over modes and K values (like seen domains)
        for mode in modes:
            for K in K_values:
                # no_prompt only runs once (K doesn't matter)
                if mode == 'no_prompt' and K != K_values[0]:
                    continue
                
                # Clamp K to available prompts
                K_actual = min(K, len(X_unseen_prompt))
                
                print(f"  Domain 3, Mode: {mode}, K={K_actual if mode != 'no_prompt' else 0}")
                
                # Select prompts
                seed_used = 42 + 3 * 100 + K_actual
                np.random.seed(seed_used)
                prompt_indices = np.random.choice(len(X_unseen_prompt), K_actual, replace=False)
                prompts = X_unseen_prompt[prompt_indices]
                
                # Generate samples based on mode
                if mode == 'no_prompt':
                    gen_norm = sample_no_prompt(model, N_gen_matched)
                    sample_diag = {'nan_inf_diag': check_nan_inf(gen_norm, "gen_norm")}
                elif mode == 'k_shot':
                    gen_norm, sample_diag = sample_k_shot(
                        model, prompts, N_gen_matched, 
                        debug=args.debug, 
                        eval_mask_fallback=args.eval_mask_fallback
                    )
                elif mode == 'shuffled':
                    gen_norm = sample_shuffled(model, prompts, N_gen_matched)
                    sample_diag = {'nan_inf_diag': check_nan_inf(gen_norm, "gen_norm")}
                elif mode == 'minus_pam':
                    gen_norm = sample_minus_pam(model, prompts, N_gen_matched)
                    sample_diag = {'nan_inf_diag': check_nan_inf(gen_norm, "gen_norm")}
                else:
                    print(f"    Unknown mode: {mode}, skipping")
                    continue
                
                # ========== FAIL-FAST: Check for NaN/Inf ==========
                nan_inf_diag = sample_diag['nan_inf_diag']
                if nan_inf_diag['has_nan'] or nan_inf_diag['has_inf']:
                    print(f"    [WARNING] NaN/Inf in generated samples! Skipping this mode.")
                    print(f"      NaN frac: {nan_inf_diag['nan_frac']:.6f}")
                    print(f"      Inf frac: {nan_inf_diag['inf_frac']:.6f}")
                    continue
                
                # Denormalize generated
                gen_denorm = dm.inverse_transform(gen_norm, data_name='all')
                
                # Check denormalized data too
                gen_denorm_diag = check_nan_inf(gen_denorm, "gen_denorm")
                if gen_denorm_diag['has_nan'] or gen_denorm_diag['has_inf']:
                    print(f"    [WARNING] NaN/Inf in denormalized generated data!")
                    print(f"      NaN frac: {gen_denorm_diag['nan_frac']:.6f}")
                
                # Compute metrics
                mmd = compute_mmd(eval_denorm, gen_denorm)
                kl_avg, kl_per_ch = compute_kl(eval_denorm, gen_denorm)
                mdd_avg, mdd_per_ch = compute_mdd(eval_denorm, gen_denorm)
                v_wass = compute_wasserstein(eval_denorm, gen_denorm, 2)
                a_wass = compute_wasserstein(eval_denorm, gen_denorm, 3)
                curv_wass = compute_wasserstein(eval_denorm, gen_denorm, 5)
                corr_frob = compute_corr_frob(eval_denorm, gen_denorm)
                
                # Sanity stats
                sanity = {}
                for ch_name, ch_idx in [('v', 2), ('a', 3), ('curvature', 5)]:
                    sanity.update(compute_stats(eval_denorm[..., ch_idx], f'real_{ch_name}'))
                    sanity.update(compute_stats(gen_denorm[..., ch_idx], f'gen_{ch_name}'))
                
                result = {
                    'domain': 3,
                    'mode': mode,
                    'K': K_actual if mode != 'no_prompt' else 0,
                    'N_gen': N_gen_matched,
                    'mmd': mmd,
                    'kl': kl_avg,
                    'mdd': mdd_avg,
                    'v_wass': v_wass,
                    'a_wass': a_wass,
                    'curvature_wass': curv_wass,
                    'corr_frob': corr_frob,
                    **sanity
                }
                all_results.append(result)
                
                print(f"    MMD={mmd:.4f}, KL={kl_avg:.4f}, MDD={mdd_avg:.4f}")
                print(f"    v_wass={v_wass:.4f}, a_wass={a_wass:.4f}, curv_wass={curv_wass:.4f}")
                print(f"    corr_frob={corr_frob:.4f}")
                
                # Store for diagnostics (last run)
                last_real_norm = X_unseen_eval_t
                last_gen_norm = gen_norm
                last_gen_denorm = gen_denorm
                last_real_denorm = eval_denorm
                last_domain_id = 3
                last_K = K_actual
    
    # Save results (merge with existing if present)
    summary_path = args.output_dir / 'summary.json'
    
    # Load existing results if file exists
    existing_results = []
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                existing_results = json.load(f)
            print(f"\nLoaded {len(existing_results)} existing results from {summary_path}")
        except Exception as e:
            print(f"\nWarning: Could not load existing summary.json: {e}")
            existing_results = []
    
    # Create a key for each result to identify duplicates
    def result_key(r):
        return (r.get('domain'), r.get('mode'), r.get('K'))
    
    # Build a dict of existing results
    existing_dict = {result_key(r): r for r in existing_results}
    
    # Update with new results (overwrite duplicates)
    for r in all_results:
        existing_dict[result_key(r)] = r
    
    # Convert back to list
    merged_results = list(existing_dict.values())
    
    # Sort by domain, mode, K for consistency
    merged_results.sort(key=lambda r: (r.get('domain', 0), r.get('mode', ''), r.get('K', 0)))
    
    with open(summary_path, 'w') as f:
        json.dump(merged_results, f, indent=2)
    print(f"Results saved to {summary_path} (total: {len(merged_results)} entries, new: {len(all_results)})")
    
    # Print summary table (show new results only)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (this run)")
    print("=" * 80)
    print(f"{'Domain':<8} {'Mode':<12} {'K':<4} {'MMD':<8} {'KL':<8} {'v_wass':<8} {'corr_frob':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['domain']:<8} {r['mode']:<12} {r['K']:<4} {r['mmd']:.4f}   {r['kl']:.4f}   {r['v_wass']:.4f}   {r['corr_frob']:.4f}")
    
    # ========================================
    # Save samples for UMAP visualization
    # ========================================
    if args.save_samples and last_gen_norm is not None and last_real_norm is not None:
        print("\n" + "=" * 80)
        print("SAVING SAMPLES FOR UMAP")
        print("=" * 80)
        
        samples_dir = args.output_dir / 'samples'
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Save normalized samples
        gen_norm_path = samples_dir / f'D{last_domain_id}_K{last_K}_gen_norm.npy'
        real_norm_path = samples_dir / f'D{last_domain_id}_K{last_K}_real_norm.npy'
        np.save(gen_norm_path, last_gen_norm)
        np.save(real_norm_path, last_real_norm)
        print(f"  Saved normalized: {gen_norm_path.name}, {real_norm_path.name}")
        
        # Save denormalized samples
        gen_denorm_path = samples_dir / f'D{last_domain_id}_K{last_K}_gen_denorm.npy'
        real_denorm_path = samples_dir / f'D{last_domain_id}_K{last_K}_real_denorm.npy'
        np.save(gen_denorm_path, last_gen_denorm)
        np.save(real_denorm_path, last_real_denorm)
        print(f"  Saved denormalized: {gen_denorm_path.name}, {real_denorm_path.name}")
        
        # Also save a convenient gen_samples.npy for UMAP script
        np.save(samples_dir / 'gen_samples.npy', last_gen_denorm)
        np.save(samples_dir / 'real_eval.npy', last_real_denorm)
        print(f"  Saved for UMAP: gen_samples.npy, real_eval.npy")
        
        # Save metadata
        samples_meta = {
            'domain': last_domain_id,
            'K': last_K,
            'gen_shape': list(last_gen_norm.shape),
            'real_shape': list(last_real_norm.shape),
            'normalizer_type': normalizer_type
        }
        with open(samples_dir / 'samples_meta.json', 'w') as f:
            json.dump(samples_meta, f, indent=2)
        print(f"  Saved metadata: samples_meta.json")
        print(f"\n  Output dir: {samples_dir}")
    
    # ========================================
    # TASK 1: PIT Diagnostics
    # ========================================
    if args.run_pit_diagnostics and last_gen_norm is not None and last_real_norm is not None:
        print("\n" + "=" * 80)
        print("TASK 1: PIT DIAGNOSTICS")
        print("=" * 80)
        
        diag_output_dir = args.output_dir / 'pit_diagnostics'
        
        run_pit_diagnostics(
            real_norm=last_real_norm,
            gen_norm=last_gen_norm,
            real_denorm=last_real_denorm,
            gen_denorm=last_gen_denorm,
            normalizer_type=normalizer_type,
            pit_params=pit_params,
            apply_clip=args.pit_clip_output,
            output_dir=diag_output_dir
        )
        
        # If clip was applied, re-run evaluation with clipped samples
        if args.pit_clip_output and normalizer_type == 'centered_pit':
            print("\n[pit_clip_output] Re-evaluating metrics with clipped normalized outputs...")
            gen_clipped = apply_pit_clip(last_gen_norm, clip_range=1.0)
            gen_clipped_denorm = dm.inverse_transform(gen_clipped, data_name='all')
            
            # Compute metrics on clipped samples
            mmd_clipped = compute_mmd(last_real_denorm, gen_clipped_denorm)
            kl_clipped, _ = compute_kl(last_real_denorm, gen_clipped_denorm)
            mdd_clipped, _ = compute_mdd(last_real_denorm, gen_clipped_denorm)
            corr_frob_clipped = compute_corr_frob(last_real_denorm, gen_clipped_denorm)
            
            print(f"\n[pit_clip_output] Metrics after clipping:")
            print(f"  MMD: {mmd_clipped:.4f}")
            print(f"  KL: {kl_clipped:.4f}")
            print(f"  MDD: {mdd_clipped:.4f}")
            print(f"  corr_frob: {corr_frob_clipped:.4f}")
    
    # ========================================
    # TASK 2: Correlation Heatmaps
    # ========================================
    if args.save_corr_heatmaps and last_gen_norm is not None and last_real_norm is not None:
        print("\n" + "=" * 80)
        print("TASK 2: CORRELATION HEATMAPS")
        print("=" * 80)
        
        heatmap_output_dir = args.output_dir / 'corr_heatmaps'
        prefix = f"D{last_domain_id}_K{last_K}_"
        
        heatmap_results = run_corr_heatmaps(
            real_norm=last_real_norm,
            gen_norm=last_gen_norm,
            normalizer_type=normalizer_type,
            output_dir=heatmap_output_dir,
            prefix=prefix,
            zscore_mean=zscore_mean,
            zscore_std=zscore_std,
            pit_params=pit_params,
            compute_lag1=args.compute_lag1,
            compute_noise_floor_flag=args.compute_noise_floor,
            n_bootstrap=args.n_bootstrap
        )
        
        if heatmap_results:
            print(f"\n[Heatmaps] Output saved to: {heatmap_output_dir}")
            print(f"  Files: {prefix}heatmap_real_lag0.png, {prefix}heatmap_gen_lag0.png, {prefix}heatmap_diff_lag0.png")
            if args.compute_lag1:
                print(f"         {prefix}heatmap_real_lag1.png, {prefix}heatmap_gen_lag1.png, {prefix}heatmap_diff_lag1.png")
            print(f"         {prefix}corr_heatmaps_summary.json")


if __name__ == '__main__':
    main()
