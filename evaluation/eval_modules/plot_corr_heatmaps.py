#!/usr/bin/env python3
"""
Correlation Heatmap Generation Module.

Task 2: Generate correlation heatmaps for REAL, GEN, and DIFF
- lag0: same-time correlation (D×D)
- lag1: cross-correlation between t and t+1 (D×D)

Key features:
- Uses masked Spearman correlation (rank-based, robust to monotonic transforms)
- Computes in PHYSICAL SPACE (denormalized)
- Proper per-timestep aggregation with masking
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json


# Channel names for axis labels
CHANNEL_NAMES = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']


def _spearman_corr_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute Spearman (rank) correlation matrix for 2D array X of shape (N, D).
    
    Spearman correlation is invariant to monotonic per-channel transforms,
    making it fair across different normalization schemes.
    
    Args:
        X: array of shape (N, D) where N is samples, D is channels
        
    Returns:
        C: array of shape (D, D) - Spearman correlation matrix
    """
    N, D = X.shape
    
    # Convert each column to ranks
    ranks = np.zeros_like(X, dtype=np.float64)
    for j in range(D):
        ranks[:, j] = rankdata(X[:, j], method='average')
    
    # Compute Pearson correlation on ranks = Spearman correlation
    C = np.corrcoef(ranks.T)
    return C


def _spearman_cross_corr_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute Spearman cross-correlation matrix between X and Y.
    
    Args:
        X: array of shape (N, D1)
        Y: array of shape (N, D2)
        
    Returns:
        C: array of shape (D1, D2) where C[i,j] = spearman_corr(X[:,i], Y[:,j])
    """
    N, D1 = X.shape
    D2 = Y.shape[1]
    
    # Rank transform
    ranks_X = np.zeros_like(X, dtype=np.float64)
    ranks_Y = np.zeros_like(Y, dtype=np.float64)
    for j in range(D1):
        ranks_X[:, j] = rankdata(X[:, j], method='average')
    for j in range(D2):
        ranks_Y[:, j] = rankdata(Y[:, j], method='average')
    
    # Standardize
    ranks_X = (ranks_X - ranks_X.mean(axis=0)) / (ranks_X.std(axis=0) + 1e-10)
    ranks_Y = (ranks_Y - ranks_Y.mean(axis=0)) / (ranks_Y.std(axis=0) + 1e-10)
    
    # Cross-correlation matrix
    C = (ranks_X.T @ ranks_Y) / N
    return C


def compute_lag0_corr_matrix(data: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lag-0 (same-time) Spearman correlation matrix averaged over timesteps.
    
    For each timestep t:
    1. Compute correlation matrix on (N, D) samples
    2. Build validity mask based on std > eps
    3. Accumulate with proper masking
    
    Args:
        data: array of shape (N, T, D) - denormalized data
        eps: threshold for near-constant detection
        
    Returns:
        C_avg: (D, D) averaged correlation matrix
        count_matrix: (D, D) count of valid timesteps per entry
    """
    N, T, D = data.shape
    
    # Accumulators
    C_sum = np.zeros((D, D), dtype=np.float64)
    count_matrix = np.zeros((D, D), dtype=np.float64)
    
    for t in range(T):
        data_t = data[:, t, :]  # (N, D)
        
        # Compute per-channel std
        std_t = np.std(data_t, axis=0)  # (D,)
        
        # Valid channels: std > eps
        valid_mask = std_t > eps  # (D,)
        n_valid = valid_mask.sum()
        
        # Need at least 2 valid channels for meaningful correlation
        if n_valid < 2:
            continue
        
        # Compute full correlation matrix
        C_t = _spearman_corr_matrix(data_t)  # (D, D)
        
        # Create entry mask: both i and j must be valid
        entry_mask = np.outer(valid_mask, valid_mask).astype(np.float64)  # (D, D)
        
        # Handle potential NaN in correlation (shouldn't happen with valid mask, but be safe)
        C_t = np.nan_to_num(C_t, nan=0.0)
        
        # Accumulate
        C_sum += C_t * entry_mask
        count_matrix += entry_mask
    
    # Average with tiny epsilon to avoid division by zero
    C_avg = C_sum / (count_matrix + 1e-10)
    
    # Set entries with zero counts to NaN for clarity
    C_avg[count_matrix == 0] = np.nan
    
    return C_avg, count_matrix


def compute_lag1_cross_corr_matrix(data: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lag-1 cross-correlation matrix: C[i,j] = corr(x_t^i, x_{t+1}^j)
    
    For each valid timestep pair (t, t+1):
    1. Compute cross-correlation between X=data[:,t,:] and Y=data[:,t+1,:]
    2. Build validity mask based on std > eps
    3. Accumulate with proper masking
    
    Args:
        data: array of shape (N, T, D) - denormalized data
        eps: threshold for near-constant detection
        
    Returns:
        C_avg: (D, D) averaged cross-correlation matrix
        count_matrix: (D, D) count of valid timestep pairs per entry
    """
    N, T, D = data.shape
    
    # Accumulators
    C_sum = np.zeros((D, D), dtype=np.float64)
    count_matrix = np.zeros((D, D), dtype=np.float64)
    
    for t in range(T - 1):
        X = data[:, t, :]      # (N, D)
        Y = data[:, t + 1, :]  # (N, D)
        
        # Compute per-channel std
        std_X = np.std(X, axis=0)  # (D,)
        std_Y = np.std(Y, axis=0)  # (D,)
        
        # Valid channels
        valid_i = std_X > eps  # for X dimension
        valid_j = std_Y > eps  # for Y dimension
        
        # Need at least 1 valid in each dimension
        if valid_i.sum() < 1 or valid_j.sum() < 1:
            continue
        
        # Compute full cross-correlation matrix
        C_t = _spearman_cross_corr_matrix(X, Y)  # (D, D)
        
        # Create entry mask: i valid in X, j valid in Y
        entry_mask = np.outer(valid_i, valid_j).astype(np.float64)  # (D, D)
        
        # Handle potential NaN
        C_t = np.nan_to_num(C_t, nan=0.0)
        
        # Accumulate
        C_sum += C_t * entry_mask
        count_matrix += entry_mask
    
    # Average
    C_avg = C_sum / (count_matrix + 1e-10)
    C_avg[count_matrix == 0] = np.nan
    
    return C_avg, count_matrix


def plot_corr_heatmap(
    C: np.ndarray,
    title: str,
    output_path: Path,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = 'RdBu_r',
    channel_names: List[str] = None
):
    """
    Plot and save a correlation heatmap.
    
    Args:
        C: (D, D) correlation matrix
        title: plot title
        output_path: path to save the figure
        vmin, vmax: color scale range
        cmap: colormap name
        channel_names: axis labels
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    D = C.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use masked array to handle NaN properly
    C_masked = np.ma.masked_invalid(C)
    
    im = ax.imshow(C_masked, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=10)
    
    # Labels
    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(channel_names[:D], fontsize=9)
    ax.set_yticklabels(channel_names[:D], fontsize=9)
    
    # Add values as text
    for i in range(D):
        for j in range(D):
            val = C[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=8, color=text_color)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Channel', fontsize=10)
    ax.set_ylabel('Channel', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def find_top_diff_pairs(
    C_real: np.ndarray, 
    C_gen: np.ndarray,
    top_k: int = 5,
    exclude_diagonal: bool = True
) -> List[Dict]:
    """
    Find top-k pairs with largest absolute difference (off-diagonal only).
    
    Args:
        C_real: (D, D) real correlation matrix
        C_gen: (D, D) generated correlation matrix
        top_k: number of top pairs to return
        exclude_diagonal: if True, exclude diagonal entries
        
    Returns:
        List of dicts with (i, j, real_corr, gen_corr, diff)
    """
    D = C_real.shape[0]
    diff = C_gen - C_real
    
    pairs = []
    for i in range(D):
        for j in range(D):
            if exclude_diagonal and i == j:
                continue
            if np.isnan(C_real[i, j]) or np.isnan(C_gen[i, j]):
                continue
            
            pairs.append({
                'i': i,
                'j': j,
                'i_name': CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f'ch{i}',
                'j_name': CHANNEL_NAMES[j] if j < len(CHANNEL_NAMES) else f'ch{j}',
                'real_corr': float(C_real[i, j]),
                'gen_corr': float(C_gen[i, j]),
                'diff': float(diff[i, j]),
                'abs_diff': float(abs(diff[i, j]))
            })
    
    # Sort by absolute difference
    pairs.sort(key=lambda x: x['abs_diff'], reverse=True)
    
    return pairs[:top_k]


def print_top_diffs(pairs: List[Dict], lag_name: str = "lag0"):
    """Print top difference pairs."""
    print(f"\n  Top-5 largest absolute diffs ({lag_name}, off-diagonal):")
    print(f"  {'(i,j)':<20} {'real_corr':>12} {'gen_corr':>12} {'diff':>12}")
    print("  " + "-" * 60)
    
    for p in pairs:
        pair_str = f"({p['i_name']}, {p['j_name']})"
        print(f"  {pair_str:<20} {p['real_corr']:>12.4f} {p['gen_corr']:>12.4f} {p['diff']:>+12.4f}")


def generate_corr_heatmaps(
    real_denorm: np.ndarray,
    gen_denorm: np.ndarray,
    output_dir: Path,
    prefix: str = "",
    compute_lag1: bool = True,
    eps: float = 1e-6
) -> Dict:
    """
    Generate correlation heatmaps for real, gen, and diff.
    
    Args:
        real_denorm: (N, T, D) denormalized real data
        gen_denorm: (N, T, D) denormalized generated data
        output_dir: directory to save heatmaps
        prefix: prefix for filenames (e.g., "D3_K16_")
        compute_lag1: whether to compute lag-1 cross-correlations
        eps: threshold for near-constant detection
        
    Returns:
        Dict with correlation matrices and top diffs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # === Lag-0 correlations ===
    print("\n[Heatmaps] Computing lag-0 correlations...")
    
    C_real_lag0, count_real_lag0 = compute_lag0_corr_matrix(real_denorm, eps=eps)
    C_gen_lag0, count_gen_lag0 = compute_lag0_corr_matrix(gen_denorm, eps=eps)
    C_diff_lag0 = C_gen_lag0 - C_real_lag0
    
    # Plot lag-0 heatmaps
    plot_corr_heatmap(C_real_lag0, "Real Correlation (lag-0)", 
                      output_dir / f"{prefix}heatmap_real_lag0.png")
    plot_corr_heatmap(C_gen_lag0, "Generated Correlation (lag-0)", 
                      output_dir / f"{prefix}heatmap_gen_lag0.png")
    plot_corr_heatmap(C_diff_lag0, "Difference (Gen - Real, lag-0)", 
                      output_dir / f"{prefix}heatmap_diff_lag0.png",
                      vmin=-1.0, vmax=1.0)
    
    # Top diffs for lag-0
    top_diffs_lag0 = find_top_diff_pairs(C_real_lag0, C_gen_lag0)
    print_top_diffs(top_diffs_lag0, "lag0")
    
    results['lag0'] = {
        'C_real': C_real_lag0.tolist(),
        'C_gen': C_gen_lag0.tolist(),
        'C_diff': C_diff_lag0.tolist(),
        'top_diffs': top_diffs_lag0,
        'count_matrix_real': count_real_lag0.tolist(),
        'count_matrix_gen': count_gen_lag0.tolist()
    }
    
    # === Lag-1 correlations (optional) ===
    if compute_lag1:
        print("\n[Heatmaps] Computing lag-1 cross-correlations...")
        
        C_real_lag1, count_real_lag1 = compute_lag1_cross_corr_matrix(real_denorm, eps=eps)
        C_gen_lag1, count_gen_lag1 = compute_lag1_cross_corr_matrix(gen_denorm, eps=eps)
        C_diff_lag1 = C_gen_lag1 - C_real_lag1
        
        # Plot lag-1 heatmaps
        plot_corr_heatmap(C_real_lag1, "Real Cross-Correlation (lag-1)", 
                          output_dir / f"{prefix}heatmap_real_lag1.png")
        plot_corr_heatmap(C_gen_lag1, "Generated Cross-Correlation (lag-1)", 
                          output_dir / f"{prefix}heatmap_gen_lag1.png")
        plot_corr_heatmap(C_diff_lag1, "Difference (Gen - Real, lag-1)", 
                          output_dir / f"{prefix}heatmap_diff_lag1.png",
                          vmin=-1.0, vmax=1.0)
        
        # Top diffs for lag-1
        top_diffs_lag1 = find_top_diff_pairs(C_real_lag1, C_gen_lag1)
        print_top_diffs(top_diffs_lag1, "lag1")
        
        results['lag1'] = {
            'C_real': C_real_lag1.tolist(),
            'C_gen': C_gen_lag1.tolist(),
            'C_diff': C_diff_lag1.tolist(),
            'top_diffs': top_diffs_lag1,
            'count_matrix_real': count_real_lag1.tolist(),
            'count_matrix_gen': count_gen_lag1.tolist()
        }
    
    # Save summary JSON
    summary_path = output_dir / f"{prefix}corr_heatmaps_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    
    return results


def compute_noise_floor(
    data: np.ndarray,
    n_bootstrap: int = 50,
    eps: float = 1e-6,
    compute_lag1: bool = True,
    seed: int = 42
) -> Dict:
    """
    Estimate the noise floor of correlation metrics via bootstrap.
    
    Splits real data into two halves (or bootstrap samples) repeatedly,
    computes correlation matrices for each half, and measures |C_A - C_B|.
    This gives the "intrinsic noise" due to finite sample size.
    
    Args:
        data: (N, T, D) denormalized real data
        n_bootstrap: number of bootstrap iterations
        eps: threshold for near-constant detection
        compute_lag1: whether to compute lag-1 as well
        seed: random seed for reproducibility
        
    Returns:
        Dict with noise floor statistics:
        - lag0_abs_diff: all |C_A - C_B| values (flattened off-diagonal)
        - lag0_stats: {mean, median, p90, p95, max}
        - lag1_abs_diff, lag1_stats: same for lag-1 (if computed)
    """
    rng = np.random.RandomState(seed)
    N = data.shape[0]
    half_n = N // 2
    
    if half_n < 5:
        print(f"  WARNING: Too few samples ({N}) for bootstrap noise floor estimation")
        return {'error': 'insufficient_samples', 'N': N}
    
    lag0_diffs = []
    lag1_diffs = []
    
    print(f"  [Noise Floor] Running {n_bootstrap} bootstrap iterations (N={N}, half={half_n})...")
    
    for b in range(n_bootstrap):
        # Randomly split into two halves
        idx = rng.permutation(N)
        idx_A = idx[:half_n]
        idx_B = idx[half_n:2*half_n]
        
        data_A = data[idx_A]
        data_B = data[idx_B]
        
        # Compute lag-0 correlations
        C_A_lag0, _ = compute_lag0_corr_matrix(data_A, eps=eps)
        C_B_lag0, _ = compute_lag0_corr_matrix(data_B, eps=eps)
        
        # |C_A - C_B| for off-diagonal elements
        diff_lag0 = np.abs(C_A_lag0 - C_B_lag0)
        D = diff_lag0.shape[0]
        
        # Extract off-diagonal
        for i in range(D):
            for j in range(D):
                if i != j and not np.isnan(diff_lag0[i, j]):
                    lag0_diffs.append(diff_lag0[i, j])
        
        # Compute lag-1 if requested
        if compute_lag1:
            C_A_lag1, _ = compute_lag1_cross_corr_matrix(data_A, eps=eps)
            C_B_lag1, _ = compute_lag1_cross_corr_matrix(data_B, eps=eps)
            
            diff_lag1 = np.abs(C_A_lag1 - C_B_lag1)
            for i in range(D):
                for j in range(D):
                    if not np.isnan(diff_lag1[i, j]):
                        lag1_diffs.append(diff_lag1[i, j])
    
    # Compute statistics
    def compute_stats(diffs):
        if len(diffs) == 0:
            return {'mean': np.nan, 'median': np.nan, 'p90': np.nan, 'p95': np.nan, 'max': np.nan}
        arr = np.array(diffs)
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
            'max': float(np.max(arr)),
            'n_samples': len(arr)
        }
    
    results = {
        'n_bootstrap': n_bootstrap,
        'half_n': half_n,
        'lag0_stats': compute_stats(lag0_diffs)
    }
    
    if compute_lag1:
        results['lag1_stats'] = compute_stats(lag1_diffs)
    
    # Print summary
    print(f"\n  [Noise Floor] Lag-0 |C_A - C_B| statistics (off-diagonal):")
    stats = results['lag0_stats']
    print(f"    mean={stats['mean']:.4f}, median={stats['median']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}, max={stats['max']:.4f}")
    
    if compute_lag1:
        print(f"  [Noise Floor] Lag-1 |C_A - C_B| statistics:")
        stats = results['lag1_stats']
        print(f"    mean={stats['mean']:.4f}, median={stats['median']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}, max={stats['max']:.4f}")
    
    return results


def run_corr_heatmaps(
    real_norm: np.ndarray,
    gen_norm: np.ndarray,
    normalizer_type: str,
    output_dir: Path,
    prefix: str = "",
    zscore_mean: Optional[np.ndarray] = None,
    zscore_std: Optional[np.ndarray] = None,
    pit_params: Optional[Dict] = None,
    compute_lag1: bool = True,
    eps: float = 1e-6,
    compute_noise_floor_flag: bool = False,
    n_bootstrap: int = 50
) -> Optional[Dict]:
    """
    Main entry point for correlation heatmap generation.
    
    Handles denormalization based on normalizer type, then generates heatmaps.
    
    Args:
        real_norm: (N, T, D) normalized real data
        gen_norm: (N, T, D) normalized generated data
        normalizer_type: "zscore", "zscore_winsor", or "centered_pit"
        output_dir: directory to save heatmaps
        prefix: prefix for filenames
        zscore_mean, zscore_std: for zscore/zscore_winsor denorm
        pit_params: for centered_pit inverse ECDF
        compute_lag1: whether to compute lag-1 correlations
        eps: threshold for near-constant detection
        compute_noise_floor_flag: whether to compute bootstrap noise floor
        n_bootstrap: number of bootstrap iterations for noise floor
        
    Returns:
        Dict with correlation results, or None if denorm not possible
    """
    print(f"\n[Heatmaps] Generating correlation heatmaps (normalizer: {normalizer_type})")
    
    # Denormalize to physical space
    if normalizer_type in ['zscore', 'zscore_winsor']:
        if zscore_mean is None or zscore_std is None:
            print("  WARNING: zscore_mean/std not provided, skipping heatmaps")
            return None
        
        real_denorm = real_norm * zscore_std + zscore_mean
        gen_denorm = gen_norm * zscore_std + zscore_mean
        
    elif normalizer_type == 'centered_pit':
        if pit_params is None:
            print("  WARNING: pit_params not provided, cannot inverse-ECDF. Skipping heatmaps.")
            return None
        
        from evaluation.eval_modules.pit_diagnostics import CenteredPITInverter
        inverter = CenteredPITInverter(pit_params)
        real_denorm = inverter.inverse_transform(real_norm)
        gen_denorm = inverter.inverse_transform(gen_norm)
        
    else:
        print(f"  WARNING: Unknown normalizer type '{normalizer_type}', skipping heatmaps")
        return None
    
    # Generate heatmaps
    results = generate_corr_heatmaps(
        real_denorm, gen_denorm, output_dir, prefix,
        compute_lag1=compute_lag1, eps=eps
    )
    
    # Compute noise floor if requested
    if compute_noise_floor_flag:
        print("\n[Heatmaps] Computing noise floor via bootstrap...")
        noise_floor = compute_noise_floor(
            real_denorm, 
            n_bootstrap=n_bootstrap,
            eps=eps,
            compute_lag1=compute_lag1
        )
        results['noise_floor'] = noise_floor
        
        # Save updated summary with noise floor
        output_dir = Path(output_dir)
        summary_path = output_dir / f"{prefix}corr_heatmaps_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Updated summary with noise floor: {summary_path}")
    
    return results

