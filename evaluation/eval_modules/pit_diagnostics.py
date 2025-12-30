#!/usr/bin/env python3
"""
PIT (Probability Integral Transform) Diagnostics Module.

Task 1: Centered-PIT diagnosis
- Normalized-space range diagnostics (before denorm) for both real and gen
- PIT round-trip consistency check on real data
- Optional gated stabilization experiment (--pit_clip_output flag)
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


def compute_norm_space_diagnostics(data: np.ndarray, name: str = "data", 
                                   normalizer_type: str = "centered_pit") -> Dict:
    """
    Compute normalized-space diagnostics for data (before denorm).
    
    Args:
        data: array of shape (N, T, D) - normalized data
        name: identifier for this data (e.g., "real", "gen")
        normalizer_type: "centered_pit", "zscore", or "zscore_winsor"
        
    Returns:
        Dict with per-channel statistics
    """
    channel_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
    N, T, D = data.shape
    
    # Global NaN/Inf check
    total_values = data.size
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    nan_frac = nan_count / total_values
    inf_frac = inf_count / total_values
    
    stats = {
        'name': name,
        'normalizer_type': normalizer_type,
        'shape': [N, T, D],
        'nan_frac': float(nan_frac),
        'inf_frac': float(inf_frac),
        'has_nan': bool(nan_count > 0),  # Convert numpy.bool_ to Python bool
        'has_inf': bool(inf_count > 0),
        'per_channel': {}
    }
    
    # Flatten across N and T for per-channel stats
    data_flat = data.reshape(-1, D)  # (N*T, D)
    
    # Quantiles to report
    quantiles = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    
    for d in range(D):
        ch_data = data_flat[:, d]
        ch_name = channel_names[d] if d < len(channel_names) else f"ch{d}"
        
        # Per-channel NaN check
        ch_nan_frac = float(np.isnan(ch_data).mean())
        ch_inf_frac = float(np.isinf(ch_data).mean())
        
        # Use nanmin/nanmax to handle NaN gracefully
        ch_stats = {
            'min': float(np.nanmin(ch_data)) if not np.all(np.isnan(ch_data)) else float('nan'),
            'max': float(np.nanmax(ch_data)) if not np.all(np.isnan(ch_data)) else float('nan'),
            'mean': float(np.nanmean(ch_data)) if not np.all(np.isnan(ch_data)) else float('nan'),
            'std': float(np.nanstd(ch_data)) if not np.all(np.isnan(ch_data)) else float('nan'),
            'nan_frac': ch_nan_frac,
            'inf_frac': ch_inf_frac,
            'quantiles': {}
        }
        
        # Compute quantiles only on finite values
        finite_mask = np.isfinite(ch_data)
        if finite_mask.sum() > 0:
            ch_finite = ch_data[finite_mask]
            ch_stats['quantiles'] = {str(q): float(np.quantile(ch_finite, q)) for q in quantiles}
        else:
            ch_stats['quantiles'] = {str(q): float('nan') for q in quantiles}
        
        # For centered_pit, report fraction outside expected range
        if normalizer_type == 'centered_pit':
            finite_data = ch_data[finite_mask] if finite_mask.sum() > 0 else np.array([0])
            abs_data = np.abs(finite_data)
            ch_stats['p_out_gt_1.0'] = float(np.mean(abs_data > 1.0)) if len(finite_data) > 0 else float('nan')
            ch_stats['p_out_gt_1.2'] = float(np.mean(abs_data > 1.2)) if len(finite_data) > 0 else float('nan')
            ch_stats['p_out_gt_1.5'] = float(np.mean(abs_data > 1.5)) if len(finite_data) > 0 else float('nan')
            # Also report fraction at exact boundaries (-1 or 1)
            ch_stats['p_at_neg1'] = float(np.mean(finite_data <= -1.0)) if len(finite_data) > 0 else float('nan')
            ch_stats['p_at_pos1'] = float(np.mean(finite_data >= 1.0)) if len(finite_data) > 0 else float('nan')
        
        stats['per_channel'][ch_name] = ch_stats
    
    return stats


def compute_denorm_edge_stats(data_denorm: np.ndarray, pit_params: Dict, name: str = "data") -> Dict:
    """
    Compute denormalized-space edge statistics for centered_pit.
    
    Reports fraction of values at ECDF edge boundaries (min/max of training data).
    
    Args:
        data_denorm: array of shape (N, T, D) - denormalized data
        pit_params: PIT parameters with ecdf_x (edge values)
        name: identifier for this data
        
    Returns:
        Dict with per-channel edge statistics
    """
    channel_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
    N, T, D = data_denorm.shape
    
    stats = {
        'name': name,
        'shape': [N, T, D],
        'per_channel': {}
    }
    
    data_flat = data_denorm.reshape(-1, D)
    
    for d in range(D):
        ch_data = data_flat[:, d]
        ch_name = channel_names[d] if d < len(channel_names) else f"ch{d}"
        
        # Get ECDF edge values
        ecdf_x = np.array(pit_params['ecdf_x'][str(d)])
        edge_min = ecdf_x[0]
        edge_max = ecdf_x[-1]
        
        # Compute fraction at edges (with small tolerance for numerical precision)
        tol = (edge_max - edge_min) * 1e-6
        p_edge_min = float(np.mean(ch_data <= edge_min + tol))
        p_edge_max = float(np.mean(ch_data >= edge_max - tol))
        
        ch_stats = {
            'edge_min': float(edge_min),
            'edge_max': float(edge_max),
            'p_edge_min': p_edge_min,
            'p_edge_max': p_edge_max,
            'p_edge_total': p_edge_min + p_edge_max,
            'data_min': float(np.min(ch_data)),
            'data_max': float(np.max(ch_data)),
            'data_mean': float(np.mean(ch_data)),
            'data_std': float(np.std(ch_data))
        }
        
        stats['per_channel'][ch_name] = ch_stats
    
    return stats


def print_norm_diagnostics(real_stats: Dict, gen_stats: Dict, normalizer_type: str):
    """Print concise normalized-space diagnostics block."""
    channel_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
    
    print(f"\n[{normalizer_type}][NORM] Normalized-Space Diagnostics")
    print("=" * 90)
    
    # Global NaN/Inf warning
    if gen_stats.get('has_nan') or gen_stats.get('has_inf'):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"WARNING: Gen data has NaN={gen_stats['nan_frac']:.4f}, Inf={gen_stats['inf_frac']:.4f}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    # Header
    print(f"{'Channel':<12} {'Metric':<18} {'Real':>15} {'Gen':>15}")
    print("-" * 90)
    
    for ch_name in channel_names:
        if ch_name not in real_stats['per_channel']:
            continue
            
        r = real_stats['per_channel'][ch_name]
        g = gen_stats['per_channel'][ch_name]
        
        # NaN warning per channel
        if g.get('nan_frac', 0) > 0:
            print(f"{ch_name:<12} [NaN frac]         {r.get('nan_frac', 0):.4f}           {g['nan_frac']:.4f} <<<")
        
        # Basic stats (handle NaN gracefully)
        r_min = r['min'] if np.isfinite(r['min']) else float('nan')
        r_max = r['max'] if np.isfinite(r['max']) else float('nan')
        g_min = g['min'] if np.isfinite(g['min']) else float('nan')
        g_max = g['max'] if np.isfinite(g['max']) else float('nan')
        
        if np.isnan(g_min) or np.isnan(g_max):
            print(f"{ch_name:<12} min/max           [{r_min:+.3f},{r_max:+.3f}]  [NaN, NaN] <<<")
        else:
            print(f"{ch_name:<12} min/max           [{r_min:+.3f},{r_max:+.3f}]  [{g_min:+.3f},{g_max:+.3f}]")
        
        r_mean = r['mean'] if np.isfinite(r['mean']) else float('nan')
        r_std = r['std'] if np.isfinite(r['std']) else float('nan')
        g_mean = g['mean'] if np.isfinite(g['mean']) else float('nan')
        g_std = g['std'] if np.isfinite(g['std']) else float('nan')
        
        if np.isnan(g_mean):
            print(f"{'':<12} mean±std           {r_mean:+.4f}±{r_std:.3f}    NaN±NaN <<<")
        else:
            print(f"{'':<12} mean±std           {r_mean:+.4f}±{r_std:.3f}    {g_mean:+.4f}±{g_std:.3f}")
        
        # Out-of-range fractions for centered_pit
        if normalizer_type == 'centered_pit':
            r_p1 = r.get('p_out_gt_1.0', 0)
            g_p1 = g.get('p_out_gt_1.0', float('nan'))
            if not np.isnan(g_p1):
                print(f"{'':<12} p_out(|x|>1.0)     {r_p1:.4f}           {g_p1:.4f}")
                if g_p1 > 0.01:  # Only show if significant
                    g_p12 = g.get('p_out_gt_1.2', float('nan'))
                    print(f"{'':<12} p_out(|x|>1.2)     {r.get('p_out_gt_1.2', 0):.4f}           {g_p12:.4f}")
        
        print()


def print_denorm_edge_stats(real_edge_stats: Dict, gen_edge_stats: Dict):
    """Print denormalized-space edge statistics for centered_pit."""
    channel_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
    
    print(f"\n[centered_pit][DENORM] Edge Statistics (fraction at ECDF boundaries)")
    print("=" * 90)
    print(f"{'Channel':<12} {'Edge Range':<25} {'Real p_edge':>15} {'Gen p_edge':>15}")
    print("-" * 90)
    
    for ch_name in channel_names:
        if ch_name not in gen_edge_stats['per_channel']:
            continue
        
        r = real_edge_stats['per_channel'][ch_name]
        g = gen_edge_stats['per_channel'][ch_name]
        
        edge_range = f"[{g['edge_min']:.4f}, {g['edge_max']:.4f}]"
        print(f"{ch_name:<12} {edge_range:<25} {r['p_edge_total']:.4f}           {g['p_edge_total']:.4f}")
        print(f"{'':<12} {'  p_edge_min':<25} {r['p_edge_min']:.4f}           {g['p_edge_min']:.4f}")
        print(f"{'':<12} {'  p_edge_max':<25} {r['p_edge_max']:.4f}           {g['p_edge_max']:.4f}")


def pit_round_trip_check(X_norm: np.ndarray, 
                         ecdf_x: Dict, ecdf_y: Dict,
                         n_samples: int = 1000) -> Dict:
    """
    PIT round-trip consistency check on real normalized data.
    
    Steps:
    1. Take subset of normalized samples X_norm
    2. inverse_ecdf -> X_raw  
    3. forward_ecdf (centered_pit transform) -> X_norm_rt
    4. Compare X_norm_rt vs X_norm
    
    Args:
        X_norm: shape (N, T, D) - normalized real data in [-1, 1]
        ecdf_x: dict of channel -> sorted raw values (from fitting)
        ecdf_y: dict of channel -> CDF values [0, 1]
        n_samples: number of samples to use for check
        
    Returns:
        Dict with per-channel MAE and max_abs_error
    """
    channel_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
    
    N, T, D = X_norm.shape
    
    # Subsample
    if N > n_samples:
        idx = np.random.choice(N, n_samples, replace=False)
        X_subset = X_norm[idx]
    else:
        X_subset = X_norm
    
    # Flatten for processing
    X_flat = X_subset.reshape(-1, D)  # (n_samples * T, D)
    
    # Step 1: Inverse ECDF ([-1, 1] -> raw)
    # Uncentering: [-1, 1] -> [0, 1]
    X_cdf = (X_flat + 1) / 2
    X_cdf = np.clip(X_cdf, 0, 1)
    
    X_raw = np.zeros_like(X_flat)
    for d in range(D):
        # Inverse CDF: interpolate CDF values back to raw values
        X_raw[:, d] = np.interp(X_cdf[:, d], ecdf_y[str(d)], ecdf_x[str(d)])
    
    # Step 2: Forward ECDF (raw -> [0, 1] -> [-1, 1])
    X_cdf_rt = np.zeros_like(X_raw)
    for d in range(D):
        # Forward CDF: interpolate raw values to CDF values
        X_cdf_rt[:, d] = np.interp(X_raw[:, d], ecdf_x[str(d)], ecdf_y[str(d)])
    
    # Center to [-1, 1]
    X_norm_rt = X_cdf_rt * 2 - 1
    
    # Step 3: Compare
    errors = np.abs(X_norm_rt - X_flat)
    
    results = {
        'n_samples': len(X_subset),
        'n_values': len(X_flat),
        'per_channel': {},
        'overall_mae': float(np.mean(errors)),
        'overall_max_error': float(np.max(errors))
    }
    
    for d in range(D):
        ch_name = channel_names[d] if d < len(channel_names) else f"ch{d}"
        ch_errors = errors[:, d]
        results['per_channel'][ch_name] = {
            'mae': float(np.mean(ch_errors)),
            'max_abs_error': float(np.max(ch_errors)),
            'std_error': float(np.std(ch_errors))
        }
    
    return results


def print_round_trip_results(results: Dict):
    """Print round-trip consistency check results."""
    print("\n[centered_pit][ROUND-TRIP] Consistency Check")
    print("=" * 60)
    print(f"Samples: {results['n_samples']}, Values: {results['n_values']}")
    print(f"Overall MAE: {results['overall_mae']:.6f}")
    print(f"Overall Max Error: {results['overall_max_error']:.6f}")
    print()
    print(f"{'Channel':<12} {'MAE':<15} {'Max Abs Error':<15}")
    print("-" * 45)
    
    for ch_name, ch_results in results['per_channel'].items():
        print(f"{ch_name:<12} {ch_results['mae']:.6f}         {ch_results['max_abs_error']:.6f}")


def apply_pit_clip(gen_norm: np.ndarray, clip_range: float = 1.0) -> np.ndarray:
    """
    Clamp generated normalized outputs to [-clip_range, clip_range].
    
    This is a stabilization experiment for centered_pit only.
    
    Args:
        gen_norm: shape (N, T, D) - generated normalized data
        clip_range: clipping range (default 1.0 for centered_pit)
        
    Returns:
        Clipped data
    """
    return np.clip(gen_norm, -clip_range, clip_range)


def compute_severity_stats(gen_stats: Dict) -> Dict:
    """Compute out-of-range severity statistics."""
    severity = {
        'channels_with_gt_1.0': [],
        'channels_with_gt_1.2': [],
        'max_p_out_gt_1.0': 0.0,
        'max_p_out_gt_1.2': 0.0
    }
    
    for ch_name, ch_stats in gen_stats['per_channel'].items():
        if 'p_out_gt_1.0' in ch_stats:
            if ch_stats['p_out_gt_1.0'] > 0.01:  # >1% threshold
                severity['channels_with_gt_1.0'].append(ch_name)
            if ch_stats['p_out_gt_1.2'] > 0.001:  # >0.1% threshold
                severity['channels_with_gt_1.2'].append(ch_name)
            severity['max_p_out_gt_1.0'] = max(severity['max_p_out_gt_1.0'], ch_stats['p_out_gt_1.0'])
            severity['max_p_out_gt_1.2'] = max(severity['max_p_out_gt_1.2'], ch_stats['p_out_gt_1.2'])
    
    return severity


def print_diagnosis_summary(real_stats: Dict, gen_stats: Dict, 
                            round_trip_results: Optional[Dict],
                            normalizer_type: str):
    """Print diagnosis summary with recommendations."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    # 0. Check for NaN/Inf first - this is fatal
    gen_has_nan = gen_stats.get('has_nan', False)
    gen_has_inf = gen_stats.get('has_inf', False)
    
    if gen_has_nan or gen_has_inf:
        print("\n!!! FATAL: Generated data contains NaN/Inf !!!")
        print(f"   NaN frac: {gen_stats.get('nan_frac', 0):.6f}")
        print(f"   Inf frac: {gen_stats.get('inf_frac', 0):.6f}")
        print("\n   >>> Debug with --debug --eval_mask_fallback to check if PAM masking is the cause.")
        print("   >>> Do NOT trust any other diagnostics until NaN is fixed.")
        return
    
    # 1. Real norm in-range?
    real_in_range = True
    for ch_name, ch_stats in real_stats['per_channel'].items():
        ch_min = ch_stats['min']
        ch_max = ch_stats['max']
        if np.isfinite(ch_min) and np.isfinite(ch_max):
            if abs(ch_min) > 1.5 or abs(ch_max) > 1.5:
                real_in_range = False
                break
    print(f"\n1. Real normalized data in expected range: {'YES' if real_in_range else 'NO'}")
    
    # 2. Round-trip errors (centered_pit only)
    if round_trip_results is not None:
        print(f"\n2. Round-trip errors:")
        print(f"   Overall MAE: {round_trip_results['overall_mae']:.6f}")
        print(f"   Overall Max: {round_trip_results['overall_max_error']:.6f}")
        rt_ok = round_trip_results['overall_mae'] < 0.01 and round_trip_results['overall_max_error'] < 0.1
        print(f"   Status: {'OK' if rt_ok else 'POTENTIAL ISSUE'}")
    else:
        print(f"\n2. Round-trip errors: N/A (not centered_pit)")
        rt_ok = True
    
    # 3. Gen norm out-of-range severity
    if normalizer_type == 'centered_pit':
        severity = compute_severity_stats(gen_stats)
        print(f"\n3. Generated data out-of-range severity:")
        print(f"   Max p_out(|x|>1.0): {severity['max_p_out_gt_1.0']:.4f}")
        print(f"   Max p_out(|x|>1.2): {severity['max_p_out_gt_1.2']:.4f}")
        if severity['channels_with_gt_1.0']:
            print(f"   Channels >1% at |x|>1.0: {severity['channels_with_gt_1.0']}")
        
        gen_severe = severity['max_p_out_gt_1.0'] > 0.05 or severity['max_p_out_gt_1.2'] > 0.01
    else:
        gen_severe = False
        print(f"\n3. Generated data out-of-range: N/A (not centered_pit)")
    
    # 4. Recommendation
    print(f"\n4. RECOMMENDATION:")
    if not rt_ok:
        print("   >>> FIX PREPROCESS + RETRAIN: Round-trip errors indicate preprocessing/inverse mismatch.")
    elif gen_severe:
        print("   >>> SAMPLING DRIFT detected. Try --pit_clip_output flag.")
        print("       If clipping helps, consider retraining with better sampling constraints.")
    else:
        print("   >>> No major issues detected. Diagnostics look healthy.")


def run_pit_diagnostics(
    real_norm: np.ndarray,
    gen_norm: np.ndarray,
    normalizer_type: str,
    pit_params: Optional[Dict] = None,
    apply_clip: bool = False,
    output_dir: Optional[Path] = None,
    real_denorm: Optional[np.ndarray] = None,
    gen_denorm: Optional[np.ndarray] = None
) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    Run full PIT diagnostics pipeline.
    
    Args:
        real_norm: (N, T, D) normalized real data
        gen_norm: (N, T, D) normalized generated data
        normalizer_type: "centered_pit", "zscore", or "zscore_winsor"
        pit_params: PIT parameters (required for round-trip check and edge stats)
        apply_clip: if True, apply pit_clip_output to gen_norm
        output_dir: directory to save diagnostics JSON
        real_denorm: (N, T, D) denormalized real data (optional, for edge stats)
        gen_denorm: (N, T, D) denormalized gen data (optional, for edge stats)
        
    Returns:
        (real_stats, gen_stats, round_trip_results)
    """
    # Optional clipping for stabilization experiment
    if apply_clip and normalizer_type == 'centered_pit':
        print("\n[pit_clip_output] Applying clamp to [-1, 1] on generated normalized outputs...")
        gen_norm = apply_pit_clip(gen_norm, clip_range=1.0)
    
    # 1. Normalized-space diagnostics
    real_stats = compute_norm_space_diagnostics(real_norm, "real", normalizer_type)
    gen_stats = compute_norm_space_diagnostics(gen_norm, "gen", normalizer_type)
    
    print_norm_diagnostics(real_stats, gen_stats, normalizer_type)
    
    # 2. Denorm edge statistics (centered_pit only)
    real_edge_stats = None
    gen_edge_stats = None
    if normalizer_type == 'centered_pit' and pit_params is not None:
        if real_denorm is not None and gen_denorm is not None:
            real_edge_stats = compute_denorm_edge_stats(real_denorm, pit_params, "real")
            gen_edge_stats = compute_denorm_edge_stats(gen_denorm, pit_params, "gen")
            print_denorm_edge_stats(real_edge_stats, gen_edge_stats)
    
    # 3. Round-trip check (centered_pit only)
    round_trip_results = None
    if normalizer_type == 'centered_pit' and pit_params is not None:
        ecdf_x = {k: np.array(v) for k, v in pit_params['ecdf_x'].items()}
        ecdf_y = {k: np.array(v) for k, v in pit_params['ecdf_y'].items()}
        
        round_trip_results = pit_round_trip_check(real_norm, ecdf_x, ecdf_y)
        print_round_trip_results(round_trip_results)
    
    # 4. Summary
    print_diagnosis_summary(real_stats, gen_stats, round_trip_results, normalizer_type)
    
    # Save to file if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        diag_file = output_dir / f'pit_diagnostics_{normalizer_type}.json'
        diag_data = {
            'real_stats': real_stats,
            'gen_stats': gen_stats,
            'real_edge_stats': real_edge_stats,
            'gen_edge_stats': gen_edge_stats,
            'round_trip': round_trip_results,
            'normalizer_type': normalizer_type,
            'pit_clip_applied': apply_clip
        }
        
        with open(diag_file, 'w') as f:
            json.dump(diag_data, f, indent=2)
        print(f"\n[Diagnostics] Saved to {diag_file}")
    
    return real_stats, gen_stats, round_trip_results


class CenteredPITInverter:
    """
    Helper class for inverse-ECDF on centered_pit normalized data.
    
    Used for denormalization in heatmap generation (Task 2).
    """
    
    def __init__(self, pit_params: Dict):
        """
        Args:
            pit_params: Dict with 'ecdf_x' and 'ecdf_y' per channel
        """
        self.ecdf_x = {int(k): np.array(v) for k, v in pit_params['ecdf_x'].items()}
        self.ecdf_y = {int(k): np.array(v) for k, v in pit_params['ecdf_y'].items()}
        self.n_channels = len(self.ecdf_x)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform from [-1, 1] back to original scale.
        
        Args:
            data: shape (N, T, D) normalized data
            
        Returns:
            Denormalized data in original scale
        """
        original_shape = data.shape
        if data.ndim == 3:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data.copy()
        
        # Uncentering: from [-1, 1] to [0, 1]
        data_cdf = (data_flat + 1) / 2
        data_cdf = np.clip(data_cdf, 0, 1)
        
        result = np.zeros_like(data_flat)
        
        for d in range(data_flat.shape[1]):
            # Inverse CDF lookup
            result[:, d] = np.interp(data_cdf[:, d], self.ecdf_y[d], self.ecdf_x[d])
        
        return result.reshape(original_shape)

