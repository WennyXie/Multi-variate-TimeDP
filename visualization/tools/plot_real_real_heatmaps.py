#!/usr/bin/env python3
"""
Generate Real-Real correlation heatmaps to demonstrate sampling noise floor.

This script:
1. Splits real data into two halves (A and B)
2. Computes correlation matrices for each half
3. Plots C_realA, C_realB, and |C_A - C_B| as heatmaps
4. Shows that correlation differences are largely due to finite sample size

Usage:
    python tools/plot_real_real_heatmaps.py \
        --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
        --out_dir results/noise_floor
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path to import from evaluation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from evaluation.eval_modules.plot_corr_heatmaps import compute_lag0_corr_matrix, CHANNEL_NAMES


def plot_corr_heatmap(
    C: np.ndarray,
    title: str,
    output_path: Path,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = 'RdBu_r',
    channel_names: List[str] = None
):
    """Plot correlation heatmap (same style as plot_corr_heatmaps.py)."""
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


def plot_diff_heatmap(
    C_diff: np.ndarray,
    title: str,
    output_path: Path,
    vmax: float = 0.3,
    channel_names: List[str] = None
):
    """Plot absolute difference heatmap."""
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    D = C_diff.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    C_masked = np.ma.masked_invalid(C_diff)
    im = ax.imshow(C_masked, cmap='Oranges', vmin=0, vmax=vmax, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Correlation Difference|', fontsize=10)
    
    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(channel_names[:D], fontsize=9)
    ax.set_yticklabels(channel_names[:D], fontsize=9)
    
    for i in range(D):
        for j in range(D):
            val = C_diff[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=8, color=text_color)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Channel', fontsize=10)
    ax.set_ylabel('Channel', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Real-Real correlation heatmaps')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / 'noise_floor_heatmaps'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Real-Real Noise Floor] Data dir: {data_dir}")
    print(f"[Real-Real Noise Floor] Output dir: {out_dir}")
    
    # Load real data
    unseen_path = data_dir / 'unseen_eval_pool.npy'
    if not unseen_path.exists():
        print(f"ERROR: {unseen_path} not found!")
        return
    
    real_data = np.load(unseen_path)
    print(f"[Real-Real Noise Floor] Loaded real data: {real_data.shape}")
    
    # Check and fix shape: expect (N, T, D) but data might be (N, D, T)
    if real_data.shape[1] == 6 and real_data.shape[2] == 32:
        print("  Detected (N, D, T) format, transposing to (N, T, D)...")
        real_data = real_data.transpose(0, 2, 1)  # (N, D, T) -> (N, T, D)
    
    N, T, D = real_data.shape
    print(f"  Final shape: (N={N}, T={T}, D={D})")
    
    if N < 10:
        print(f"ERROR: Too few samples ({N}) for meaningful split")
        return
    
    rng = np.random.RandomState(args.seed)
    half_n = N // 2
    
    print(f"[Real-Real Noise Floor] Splitting {N} samples into two halves of {half_n}")
    
    # Perform split
    idx = rng.permutation(N)
    idx_A = idx[:half_n]
    idx_B = idx[half_n:2*half_n]
    
    data_A = real_data[idx_A]
    data_B = real_data[idx_B]
    
    print(f"  Group A: {data_A.shape}")
    print(f"  Group B: {data_B.shape}")
    
    # Compute correlation matrices (using same function as corr_heatmaps)
    print("\n[Real-Real Noise Floor] Computing correlation matrices...")
    C_A, _ = compute_lag0_corr_matrix(data_A)
    C_B, _ = compute_lag0_corr_matrix(data_B)
    C_diff = np.abs(C_A - C_B)
    
    print(f"  C_A shape: {C_A.shape}")
    print(f"  C_B shape: {C_B.shape}")
    
    # Statistics
    off_diag_mask = ~np.eye(D, dtype=bool)
    diff_values = C_diff[off_diag_mask & ~np.isnan(C_diff)]
    
    print(f"\n[Real-Real Noise Floor] |C_A - C_B| statistics (off-diagonal):")
    print(f"  Mean:   {np.mean(diff_values):.4f}")
    print(f"  Median: {np.median(diff_values):.4f}")
    print(f"  P90:    {np.percentile(diff_values, 90):.4f}")
    print(f"  P95:    {np.percentile(diff_values, 95):.4f}")
    print(f"  Max:    {np.max(diff_values):.4f}")
    
    # Plot heatmaps
    print("\n[Real-Real Noise Floor] Generating heatmaps...")
    
    plot_corr_heatmap(C_A, f'Real Group A (n={half_n})', out_dir / 'heatmap_real_A.png')
    plot_corr_heatmap(C_B, f'Real Group B (n={half_n})', out_dir / 'heatmap_real_B.png')
    plot_diff_heatmap(C_diff, f'|Real_A - Real_B| (Sampling Noise)', out_dir / 'heatmap_real_diff.png')
    
    # Also load gen-real diff for comparison if available
    gen_real_diff_path = None
    for eval_dir in sorted(data_dir.glob('fewshot_eval*')):
        heatmap_dir = eval_dir / 'corr_heatmaps'
        if heatmap_dir.exists():
            for f in heatmap_dir.glob('*_corr_heatmaps_summary.json'):
                gen_real_diff_path = f
                break
    
    if gen_real_diff_path:
        print(f"\n[Comparison] Loading gen-real diff from {gen_real_diff_path.name}")
        with open(gen_real_diff_path) as f:
            gen_real_data = json.load(f)
        
        if 'lag0' in gen_real_data:
            C_real = np.array(gen_real_data['lag0']['C_real'])
            C_gen = np.array(gen_real_data['lag0']['C_gen'])
            gen_real_diff = np.abs(C_gen - C_real)
            
            # Use shape from gen_real_diff for mask
            D_gen = gen_real_diff.shape[0]
            gen_off_diag_mask = ~np.eye(D_gen, dtype=bool)
            gen_diff_values = gen_real_diff[gen_off_diag_mask & ~np.isnan(gen_real_diff)]
            
            print(f"\n[Comparison] |Gen - Real| statistics (off-diagonal):")
            print(f"  Mean:   {np.mean(gen_diff_values):.4f}")
            print(f"  Median: {np.median(gen_diff_values):.4f}")
            print(f"  P90:    {np.percentile(gen_diff_values, 90):.4f}")
            print(f"  Max:    {np.max(gen_diff_values):.4f}")
            
            print(f"\n" + "=" * 60)
            print(f"[Conclusion]")
            print(f"  Sampling noise (Real_A - Real_B): mean={np.mean(diff_values):.4f}")
            print(f"  Model error   (Gen - Real):       mean={np.mean(gen_diff_values):.4f}")
            
            if np.mean(gen_diff_values) <= np.mean(diff_values) * 1.5:
                print(f"  => Gen-Real difference is within ~1.5x of sampling noise!")
                print(f"  => Correlation structure is well captured by the model.")
            else:
                ratio = np.mean(gen_diff_values) / np.mean(diff_values)
                print(f"  => Gen-Real difference is {ratio:.1f}x larger than sampling noise.")
            print("=" * 60)
    
    # ============ Additional Visualizations ============
    if gen_real_diff_path and 'lag0' in gen_real_data:
        print("\n[Additional Visualizations] Generating comparison plots...")
        
        # 1. Bar chart: Compare mean/median/max
        fig, ax = plt.subplots(figsize=(8, 5))
        
        metrics = ['Mean', 'Median', 'P90', 'Max']
        sampling_vals = [
            np.mean(diff_values), 
            np.median(diff_values),
            np.percentile(diff_values, 90),
            np.max(diff_values)
        ]
        model_vals = [
            np.mean(gen_diff_values),
            np.median(gen_diff_values),
            np.percentile(gen_diff_values, 90),
            np.max(gen_diff_values)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, sampling_vals, width, label='Sampling noise floor\n(Real vs Real)', color='#4ECDC4', edgecolor='black')
        bars2 = ax.bar(x + width/2, model_vals, width, label='Generation error\n(Gen vs Real)', color='#FF6B6B', edgecolor='black')
        
        ax.set_ylabel(r'$E_{\mathrm{corr}}$', fontsize=11)
        # ax.set_title('Sampling Noise vs Model Error', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(fontsize=9)
        # ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(out_dir / 'comparison_bar.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / 'comparison_bar.png'}")
        
        # 2. Histogram overlay: Distribution comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bins = np.linspace(0, max(np.max(diff_values), np.max(gen_diff_values)) * 1.1, 20)
        
        ax.hist(diff_values, bins=bins, alpha=0.6, label='Sampling Noise (Real_A vs Real_B)', 
                color='#4ECDC4', edgecolor='black', linewidth=0.5)
        ax.hist(gen_diff_values, bins=bins, alpha=0.6, label='Model Error (Gen vs Real)', 
                color='#FF6B6B', edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for means
        ax.axvline(np.mean(diff_values), color='#2E8B7A', linestyle='--', linewidth=2, 
                   label=f'Sampling Mean: {np.mean(diff_values):.3f}')
        ax.axvline(np.mean(gen_diff_values), color='#CC4444', linestyle='--', linewidth=2,
                   label=f'Model Mean: {np.mean(gen_diff_values):.3f}')
        
        ax.set_xlabel('|Correlation Difference|', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Distribution of Correlation Differences (Off-diagonal)', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / 'comparison_histogram.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / 'comparison_histogram.png'}")
        
        # 3. Scatter plot: Per-pair comparison
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Get per-pair differences
        D_plot = min(D, D_gen)
        scatter_sampling = []
        scatter_model = []
        pair_labels = []
        
        for i in range(D_plot):
            for j in range(D_plot):
                if i != j:
                    samp_val = C_diff[i, j]
                    model_val = gen_real_diff[i, j]
                    if not (np.isnan(samp_val) or np.isnan(model_val)):
                        scatter_sampling.append(samp_val)
                        scatter_model.append(model_val)
                        pair_labels.append(f'{CHANNEL_NAMES[i]}-{CHANNEL_NAMES[j]}')
        
        scatter_sampling = np.array(scatter_sampling)
        scatter_model = np.array(scatter_model)
        
        # Plot diagonal line (y=x)
        max_val = max(scatter_sampling.max(), scatter_model.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y = x (Equal error)')
        
        # Scatter plot
        scatter = ax.scatter(scatter_sampling, scatter_model, c='#6B5B95', s=80, 
                            alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Annotate points
        for i, label in enumerate(pair_labels):
            ax.annotate(label, (scatter_sampling[i], scatter_model[i]), 
                       fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')
        
        ax.set_xlabel('Sampling Noise |Real_A - Real_B|', fontsize=11)
        ax.set_ylabel('Model Error |Gen - Real|', fontsize=11)
        ax.set_title('Per-Channel-Pair Comparison', fontsize=13)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        below_diag = np.sum(scatter_model < scatter_sampling)
        total = len(scatter_model)
        ax.text(0.05, 0.95, f'{below_diag}/{total} pairs have\nModel Error < Sampling Noise', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(out_dir / 'comparison_scatter.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / 'comparison_scatter.png'}")
    
    # Save summary
    summary = {
        'N_total': int(N),
        'N_per_group': int(half_n),
        'seed': args.seed,
        'real_A_vs_B': {
            'mean': float(np.mean(diff_values)),
            'median': float(np.median(diff_values)),
            'p90': float(np.percentile(diff_values, 90)),
            'p95': float(np.percentile(diff_values, 95)),
            'max': float(np.max(diff_values))
        },
        'C_A': C_A.tolist(),
        'C_B': C_B.tolist()
    }
    
    summary_path = out_dir / 'noise_floor_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")
    
    print("\n[Real-Real Noise Floor] Done!")
    print(f"\nOutput files:")
    print(f"  - {out_dir / 'heatmap_real_A.png'}")
    print(f"  - {out_dir / 'heatmap_real_B.png'}")
    print(f"  - {out_dir / 'heatmap_real_diff.png'}")
    print(f"  - {out_dir / 'noise_floor_summary.json'}")


if __name__ == '__main__':
    main()
