#!/usr/bin/env python3
"""
Generate UMAP/t-SNE visualization of REAL vs GEN samples from evaluation outputs.

Usage:
    python tools/make_umap_from_eval.py --data_dir data/nuscenes_planA_... --out_dir results/

Key feature: Uses DENORMALIZED data and standardizes with global real-train stats
to ensure fair comparison across different normalization methods.

Output:
    - umap_real_vs_gen.png
    - umap_coords.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Try to import UMAP, fall back to t-SNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, falling back to sklearn t-SNE")

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_denorm_samples(eval_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load DENORMALIZED gen and real samples from eval directory.
    
    Returns:
        (gen_denorm, real_denorm) or (None, None) if not found
    """
    samples_dir = eval_dir / 'samples'
    gen_denorm = None
    real_denorm = None
    
    if samples_dir.exists():
        # Look for denormalized samples
        for f in samples_dir.glob('*_gen_denorm.npy'):
            gen_denorm = np.load(f)
            break
        if gen_denorm is None:
            gen_path = samples_dir / 'gen_samples.npy'
            if gen_path.exists():
                gen_denorm = np.load(gen_path)
        
        for f in samples_dir.glob('*_real_denorm.npy'):
            real_denorm = np.load(f)
            break
        if real_denorm is None:
            real_path = samples_dir / 'real_eval.npy'
            if real_path.exists():
                real_denorm = np.load(real_path)
    
    # Fallback to direct files
    if gen_denorm is None:
        for name in ['gen_samples.npy', 'generated_samples.npy']:
            path = eval_dir / name
            if path.exists():
                gen_denorm = np.load(path)
                break
    
    return gen_denorm, real_denorm


def compute_global_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from training data.
    
    Args:
        data: (N, T, D) array
        
    Returns:
        mean: (D,) per-channel mean
        std: (D,) per-channel std
    """
    # Flatten to (N*T, D)
    N, T, D = data.shape
    flat = data.reshape(-1, D)
    
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
    
    return mean, std


def standardize_with_global_stats(
    data: np.ndarray, 
    global_mean: np.ndarray, 
    global_std: np.ndarray
) -> np.ndarray:
    """
    Standardize data using pre-computed global mean/std.
    
    Args:
        data: (N, T, D) array
        global_mean: (D,) mean
        global_std: (D,) std
        
    Returns:
        standardized: (N, T, D) array
    """
    return (data - global_mean) / global_std


def flatten_trajectories(data: np.ndarray) -> np.ndarray:
    """
    Flatten trajectories from (N, T, D) to (N, T*D).
    """
    N = data.shape[0]
    return data.reshape(N, -1)


def compute_embedding(
    data: np.ndarray,
    random_state: int = 0,
    use_umap: bool = True
) -> np.ndarray:
    """
    Compute 2D embedding using UMAP or t-SNE.
    
    Note: Data should already be standardized before calling this.
    """
    if use_umap and HAS_UMAP:
        print("  Using UMAP...")
        reducer = UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=random_state
        )
    else:
        print("  Using t-SNE...")
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(data) // 4),
            random_state=random_state,
            n_iter=1000
        )
    
    coords = reducer.fit_transform(data)
    return coords


def plot_umap(
    coords_real: np.ndarray,
    coords_gen: np.ndarray,
    labels_real: np.ndarray,
    labels_gen: np.ndarray,
    output_path: Path,
    title: str = "t-SNE: Real vs Generated"
):
    """
    Create scatter plot with real/gen markers.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color map for domains
    unique_domains = np.unique(np.concatenate([labels_real, labels_gen]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
    domain_to_color = {d: colors[i] for i, d in enumerate(unique_domains)}
    
    # Plot real samples (circles)
    for domain in unique_domains:
        mask = labels_real == domain
        if mask.sum() > 0:
            ax.scatter(
                coords_real[mask, 0], coords_real[mask, 1],
                c=[domain_to_color[domain]],
                marker='o',
                s=50,
                alpha=0.6,
                label=f'D{domain} Real',
                edgecolors='black',
                linewidths=0.5
            )
    
    # Plot generated samples (triangles)
    for domain in unique_domains:
        mask = labels_gen == domain
        if mask.sum() > 0:
            ax.scatter(
                coords_gen[mask, 0], coords_gen[mask, 1],
                c=[domain_to_color[domain]],
                marker='^',
                s=60,
                alpha=0.7,
                label=f'D{domain} Gen',
                edgecolors='black',
                linewidths=0.5
            )
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved plot: {output_path}")


def save_coords_csv(
    coords_real: np.ndarray,
    coords_gen: np.ndarray,
    labels_real: np.ndarray,
    labels_gen: np.ndarray,
    output_path: Path,
    mode: str = 'k_shot',
    K: int = 16
):
    """Save coordinates to CSV."""
    records = []
    
    for i in range(len(coords_real)):
        records.append({
            'x': coords_real[i, 0],
            'y': coords_real[i, 1],
            'domain': int(labels_real[i]),
            'source': 'real',
            'mode': mode,
            'K': K
        })
    
    for i in range(len(coords_gen)):
        records.append({
            'x': coords_gen[i, 0],
            'y': coords_gen[i, 1],
            'domain': int(labels_gen[i]),
            'source': 'gen',
            'mode': mode,
            'K': K
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"  Saved coords: {output_path}")


def find_eval_folders(data_dir: Path) -> List[Path]:
    """Find all fewshot_eval* folders."""
    eval_folders = []
    for item in data_dir.iterdir():
        if item.is_dir() and 'fewshot_eval' in item.name:
            eval_folders.append(item)
    return sorted(eval_folders)


def main():
    parser = argparse.ArgumentParser(description='Generate UMAP/t-SNE visualization from eval outputs')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root data directory')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: data_dir/umap_viz)')
    parser.add_argument('--use_tsne', action='store_true',
                        help='Force t-SNE even if UMAP is available')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--use_denorm', action='store_true', default=True,
                        help='Use denormalized data (default: True)')
    parser.add_argument('--global_stats_from', type=str, default=None,
                        help='Path to real training data for global stats (default: seen_train_pool.npy in data_dir)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / 'umap_viz'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    use_umap = HAS_UMAP and not args.use_tsne
    method_name = 'UMAP' if use_umap else 't-SNE'
    
    print(f"[{method_name}] Data dir: {data_dir}")
    print(f"[{method_name}] Output dir: {out_dir}")
    print(f"[{method_name}] Using denormalized data: {args.use_denorm}")
    
    # ========== Step 1: Load global stats from real training data ==========
    # This ensures all normalization methods are compared on the same physical scale
    
    # Try to find training data for global stats
    train_pool_path = None
    if args.global_stats_from:
        train_pool_path = Path(args.global_stats_from)
    else:
        # Try seen_train_pool first, then seen_eval_pool
        for name in ['seen_train_pool.npy', 'seen_eval_pool.npy']:
            candidate = data_dir / name
            if candidate.exists():
                train_pool_path = candidate
                break
    
    if train_pool_path is None or not train_pool_path.exists():
        print(f"WARNING: No training pool found for global stats computation.")
        print("  Will use real eval data for standardization (less ideal).")
        global_mean, global_std = None, None
    else:
        print(f"[{method_name}] Loading training data for global stats: {train_pool_path}")
        train_data = np.load(train_pool_path)
        print(f"  Training data shape: {train_data.shape}")
        global_mean, global_std = compute_global_stats(train_data)
        print(f"  Global mean: {global_mean}")
        print(f"  Global std: {global_std}")
    
    # ========== Step 2: Find and load evaluation data ==========
    eval_folders = find_eval_folders(data_dir)
    print(f"[{method_name}] Found {len(eval_folders)} eval folders:")
    for f in eval_folders:
        print(f"  - {f.name}")
    
    # Load real unseen pool (for comparison)
    unseen_pool_path = data_dir / 'unseen_eval_pool.npy'
    if not unseen_pool_path.exists():
        print(f"ERROR: {unseen_pool_path} not found!")
        return
    
    real_data = np.load(unseen_pool_path)
    print(f"[{method_name}] Loaded real unseen data: {real_data.shape}")
    
    # Try to find generated samples (denormalized)
    gen_data = None
    real_eval_data = None
    
    for eval_dir in eval_folders:
        gen_denorm, real_denorm = load_denorm_samples(eval_dir)
        if gen_denorm is not None:
            print(f"[{method_name}] Found denorm samples in {eval_dir.name}")
            print(f"  gen_denorm: {gen_denorm.shape}")
            gen_data = gen_denorm
            if real_denorm is not None:
                real_eval_data = real_denorm
                print(f"  real_denorm: {real_denorm.shape}")
            break
    
    if gen_data is None:
        print("\n" + "=" * 70)
        print("ERROR: No generated samples found!")
        print("=" * 70)
        print("\nMake sure to run eval with --save_samples flag.")
        return
    
    # Use the eval real data if available (should match gen count)
    if real_eval_data is not None:
        real_data = real_eval_data
        print(f"[{method_name}] Using real_denorm from eval (matched with gen)")
    
    # ========== Step 3: Standardize with global stats ==========
    print(f"\n[{method_name}] Standardizing data with global training stats...")
    
    if global_mean is None:
        # Fall back: compute from real data
        print("  (Using real eval data for stats - not ideal for cross-method comparison)")
        global_mean, global_std = compute_global_stats(real_data)
    
    # Standardize both real and gen with THE SAME global stats
    real_std = standardize_with_global_stats(real_data, global_mean, global_std)
    gen_std = standardize_with_global_stats(gen_data, global_mean, global_std)
    
    print(f"  Real standardized range: [{real_std.min():.2f}, {real_std.max():.2f}]")
    print(f"  Gen standardized range: [{gen_std.min():.2f}, {gen_std.max():.2f}]")
    
    # Check for extreme values in gen (may indicate generation issues)
    gen_extreme_frac = np.mean(np.abs(gen_std) > 5.0)
    if gen_extreme_frac > 0.01:
        print(f"  WARNING: {gen_extreme_frac*100:.1f}% of gen values > 5 std from training mean!")
    
    # ========== Step 4: Flatten and compute embedding ==========
    real_labels = np.full(len(real_data), 3)  # Domain 3 (unseen)
    gen_labels = np.full(len(gen_data), 3)
    
    real_flat = flatten_trajectories(real_std)
    gen_flat = flatten_trajectories(gen_std)
    
    print(f"\n[{method_name}] Flattened shapes: Real {real_flat.shape}, Gen {gen_flat.shape}")
    
    # Combine for joint embedding
    combined = np.vstack([real_flat, gen_flat])
    print(f"[{method_name}] Computing embedding for {len(combined)} samples...")
    
    # Note: Data is already standardized, so we skip StandardScaler in compute_embedding
    # But t-SNE/UMAP might still benefit from it on the flattened features
    coords = compute_embedding(combined, random_state=args.seed, use_umap=use_umap)
    
    # Split back
    n_real = len(real_flat)
    coords_real = coords[:n_real]
    coords_gen = coords[n_real:]
    
    # ========== Step 5: Analyze separation ==========
    from scipy.spatial.distance import cdist
    
    d_real_real = cdist(coords_real, coords_real).mean()
    d_gen_gen = cdist(coords_gen, coords_gen).mean()
    d_gen_real = cdist(coords_gen, coords_real).mean()
    
    sep_ratio = d_gen_real / max(d_real_real, d_gen_gen)
    
    print(f"\n[{method_name}] Embedding Statistics:")
    print(f"  Avg dist within Real: {d_real_real:.2f}")
    print(f"  Avg dist within Gen:  {d_gen_gen:.2f}")
    print(f"  Avg dist Gen<->Real:  {d_gen_real:.2f}")
    print(f"  Separation ratio: {sep_ratio:.2f} (1.0=similar, >1.5=distinct clusters)")
    
    # Real/Gen center distance
    real_center = coords_real.mean(axis=0)
    gen_center = coords_gen.mean(axis=0)
    center_dist = np.linalg.norm(real_center - gen_center)
    print(f"  Center distance: {center_dist:.2f}")
    
    # ========== Step 6: Plot and save ==========
    title = f"{method_name}: Real vs Generated (Denorm + Global Std)"
    plot_umap(
        coords_real, coords_gen,
        real_labels, gen_labels,
        out_dir / 'umap_real_vs_gen.png',
        title=title
    )
    
    save_coords_csv(
        coords_real, coords_gen,
        real_labels, gen_labels,
        out_dir / 'umap_coords.csv'
    )
    
    # Save analysis summary
    summary = {
        'method': method_name,
        'n_real': int(n_real),
        'n_gen': int(len(gen_flat)),
        'global_mean': global_mean.tolist(),
        'global_std': global_std.tolist(),
        'd_real_real': float(d_real_real),
        'd_gen_gen': float(d_gen_gen),
        'd_gen_real': float(d_gen_real),
        'separation_ratio': float(sep_ratio),
        'center_distance': float(center_dist),
        'gen_extreme_frac': float(gen_extreme_frac) if 'gen_extreme_frac' in dir() else 0.0
    }
    
    summary_path = out_dir / 'umap_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")
    
    print(f"\n[{method_name}] Done!")


if __name__ == '__main__':
    main()
