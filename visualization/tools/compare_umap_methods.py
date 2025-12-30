#!/usr/bin/env python3
"""
Compare UMAP/t-SNE embeddings across different normalization methods.

Uses DENORMALIZED data from each method and standardizes ALL with
the SAME global stats (from one reference method's real data).

This ensures fair geometric comparison across zscore, zscore_winsor, centered_pit.

Usage:
    python visualization/tools/compare_umap_methods.py \
        --zscore_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
        --centered_pit_dir data/nuscenes_planA_T32_centered_pit_seed0 \
        --out_dir visualization/umap_comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Try to import UMAP, fall back to t-SNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, falling back to sklearn t-SNE")

from sklearn.manifold import TSNE


def load_denorm_samples(data_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load denormalized gen and real samples."""
    # Check fewshot_eval folders
    for eval_dir in sorted(data_dir.glob('fewshot_eval*')):
        samples_dir = eval_dir / 'samples'
        if not samples_dir.exists():
            continue
        
        gen_path = samples_dir / 'gen_samples.npy'
        real_path = samples_dir / 'real_eval.npy'
        
        if gen_path.exists() and real_path.exists():
            return np.load(gen_path), np.load(real_path)
        
        # Try D*_K*_gen_denorm.npy pattern
        for f in samples_dir.glob('*_gen_denorm.npy'):
            gen = np.load(f)
            for r in samples_dir.glob('*_real_denorm.npy'):
                real = np.load(r)
                return gen, real
    
    return None, None


def compute_global_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std."""
    N, T, D = data.shape
    flat = data.reshape(-1, D)
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize with given mean/std."""
    return (data - mean) / std


def flatten(data: np.ndarray) -> np.ndarray:
    """Flatten (N, T, D) to (N, T*D)."""
    return data.reshape(data.shape[0], -1)


def compute_embedding(data: np.ndarray, random_state: int = 0, use_umap: bool = True) -> np.ndarray:
    """Compute 2D embedding."""
    if use_umap and HAS_UMAP:
        print("  Using UMAP...")
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, 
                       metric='euclidean', random_state=random_state)
    else:
        print("  Using t-SNE...")
        perp = min(30, len(data) // 4)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=random_state, n_iter=1000)
    
    return reducer.fit_transform(data)


def analyze_separation(coords_real: np.ndarray, coords_gen: np.ndarray, method_name: str) -> Dict:
    """Compute separation statistics."""
    d_real_real = cdist(coords_real, coords_real).mean()
    d_gen_gen = cdist(coords_gen, coords_gen).mean()
    d_gen_real = cdist(coords_gen, coords_real).mean()
    sep_ratio = d_gen_real / max(d_real_real, d_gen_gen, 1e-8)
    
    real_center = coords_real.mean(axis=0)
    gen_center = coords_gen.mean(axis=0)
    center_dist = np.linalg.norm(real_center - gen_center)
    
    print(f"\n  [{method_name}] Separation Analysis:")
    print(f"    Avg dist within Real: {d_real_real:.2f}")
    print(f"    Avg dist within Gen:  {d_gen_gen:.2f}")
    print(f"    Avg dist Gen<->Real:  {d_gen_real:.2f}")
    print(f"    Separation ratio: {sep_ratio:.2f}")
    print(f"    Center distance: {center_dist:.2f}")
    
    return {
        'd_real_real': float(d_real_real),
        'd_gen_gen': float(d_gen_gen),
        'd_gen_real': float(d_gen_real),
        'separation_ratio': float(sep_ratio),
        'center_distance': float(center_dist)
    }


def main():
    parser = argparse.ArgumentParser(description='Compare UMAP across normalization methods')
    parser.add_argument('--zscore_dir', type=str, default=None, help='zscore data dir')
    parser.add_argument('--zscore_winsor_dir', type=str, default=None, help='zscore_winsor data dir')
    parser.add_argument('--centered_pit_dir', type=str, default=None, help='centered_pit data dir')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--reference', type=str, default='zscore', 
                        choices=['zscore', 'zscore_winsor', 'centered_pit'],
                        help='Method to use as reference for global stats')
    parser.add_argument('--use_tsne', action='store_true', help='Force t-SNE')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    use_umap = HAS_UMAP and not args.use_tsne
    method_name = 'UMAP' if use_umap else 't-SNE'
    
    print(f"[Compare {method_name}] Output: {out_dir}")
    
    # ========== Load all available methods ==========
    methods = {}
    method_dirs = {
        'zscore': args.zscore_dir,
        'zscore_winsor': args.zscore_winsor_dir,
        'centered_pit': args.centered_pit_dir
    }
    
    for name, dir_path in method_dirs.items():
        if dir_path is None:
            continue
        data_dir = Path(dir_path)
        if not data_dir.exists():
            print(f"  [{name}] Directory not found: {data_dir}")
            continue
        
        gen, real = load_denorm_samples(data_dir)
        if gen is None:
            print(f"  [{name}] No samples found")
            continue
        
        print(f"  [{name}] Loaded: gen={gen.shape}, real={real.shape}")
        methods[name] = {'gen': gen, 'real': real, 'dir': data_dir}
    
    if len(methods) == 0:
        print("ERROR: No methods found with valid samples!")
        return
    
    print(f"\nLoaded {len(methods)} methods: {list(methods.keys())}")
    
    # ========== Compute global stats from reference ==========
    ref_method = args.reference
    if ref_method not in methods:
        ref_method = list(methods.keys())[0]
        print(f"  Reference '{args.reference}' not available, using '{ref_method}'")
    
    ref_real = methods[ref_method]['real']
    global_mean, global_std = compute_global_stats(ref_real)
    
    print(f"\n[Global Stats for UMAP input normalization]")
    print(f"  (Computed from real data - same physical space for all methods)")
    print(f"  Mean: {global_mean}")
    print(f"  Std:  {global_std}")
    
    # ========== Standardize all data with global stats ==========
    all_data = []
    all_labels = []  # (method, source)
    all_method_names = []
    all_sources = []
    
    for name, data in methods.items():
        real_std = standardize(data['real'], global_mean, global_std)
        gen_std = standardize(data['gen'], global_mean, global_std)
        
        real_flat = flatten(real_std)
        gen_flat = flatten(gen_std)
        
        n_real = len(real_flat)
        n_gen = len(gen_flat)
        
        all_data.append(real_flat)
        all_data.append(gen_flat)
        
        all_method_names.extend([name] * n_real)
        all_method_names.extend([name] * n_gen)
        all_sources.extend(['real'] * n_real)
        all_sources.extend(['gen'] * n_gen)
        
        # Check for extreme values
        gen_extreme = np.mean(np.abs(gen_std) > 5.0)
        print(f"  [{name}] Gen extreme values (>5σ): {gen_extreme*100:.2f}%")
    
    combined = np.vstack(all_data)
    print(f"\n[{method_name}] Computing embedding for {len(combined)} samples...")
    
    # ========== Compute joint embedding ==========
    coords = compute_embedding(combined, random_state=args.seed, use_umap=use_umap)
    
    # ========== Build DataFrame ==========
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'method': all_method_names,
        'source': all_sources
    })
    
    # ========== Analyze each method ==========
    results = {}
    for name in methods.keys():
        mask_real = (df['method'] == name) & (df['source'] == 'real')
        mask_gen = (df['method'] == name) & (df['source'] == 'gen')
        
        coords_real = df.loc[mask_real, ['x', 'y']].values
        coords_gen = df.loc[mask_gen, ['x', 'y']].values
        
        results[name] = analyze_separation(coords_real, coords_gen, name)
    
    # ========== Plot comparison ==========
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    colors = {'real': 'blue', 'gen': 'red'}
    markers = {'real': 'o', 'gen': '^'}
    
    for ax, name in zip(axes, methods.keys()):
        for source in ['real', 'gen']:
            mask = (df['method'] == name) & (df['source'] == source)
            data_subset = df.loc[mask]
            ax.scatter(
                data_subset['x'], data_subset['y'],
                c=colors[source],
                marker=markers[source],
                s=40,
                alpha=0.6,
                label=source.capitalize(),
                edgecolors='black',
                linewidths=0.3
            )
        
        sep = results[name]['separation_ratio']
        ax.set_title(f'{name}\n(sep ratio: {sep:.2f})', fontsize=12)
        # ax.set_xlabel('Component 1')
        # ax.set_ylabel('Component 2')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # plt.suptitle(f'{method_name} Comparison (Denormalized Physical Space)', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'umap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir / 'umap_comparison.png'}")
    
    # ========== Combined plot ==========
    fig, ax = plt.subplots(figsize=(12, 10))
    
    method_colors = {
        'zscore': plt.cm.tab10(0),
        'zscore_winsor': plt.cm.tab10(1),
        'centered_pit': plt.cm.tab10(2)
    }
    
    for name in methods.keys():
        for source in ['real', 'gen']:
            mask = (df['method'] == name) & (df['source'] == source)
            data_subset = df.loc[mask]
            marker = 'o' if source == 'real' else '^'
            alpha = 0.7 if source == 'real' else 0.5
            ax.scatter(
                data_subset['x'], data_subset['y'],
                c=[method_colors.get(name, 'gray')],
                marker=marker,
                s=50,
                alpha=alpha,
                label=f'{name} {source}',
                edgecolors='black',
                linewidths=0.3
            )
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(f'{method_name}: All Methods in Physical Space (●=Real, ▲=Gen)', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'umap_all_methods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'umap_all_methods.png'}")
    
    # ========== Save data ==========
    df.to_csv(out_dir / 'umap_coords.csv', index=False)
    
    summary = {
        'embedding_method': method_name,
        'note': 'All methods compared in denormalized physical space with unified standardization',
        'global_mean': global_mean.tolist(),
        'global_std': global_std.tolist(),
        'methods': results
    }
    with open(out_dir / 'umap_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: {out_dir / 'umap_coords.csv'}")
    print(f"Saved: {out_dir / 'umap_summary.json'}")
    
    # ========== Print summary table ==========
    print("\n" + "=" * 70)
    print("SUMMARY: Separation Ratio (higher = more distinct gen vs real)")
    print("=" * 70)
    for name, res in sorted(results.items(), key=lambda x: x[1]['separation_ratio']):
        print(f"  {name:20s}: sep_ratio={res['separation_ratio']:.3f}, center_dist={res['center_distance']:.2f}")
    
    best = min(results.items(), key=lambda x: x[1]['separation_ratio'])
    print(f"\n  BEST (lowest separation): {best[0]}")
    print("=" * 70)


if __name__ == '__main__':
    main()

