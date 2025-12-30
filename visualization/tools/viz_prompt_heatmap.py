#!/usr/bin/env python3
"""
Prompt Heatmap Visualization (TimeDP Fig.3 analogue)

Shows PAM produces structured, domain-dependent prompt vectors (mask/prototype weights).
Plots a heatmap of (samples × prototypes) colored by mask values, grouped by domain.

Usage:
    python tools/viz_prompt_heatmap.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --config configs/nuscenes_planA_zscore.yaml \
        --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
        --domains 0,1,2 \
        --n_per_domain 50 \
        --output figures/prompt_heatmap.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.tools.prompt_viz.prompt_utils import (
    load_model, get_model_info, extract_prompts,
    load_domain_data, sparsify_mask, softmax_mask
)


DOMAIN_NAMES = {
    0: 'boston-seaport',
    1: 'singapore-onenorth', 
    2: 'singapore-queenstown',
    3: 'singapore-hollandvillage (unseen)'
}


def plot_prompt_heatmap(
    masks: np.ndarray,
    domain_labels: np.ndarray,
    output_path: Path,
    sort_by: str = 'argmax',
    use_softmax: bool = False,
    title: str = 'Prompt Heatmap (Samples × Prototypes)'
):
    """
    Plot heatmap of prompt masks grouped by domain.
    
    Args:
        masks: (N, K) mask logits or weights
        domain_labels: (N,) domain indices
        output_path: Where to save figure
        sort_by: 'argmax' or 'sum' - how to sort samples within domain
        use_softmax: Whether to apply softmax for visualization
        title: Plot title
    """
    K = masks.shape[1]
    unique_domains = np.unique(domain_labels)
    
    # Optionally apply softmax for visualization
    if use_softmax:
        display_masks = softmax_mask(masks)
        cmap = 'viridis'
        vmin, vmax = 0, 1
        cbar_label = 'Attention Weight'
    else:
        # Clip extreme values for visualization
        display_masks = np.clip(masks, -5, 5)
        cmap = 'RdBu_r'
        vmin, vmax = -3, 3
        cbar_label = 'Mask Logit'
    
    # Sort and group by domain
    sorted_masks = []
    domain_boundaries = [0]
    domain_centers = []
    
    for d in unique_domains:
        domain_mask = domain_labels == d
        domain_data = display_masks[domain_mask]
        
        if sort_by == 'argmax':
            # Sort by dominant prototype index
            sort_idx = np.lexsort((
                -domain_data.max(axis=1),  # Secondary: max value (descending)
                domain_data.argmax(axis=1)  # Primary: argmax prototype
            ))
        else:
            # Sort by sum of positive values
            sort_idx = np.argsort(-np.sum(np.maximum(domain_data, 0), axis=1))
        
        sorted_masks.append(domain_data[sort_idx])
        domain_boundaries.append(domain_boundaries[-1] + len(domain_data))
        domain_centers.append(domain_boundaries[-2] + len(domain_data) / 2)
    
    all_masks = np.vstack(sorted_masks)
    N_total = all_masks.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(all_masks, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    
    # Add domain separators
    for boundary in domain_boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color='black', linewidth=2)
    
    # Add domain labels on y-axis
    ax.set_yticks(domain_centers)
    ax.set_yticklabels([DOMAIN_NAMES.get(d, f'Domain {d}') for d in unique_domains], fontsize=10)
    
    # X-axis: prototype indices
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'{i}' for i in range(K)], fontsize=9)
    ax.set_xlabel('Prototype Index', fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    
    # Title
    # ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Add sample count annotations
    for i, d in enumerate(unique_domains):
        n_samples = domain_boundaries[i+1] - domain_boundaries[i]
        ax.text(-0.5, domain_centers[i], f'n={n_samples}', 
                ha='right', va='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def plot_raw_vs_sparse_heatmap(
    raw_masks: np.ndarray,
    domain_labels: np.ndarray,
    output_path: Path,
    title: str = 'Raw vs Sparsified Masks'
):
    """
    Plot side-by-side comparison of raw and sparsified masks.
    Each domain is shown in a separate row.
    """
    sparse_masks = sparsify_mask(raw_masks, threshold=0.0)
    K = raw_masks.shape[1]
    unique_domains = sorted(np.unique(domain_labels))
    
    # Prepare data for each domain
    domain_data = {}
    for d in unique_domains:
        domain_mask = domain_labels == d
        raw_d = raw_masks[domain_mask]
        sparse_d = sparse_masks[domain_mask]
        
        sort_idx = np.lexsort((
            -raw_d.max(axis=1),
            raw_d.argmax(axis=1)
        ))
        
        domain_data[d] = {
            'raw': raw_d[sort_idx],
            'sparse': sparse_d[sort_idx]
        }
    
    # Create figure with vertical layout: one row per domain, two columns (raw/sparse)
    n_domains = len(unique_domains)
    fig, axes = plt.subplots(n_domains, 2, figsize=(14, 4 * n_domains))
    
    # Handle single domain case
    if n_domains == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each domain
    for row_idx, d in enumerate(unique_domains):
        ax_raw = axes[row_idx, 0]
        ax_sparse = axes[row_idx, 1]
        
        raw_d = domain_data[d]['raw']
        sparse_d = domain_data[d]['sparse']
        
        # Raw masks
        raw_clipped = np.clip(raw_d, -5, 5)
        im1 = ax_raw.imshow(raw_clipped, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax_raw.set_xlabel('Prototype Index', fontsize=11)
        ax_raw.set_ylabel(f'D{d}', fontsize=11)
        ax_raw.set_xticks(range(K))
        cbar1 = plt.colorbar(im1, ax=ax_raw, shrink=0.8)
        cbar1.set_label('Logit', fontsize=11)
        
        # Sparsified masks (show active prototypes)
        sparse_binary = (sparse_d > -1e8).astype(float)
        im2 = ax_sparse.imshow(sparse_binary, aspect='auto', cmap='Greens', vmin=0, vmax=1)
        ax_sparse.set_xlabel('Prototype Index', fontsize=11)
        ax_sparse.set_ylabel(f'D{d}', fontsize=11)
        ax_sparse.set_xticks(range(K))
        cbar2 = plt.colorbar(im2, ax=ax_sparse, shrink=0.8)
        cbar2.set_label('Active (1) / Masked (0)', fontsize=11)
    
    # Add column labels at the top of first row only
    if n_domains > 0:
        axes[0, 0].set_title('Raw Mask Logits', fontsize=14, pad=10)
        axes[0, 1].set_title('Active Prototypes (after sparsify)', fontsize=14, pad=10)
    
    # Increase spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prompt Heatmap Visualization')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--split', type=str, default='seen_train',
                        choices=['seen_train', 'seen_eval', 'unseen_eval'],
                        help='Data split to use')
    parser.add_argument('--domains', type=str, default='0,1,2',
                        help='Comma-separated domain indices')
    parser.add_argument('--n_per_domain', type=int, default=50,
                        help='Number of samples per domain')
    parser.add_argument('--output', type=str, default='figures/prompt_heatmap.png',
                        help='Output path for figure')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sort_by', type=str, default='argmax',
                        choices=['argmax', 'sum'], help='How to sort samples within domain')
    parser.add_argument('--use_softmax', action='store_true',
                        help='Apply softmax to masks for visualization')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Parse domains
    domains = [int(d.strip()) for d in args.domains.split(',')]
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PROMPT HEATMAP VISUALIZATION")
    print("=" * 60)
    
    # Load model
    print(f"\n[1] Loading model from {args.ckpt}")
    model = load_model(args.ckpt, args.config, args.device)
    model_info = get_model_info(model)
    print(f"    Model type: {model_info.get('cond_model_type', 'unknown')}")
    print(f"    Impl type: {model_info.get('impl_type', 'unknown')}")
    print(f"    Num latents (K): {model_info.get('num_latents', 'unknown')}")
    print(f"    Latent dim: {model_info.get('latent_dim', 'unknown')}")
    
    # Load data
    print(f"\n[2] Loading data from {args.data_dir}")
    print(f"    Split: {args.split}, Domains: {domains}, N per domain: {args.n_per_domain}")
    
    data, domain_labels = load_domain_data(
        Path(args.data_dir),
        split=args.split,
        domains=domains,
        n_per_domain=args.n_per_domain
    )
    print(f"    Loaded {len(data)} samples")
    for d in np.unique(domain_labels):
        print(f"      Domain {d}: {np.sum(domain_labels == d)} samples")
    
    # Extract prompts
    print(f"\n[3] Extracting prompt masks...")
    masks, _ = extract_prompts(model, data, device=args.device)
    print(f"    Mask shape: {masks.shape}")
    print(f"    Mask range: [{masks.min():.2f}, {masks.max():.2f}]")
    
    # Plot main heatmap
    print(f"\n[4] Generating heatmap...")
    plot_prompt_heatmap(
        masks, domain_labels, output_path,
        sort_by=args.sort_by,
        use_softmax=args.use_softmax,
        title=f'Prompt Heatmap ({args.split})'
    )
    
    # Also plot raw vs sparse comparison
    raw_sparse_path = output_path.parent / f'{output_path.stem}_raw_vs_sparse.png'
    plot_raw_vs_sparse_heatmap(
        masks, domain_labels, raw_sparse_path,
        title=f'Raw vs Sparsified Masks ({args.split})'
    )
    
    # Print statistics
    print(f"\n[5] Mask Statistics:")
    for d in np.unique(domain_labels):
        d_masks = masks[domain_labels == d]
        active_per_sample = np.sum(d_masks > 0, axis=1)
        dominant_proto = np.argmax(d_masks, axis=1)
        print(f"    Domain {d}:")
        print(f"      Active prototypes per sample: {active_per_sample.mean():.1f} ± {active_per_sample.std():.1f}")
        print(f"      Most common dominant prototype: {np.bincount(dominant_proto).argmax()}")
    
    print(f"\n[Done] Outputs saved to {output_path.parent}/")


if __name__ == '__main__':
    main()

