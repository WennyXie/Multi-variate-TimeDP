#!/usr/bin/env python3
"""
FIG 5: Prompt Sparsity Statistics (effective prototype count)

Quantifies PAM sparsity: how many prototypes are active per sample.

Usage:
    # With pre-extracted mask logits:
    python tools/plot_prompt_sparsity.py \
        --mask_logits_npz figures/mask_logits.npz \
        --outdir figures

    # Extract from checkpoint:
    python tools/plot_prompt_sparsity.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --config configs/nuscenes_planA_timedp.yaml \
        --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
        --n_per_domain 100 \
        --outdir figures
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


DOMAIN_NAMES = {
    0: 'boston-seaport',
    1: 'singapore-onenorth', 
    2: 'singapore-queenstown',
    3: 'singapore-hollandvillage'
}

DOMAIN_COLORS = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: '#2ca02c',
    3: '#d62728'
}


def extract_masks_from_checkpoint(
    ckpt_path: str,
    config_path: str,
    data_dir: str,
    n_per_domain: int = 100,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mask logits from checkpoint.
    
    Returns:
        masks: (N, K) mask logits
        domain_labels: (N,) domain indices
    """
    from visualization.tools.prompt_viz.prompt_utils import (
        load_model, extract_prompts, load_domain_data
    )
    
    print("  Loading model...")
    model = load_model(ckpt_path, config_path, device)
    
    # Load seen domains
    print("  Loading seen domain data...")
    seen_data, seen_labels = load_domain_data(
        Path(data_dir),
        split='seen_train',
        domains=[0, 1, 2],
        n_per_domain=n_per_domain
    )
    
    # Load unseen domain
    print("  Loading unseen domain data...")
    try:
        unseen_data, unseen_labels = load_domain_data(
            Path(data_dir),
            split='unseen_eval',
            domains=[3],
            n_per_domain=n_per_domain
        )
        data = np.concatenate([seen_data, unseen_data], axis=0)
        labels = np.concatenate([seen_labels, unseen_labels], axis=0)
    except ValueError:
        # No unseen data available
        data = seen_data
        labels = seen_labels
    
    print(f"  Total samples: {len(data)}")
    
    # Extract masks
    print("  Extracting masks...")
    masks, _ = extract_prompts(model, data, device=device)
    
    return masks, labels


def compute_active_counts(masks: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Compute number of active prototypes per sample.
    
    Args:
        masks: (N, K) mask logits
        threshold: Logits > threshold are considered active
        
    Returns:
        active_counts: (N,) number of active prototypes per sample
    """
    return np.sum(masks > threshold, axis=1)


def plot_boxplot_by_domain(
    active_counts: np.ndarray,
    domain_labels: np.ndarray,
    output_path: Path
):
    """
    Plot boxplot of active prototype count grouped by domain.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    unique_domains = sorted(np.unique(domain_labels))
    
    # Prepare data for boxplot
    data_by_domain = [active_counts[domain_labels == d] for d in unique_domains]
    domain_names = [DOMAIN_NAMES.get(d, f'Domain {d}') for d in unique_domains]
    
    # Create boxplot
    bp = ax.boxplot(data_by_domain, patch_artist=True, labels=domain_names)
    
    # Color boxes by domain
    for i, (box, d) in enumerate(zip(bp['boxes'], unique_domains)):
        box.set_facecolor(DOMAIN_COLORS.get(d, 'gray'))
        box.set_alpha(0.7)
    
    # Axis labels
    ax.set_ylabel('#active prototypes')
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def plot_histogram_seen_vs_unseen(
    active_counts: np.ndarray,
    domain_labels: np.ndarray,
    output_path: Path
):
    """
    Plot histogram comparing seen vs unseen distributions.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Split into seen (0,1,2) and unseen (3)
    seen_mask = np.isin(domain_labels, [0, 1, 2])
    unseen_mask = domain_labels == 3
    
    seen_counts = active_counts[seen_mask]
    unseen_counts = active_counts[unseen_mask]
    
    # Determine bin range
    max_count = max(active_counts.max(), 16)
    bins = np.arange(0, max_count + 2) - 0.5
    
    # Plot histograms
    if len(seen_counts) > 0:
        ax.hist(seen_counts, bins=bins, alpha=0.6, 
               label=f'seen (n={len(seen_counts)})',
               color='#1f77b4', edgecolor='black', linewidth=0.5)
    
    if len(unseen_counts) > 0:
        ax.hist(unseen_counts, bins=bins, alpha=0.6,
               label=f'unseen (n={len(unseen_counts)})',
               color='#d62728', edgecolor='black', linewidth=0.5)
    
    # Add mean lines
    if len(seen_counts) > 0:
        ax.axvline(seen_counts.mean(), color='#1f77b4', linestyle='--', 
                  linewidth=2, alpha=0.8)
    if len(unseen_counts) > 0:
        ax.axvline(unseen_counts.mean(), color='#d62728', linestyle='--',
                  linewidth=2, alpha=0.8)
    
    # Axis labels
    ax.set_xlabel('#active prototypes')
    ax.set_ylabel('Count')
    
    # X-axis ticks at integers
    ax.set_xticks(range(0, int(max_count) + 1, 2))
    
    # Legend
    ax.legend(loc='upper right', fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def print_statistics(
    active_counts: np.ndarray,
    domain_labels: np.ndarray
):
    """Print summary statistics."""
    print("\n  Sparsity Statistics:")
    print(f"  {'Domain':<25} {'Mean':<8} {'Std':<8} {'Median':<8} {'Min':<6} {'Max':<6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    
    for d in sorted(np.unique(domain_labels)):
        counts = active_counts[domain_labels == d]
        name = DOMAIN_NAMES.get(d, f'Domain {d}')
        print(f"  {name:<25} {counts.mean():<8.2f} {counts.std():<8.2f} "
              f"{np.median(counts):<8.1f} {counts.min():<6} {counts.max():<6}")
    
    # Overall
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    print(f"  {'Overall':<25} {active_counts.mean():<8.2f} {active_counts.std():<8.2f} "
          f"{np.median(active_counts):<8.1f} {active_counts.min():<6} {active_counts.max():<6}")


def main():
    parser = argparse.ArgumentParser(description='Plot prompt sparsity statistics')
    parser.add_argument('--mask_logits_npz', type=str, default=None,
                        help='Path to pre-extracted mask logits (.npz with "masks" and "labels")')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint (if not using mask_logits_npz)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config yaml')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--n_per_domain', type=int, default=100,
                        help='Number of samples per domain')
    parser.add_argument('--outdir', type=str, default='figures',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Threshold for counting active prototypes')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save extracted masks to npz')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FIG 5: PROMPT SPARSITY STATISTICS")
    print("=" * 60)
    
    # Load or extract masks
    if args.mask_logits_npz:
        print(f"\n[1] Loading pre-extracted masks from {args.mask_logits_npz}")
        npz = np.load(args.mask_logits_npz)
        masks = npz['masks']
        domain_labels = npz['labels']
    else:
        if not args.ckpt or not args.config or not args.data_dir:
            raise ValueError("Must provide either --mask_logits_npz or (--ckpt, --config, --data_dir)")
        
        print(f"\n[1] Extracting masks from checkpoint...")
        masks, domain_labels = extract_masks_from_checkpoint(
            args.ckpt, args.config, args.data_dir,
            n_per_domain=args.n_per_domain,
            device=args.device
        )
        
        # Optionally save for future use
        if args.save_masks:
            save_path = outdir / 'mask_logits.npz'
            np.savez(save_path, masks=masks, labels=domain_labels)
            print(f"  Saved masks to: {save_path}")
    
    print(f"  Masks shape: {masks.shape}")
    print(f"  Unique domains: {np.unique(domain_labels)}")
    
    # Compute active counts
    print(f"\n[2] Computing active prototype counts (threshold={args.threshold})...")
    active_counts = compute_active_counts(masks, threshold=args.threshold)
    print(f"  Active count range: [{active_counts.min()}, {active_counts.max()}]")
    
    # Print statistics
    print_statistics(active_counts, domain_labels)
    
    # Plot boxplot by domain
    print(f"\n[3] Generating boxplot...")
    plot_boxplot_by_domain(
        active_counts, domain_labels,
        outdir / 'prompt_active_count_by_domain.png'
    )
    
    # Plot histogram seen vs unseen
    print(f"\n[4] Generating histogram...")
    plot_histogram_seen_vs_unseen(
        active_counts, domain_labels,
        outdir / 'prompt_active_count_seen_vs_unseen.png'
    )
    
    # Save statistics to CSV
    import pandas as pd
    rows = []
    for i in range(len(masks)):
        rows.append({
            'domain': int(domain_labels[i]),
            'domain_name': DOMAIN_NAMES.get(domain_labels[i], f'D{domain_labels[i]}'),
            'active_count': int(active_counts[i]),
            'active_frac': active_counts[i] / masks.shape[1]
        })
    df = pd.DataFrame(rows)
    csv_path = outdir / 'prompt_sparsity.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved statistics: {csv_path}")
    
    print(f"\n[Done] Outputs saved to {outdir}/")


if __name__ == '__main__':
    main()

