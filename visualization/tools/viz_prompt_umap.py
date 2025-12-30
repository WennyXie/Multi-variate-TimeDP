#!/usr/bin/env python3
"""
Prompt-space UMAP/t-SNE Visualization (TimeDP Fig.4/5 analogue)

Shows prompts form domain-separated clusters and unseen prompts lie near related seen domains.

Embedding method:
- Uses raw mask logits (default) as the prompt representation
- Alternative: softmax(mask) for normalized weights

Usage:
    python tools/viz_prompt_umap.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --config configs/nuscenes_planA_zscore.yaml \
        --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
        --domains 0,1,2,3 \
        --n_per_domain 100 \
        --output figures/prompt_umap.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.tools.prompt_viz.prompt_utils import (
    load_model, get_model_info, extract_prompts,
    load_domain_data, softmax_mask
)

# Try to import UMAP, fall back to t-SNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, will use sklearn t-SNE")

from sklearn.manifold import TSNE


DOMAIN_NAMES = {
    0: 'boston-seaport',
    1: 'singapore-onenorth', 
    2: 'singapore-queenstown',
    3: 'singapore-hollandvillage (unseen)'
}

DOMAIN_COLORS = {
    0: '#1f77b4',  # Blue
    1: '#ff7f0e',  # Orange
    2: '#2ca02c',  # Green
    3: '#d62728',  # Red (unseen)
}

DOMAIN_MARKERS = {
    0: 'o',
    1: 's',
    2: '^',
    3: '*',  # Star for unseen
}


def compute_embedding(
    data: np.ndarray,
    method: str = 'umap',
    random_state: int = 42
) -> np.ndarray:
    """
    Compute 2D embedding using UMAP or t-SNE.
    
    Args:
        data: (N, K) prompt vectors
        method: 'umap' or 'tsne'
        random_state: Random seed
        
    Returns:
        coords: (N, 2) 2D coordinates
    """
    if method == 'umap' and HAS_UMAP:
        print("  Using UMAP...")
        reducer = UMAP(
            n_neighbors=min(15, len(data) - 1),
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=random_state
        )
    else:
        print("  Using t-SNE...")
        perp = min(30, len(data) // 4)
        reducer = TSNE(
            n_components=2,
            perplexity=max(5, perp),
            random_state=random_state,
            n_iter=1000
        )
    
    coords = reducer.fit_transform(data)
    return coords


def plot_prompt_umap(
    coords: np.ndarray,
    domain_labels: np.ndarray,
    output_path: Path,
    method: str = 'umap',
    title: str = 'Prompt Space Embedding',
    show_centroids: bool = True
):
    """
    Plot 2D embedding of prompts colored by domain.
    
    Args:
        coords: (N, 2) 2D coordinates
        domain_labels: (N,) domain indices
        output_path: Where to save figure
        method: 'umap' or 'tsne' (for axis labels)
        title: Plot title
        show_centroids: Whether to show domain centroids
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_domains = np.unique(domain_labels)
    centroids = {}
    
    # Plot each domain
    for d in unique_domains:
        mask = domain_labels == d
        d_coords = coords[mask]
        
        # Compute centroid
        centroid = d_coords.mean(axis=0)
        centroids[d] = centroid
        
        # Determine if unseen
        is_unseen = (d == 3)
        
        ax.scatter(
            d_coords[:, 0], d_coords[:, 1],
            c=DOMAIN_COLORS.get(d, 'gray'),
            marker=DOMAIN_MARKERS.get(d, 'o'),
            s=100 if is_unseen else 60,
            alpha=0.8 if is_unseen else 0.6,
            label=DOMAIN_NAMES.get(d, f'Domain {d}'),
            edgecolors='black' if is_unseen else 'white',
            linewidths=1.5 if is_unseen else 0.5
        )
    
    # Plot centroids
    if show_centroids:
        for d, centroid in centroids.items():
            ax.scatter(
                centroid[0], centroid[1],
                c=DOMAIN_COLORS.get(d, 'gray'),
                marker='X',
                s=200,
                edgecolors='black',
                linewidths=2,
                zorder=10
            )
            ax.annotate(
                f'D{d}', 
                (centroid[0], centroid[1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
    
    # Labels and styling
    method_name = 'UMAP' if (method == 'umap' and HAS_UMAP) else 't-SNE'
    ax.set_xlabel(f'{method_name} 1', fontsize=11)
    ax.set_ylabel(f'{method_name} 2', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    
    return centroids


def plot_prompt_umap_simple(
    coords: np.ndarray,
    domain_labels: np.ndarray,
    output_path: Path,
    method: str = 'umap'
):
    """
    Simple UMAP plot matching TimeDP Fig.4 style.
    
    Only plots points colored by domain with clean legend.
    No centroids, no annotations, no title.
    
    Args:
        coords: (N, 2) 2D coordinates
        domain_labels: (N,) domain indices
        output_path: Where to save figure
        method: 'umap' or 'tsne' (for axis labels)
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    unique_domains = np.unique(domain_labels)
    
    # Simple domain name mapping (short names)
    SIMPLE_NAMES = {
        0: 'D0',
        1: 'D1', 
        2: 'D2',
        3: 'D3'
    }
    
    # Plot each domain - simple circles only
    for d in unique_domains:
        mask = domain_labels == d
        d_coords = coords[mask]
        
        ax.scatter(
            d_coords[:, 0], d_coords[:, 1],
            c=DOMAIN_COLORS.get(d, 'gray'),
            marker='o',
            s=50,
            alpha=0.7,
            label=SIMPLE_NAMES.get(d, f'Domain {d}'),
            edgecolors='white',
            linewidths=0.3
        )
    
    # Labels and styling - minimal
    method_name = 'UMAP' if (method == 'umap' and HAS_UMAP) else 't-SNE'
    # ax.set_xlabel(f'{method_name} 1', fontsize=11)
    # ax.set_ylabel(f'{method_name} 2', fontsize=11)
    # No title
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def compute_domain_separation(
    coords: np.ndarray,
    domain_labels: np.ndarray
) -> dict:
    """
    Compute metrics for domain separation in embedding space.
    """
    from scipy.spatial.distance import cdist
    
    unique_domains = np.unique(domain_labels)
    
    # Compute within-domain and between-domain distances
    within_dists = {}
    for d in unique_domains:
        d_coords = coords[domain_labels == d]
        if len(d_coords) > 1:
            dists = cdist(d_coords, d_coords)
            within_dists[d] = dists[np.triu_indices(len(d_coords), k=1)].mean()
    
    # Between domain distances (centroid to centroid)
    centroids = {}
    for d in unique_domains:
        centroids[d] = coords[domain_labels == d].mean(axis=0)
    
    between_dists = {}
    for i, d1 in enumerate(unique_domains):
        for d2 in unique_domains[i+1:]:
            dist = np.linalg.norm(centroids[d1] - centroids[d2])
            between_dists[(d1, d2)] = dist
    
    return {
        'within_domain': within_dists,
        'between_domain': between_dists,
        'centroids': centroids
    }


def main():
    parser = argparse.ArgumentParser(description='Prompt UMAP/t-SNE Visualization')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--domains', type=str, default='0,1,2,3',
                        help='Comma-separated domain indices')
    parser.add_argument('--n_per_domain', type=int, default=100,
                        help='Number of samples per domain')
    parser.add_argument('--output', type=str, default='figures/prompt_umap.png',
                        help='Output path for figure')
    parser.add_argument('--method', type=str, default='umap',
                        choices=['umap', 'tsne'], help='Embedding method')
    parser.add_argument('--representation', type=str, default='raw',
                        choices=['raw', 'softmax'], help='How to represent masks')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_centroids', action='store_true',
                        help='Do not show domain centroids')
    parser.add_argument('--simple', action='store_true',
                        help='Use simple TimeDP Fig.4 style (no centroids, no title, clean)')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Parse domains
    domains = [int(d.strip()) for d in args.domains.split(',')]
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PROMPT UMAP/t-SNE VISUALIZATION")
    print("=" * 60)
    
    # Load model
    print(f"\n[1] Loading model from {args.ckpt}")
    model = load_model(args.ckpt, args.config, args.device)
    model_info = get_model_info(model)
    print(f"    Num latents (K): {model_info.get('num_latents', 'unknown')}")
    
    # Load data for each domain
    print(f"\n[2] Loading data from {args.data_dir}")
    print(f"    Domains: {domains}, N per domain: {args.n_per_domain}")
    
    all_data = []
    all_labels = []
    
    # Load seen domains
    seen_domains = [d for d in domains if d != 3]
    if seen_domains:
        data, labels = load_domain_data(
            Path(args.data_dir),
            split='seen_train',
            domains=seen_domains,
            n_per_domain=args.n_per_domain
        )
        all_data.append(data)
        all_labels.append(labels)
    
    # Load unseen domain
    if 3 in domains:
        data, labels = load_domain_data(
            Path(args.data_dir),
            split='unseen_eval',
            domains=[3],
            n_per_domain=args.n_per_domain
        )
        all_data.append(data)
        all_labels.append(labels)
    
    data = np.concatenate(all_data, axis=0)
    domain_labels = np.concatenate(all_labels, axis=0)
    
    print(f"    Total samples: {len(data)}")
    for d in np.unique(domain_labels):
        print(f"      Domain {d}: {np.sum(domain_labels == d)} samples")
    
    # Extract prompts
    print(f"\n[3] Extracting prompt masks...")
    masks, _ = extract_prompts(model, data, device=args.device)
    print(f"    Mask shape: {masks.shape}")
    
    # Convert to representation
    if args.representation == 'softmax':
        print("    Applying softmax to masks...")
        prompt_vectors = softmax_mask(masks)
    else:
        print("    Using raw mask logits...")
        prompt_vectors = masks
    
    # Compute embedding
    print(f"\n[4] Computing {args.method.upper()} embedding...")
    coords = compute_embedding(prompt_vectors, method=args.method, random_state=args.seed)
    
    # Plot
    print(f"\n[5] Generating visualization...")
    method_name = 'UMAP' if (args.method == 'umap' and HAS_UMAP) else 't-SNE'
    
    if args.simple:
        # Simple TimeDP Fig.4 style
        simple_path = output_path.parent / 'prompt_umap_domains_only.png'
        plot_prompt_umap_simple(
            coords, domain_labels, simple_path,
            method=args.method
        )
        centroids = {}  # Skip centroid computation for simple mode
    else:
        centroids = plot_prompt_umap(
            coords, domain_labels, output_path,
            method=args.method,
            title=f'Prompt Space ({method_name}, {args.representation})',
            show_centroids=not args.no_centroids
        )
    
    # Compute and print separation metrics
    print(f"\n[6] Domain Separation Metrics:")
    sep_metrics = compute_domain_separation(coords, domain_labels)
    
    print("    Within-domain distances:")
    for d, dist in sep_metrics['within_domain'].items():
        print(f"      Domain {d}: {dist:.2f}")
    
    print("    Between-domain distances (centroid):")
    for (d1, d2), dist in sep_metrics['between_domain'].items():
        print(f"      D{d1} <-> D{d2}: {dist:.2f}")
    
    # Check if unseen domain is close to any seen domain
    if centroids and 3 in centroids:
        print("\n    Unseen domain (D3) nearest seen domain:")
        unseen_centroid = centroids[3]
        min_dist = float('inf')
        nearest = None
        for d in [0, 1, 2]:
            if d in centroids:
                dist = np.linalg.norm(unseen_centroid - centroids[d])
                if dist < min_dist:
                    min_dist = dist
                    nearest = d
        if nearest is not None:
            print(f"      Nearest: Domain {nearest} (dist={min_dist:.2f})")
    
    # Save coordinates to CSV
    csv_path = output_path.with_suffix('.csv')
    import pandas as pd
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'domain': domain_labels
    })
    df.to_csv(csv_path, index=False)
    print(f"\n    Saved coordinates: {csv_path}")
    
    print(f"\n[Done] Outputs saved to {output_path.parent}/")


if __name__ == '__main__':
    main()

