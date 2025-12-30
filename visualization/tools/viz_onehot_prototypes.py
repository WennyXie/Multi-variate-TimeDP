#!/usr/bin/env python3
"""
One-hot Prototype Semantics Visualization (TimeDP Fig.2 analogue)

Demonstrates each prototype corresponds to a distinct "semantic mode" by:
1. Constructing one-hot masks that activate only one prototype at a time
2. Generating samples conditioned on each one-hot mask
3. Visualizing the generated trajectories to show prototype-specific patterns

Usage:
    python tools/viz_onehot_prototypes.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --config configs/nuscenes_planA_zscore.yaml \
        --n_gen 32 \
        --proto_indices 0,1,2,3,4,5,6,7 \
        --output figures/onehot_prototypes.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.tools.prompt_viz.prompt_utils import load_model, get_model_info


CHANNEL_NAMES = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']


@torch.no_grad()
def generate_with_onehot_mask(
    model,
    proto_idx: int,
    n_samples: int,
    K: int,
    latent_dim: int,
    device: str = 'cuda',
    batch_size: int = 32,
    ddim: bool = False,
    ddim_steps: int = 50
) -> np.ndarray:
    """
    Generate samples with a one-hot prototype mask.
    
    Args:
        model: Loaded diffusion model
        proto_idx: Index of prototype to activate
        n_samples: Number of samples to generate
        K: Number of prototypes
        latent_dim: Latent dimension
        device: Device to use
        batch_size: Batch size for generation
        ddim: Whether to use DDIM sampling
        ddim_steps: Number of DDIM steps
        
    Returns:
        samples: (n_samples, D, T) generated samples
    """
    all_samples = []
    
    # Get the latents from the conditioning model
    if hasattr(model.cond_stage_model, 'impl'):
        latents = model.cond_stage_model.impl.latents  # (K, latent_dim)
    else:
        latents = model.cond_stage_model.latents
    
    for i in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - i)
        
        # Create one-hot mask: very negative for all except proto_idx
        mask = torch.full((bs, K), -1e9, device=device, dtype=torch.float32)
        mask[:, proto_idx] = 0.0  # Activate only this prototype
        
        # Create context: repeat latents for batch
        # Shape: (bs, K, latent_dim)
        context = latents.unsqueeze(0).expand(bs, -1, -1)
        
        # Generate samples
        samples, _ = model.sample_log(
            cond=context,
            batch_size=bs,
            ddim=ddim,
            ddim_steps=ddim_steps,
            mask=mask,
            cfg_scale=1.0
        )
        
        all_samples.append(samples.cpu().numpy())
    
    return np.concatenate(all_samples, axis=0)


def dx_dy_to_trajectory(dx: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert displacement to trajectory."""
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y


def plot_prototype_trajectories(
    samples: np.ndarray,
    proto_idx: int,
    ax: plt.Axes,
    color: str = 'blue',
    max_trajectories: int = 20
):
    """Plot trajectories for one prototype."""
    n_show = min(len(samples), max_trajectories)
    
    # Random subset
    indices = np.random.choice(len(samples), n_show, replace=False)
    
    for i, idx in enumerate(indices):
        dx = samples[idx, 0, :]  # Channel 0 = dx
        dy = samples[idx, 1, :]  # Channel 1 = dy
        x, y = dx_dy_to_trajectory(dx, dy)
        
        alpha = 0.3 + 0.4 * (i / n_show)
        ax.plot(x, y, color=color, alpha=alpha, linewidth=1.0)
        ax.scatter(x[0], y[0], c=color, s=20, marker='o', alpha=alpha, zorder=5)
    
    # Plot mean trajectory
    mean_dx = samples[:, 0, :].mean(axis=0)
    mean_dy = samples[:, 1, :].mean(axis=0)
    mean_x, mean_y = dx_dy_to_trajectory(mean_dx, mean_dy)
    ax.plot(mean_x, mean_y, color='black', linewidth=2.5, linestyle='--', label='Mean')
    ax.scatter(mean_x[0], mean_y[0], c='black', s=50, marker='o', zorder=10)
    
    ax.set_title(f'Prototype {proto_idx}', fontsize=11)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)


def plot_prototype_channels(
    samples: np.ndarray,
    proto_idx: int,
    ax: plt.Axes,
    channels: List[int] = [2, 4, 5],  # v, yaw_rate, curvature
    channel_names: List[str] = None
):
    """Plot channel statistics for one prototype."""
    if channel_names is None:
        channel_names = [CHANNEL_NAMES[c] for c in channels]
    
    T = samples.shape[2]
    time = np.arange(T)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(channels)))
    
    for i, (ch, name) in enumerate(zip(channels, channel_names)):
        ch_data = samples[:, ch, :]  # (N, T)
        mean = ch_data.mean(axis=0)
        std = ch_data.std(axis=0)
        
        ax.plot(time, mean, color=colors[i], linewidth=1.5, label=name)
        ax.fill_between(time, mean - std, mean + std, color=colors[i], alpha=0.2)
    
    ax.set_xlabel('Time step', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_onehot_grid(
    all_samples: dict,
    output_path: Path,
    title: str = 'One-hot Prototype Semantics'
):
    """
    Plot a grid of prototype visualizations.
    
    Args:
        all_samples: Dict mapping proto_idx -> (n_samples, D, T) array
        output_path: Where to save figure
        title: Plot title
    """
    n_protos = len(all_samples)
    
    # Determine grid layout
    n_cols = min(4, n_protos)
    n_rows = (n_protos + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_protos))
    
    for i, (proto_idx, samples) in enumerate(sorted(all_samples.items())):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        plot_prototype_trajectories(samples, proto_idx, ax, color=colors[i])
    
    #plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def plot_onehot_detailed(
    all_samples: dict,
    output_path: Path,
    title: str = 'One-hot Prototype Details'
):
    """
    Plot detailed view with trajectories and channel plots.
    """
    n_protos = len(all_samples)
    
    fig = plt.figure(figsize=(12, 3 * n_protos))
    gs = GridSpec(n_protos, 2, figure=fig, width_ratios=[1, 1.5])
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_protos))
    
    for i, (proto_idx, samples) in enumerate(sorted(all_samples.items())):
        # Left: Trajectory plot
        ax_traj = fig.add_subplot(gs[i, 0])
        plot_prototype_trajectories(samples, proto_idx, ax_traj, color=colors[i])
        
        # Right: Channel plots
        ax_ch = fig.add_subplot(gs[i, 1])
        plot_prototype_channels(samples, proto_idx, ax_ch)
        ax_ch.set_title(f'Proto {proto_idx} Channels', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def compute_prototype_statistics(all_samples: dict) -> dict:
    """Compute statistics for each prototype's generated samples."""
    stats = {}
    
    for proto_idx, samples in all_samples.items():
        # Compute trajectory statistics
        dx = samples[:, 0, :]
        dy = samples[:, 1, :]
        
        # Final displacement
        final_x = dx.sum(axis=1)
        final_y = dy.sum(axis=1)
        final_dist = np.sqrt(final_x**2 + final_y**2)
        
        # Heading (final direction)
        final_heading = np.arctan2(final_y, final_x) * 180 / np.pi
        
        # Speed statistics
        v = samples[:, 2, :]
        
        stats[proto_idx] = {
            'final_dist_mean': final_dist.mean(),
            'final_dist_std': final_dist.std(),
            'final_heading_mean': final_heading.mean(),
            'final_heading_std': final_heading.std(),
            'speed_mean': v.mean(),
            'speed_std': v.std(),
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='One-hot Prototype Semantics Visualization')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--n_gen', type=int, default=64, help='Samples per prototype')
    parser.add_argument('--proto_indices', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated prototype indices to visualize')
    parser.add_argument('--output', type=str, default='figures/onehot_prototypes.png',
                        help='Output path for figure')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ddim', action='store_true', help='Use DDIM sampling')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse prototype indices
    proto_indices = [int(p.strip()) for p in args.proto_indices.split(',')]
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ONE-HOT PROTOTYPE SEMANTICS VISUALIZATION")
    print("=" * 60)
    
    # Load model
    print(f"\n[1] Loading model from {args.ckpt}")
    model = load_model(args.ckpt, args.config, args.device)
    model_info = get_model_info(model)
    
    K = model_info.get('num_latents', 16)
    latent_dim = model_info.get('latent_dim', 32)
    
    print(f"    Num latents (K): {K}")
    print(f"    Latent dim: {latent_dim}")
    
    # Validate prototype indices
    proto_indices = [p for p in proto_indices if p < K]
    if len(proto_indices) == 0:
        proto_indices = list(range(min(8, K)))
    
    print(f"    Prototypes to visualize: {proto_indices}")
    
    # Generate samples for each prototype
    print(f"\n[2] Generating samples with one-hot masks...")
    all_samples = {}
    
    for proto_idx in proto_indices:
        print(f"    Prototype {proto_idx}...", end=' ', flush=True)
        samples = generate_with_onehot_mask(
            model,
            proto_idx=proto_idx,
            n_samples=args.n_gen,
            K=K,
            latent_dim=latent_dim,
            device=args.device,
            batch_size=args.batch_size,
            ddim=args.ddim,
            ddim_steps=args.ddim_steps
        )
        all_samples[proto_idx] = samples
        print(f"shape={samples.shape}")
    
    # Plot grid view
    print(f"\n[3] Generating visualizations...")
    plot_onehot_grid(
        all_samples, output_path,
        title='One-hot Prototype Trajectories'
    )
    
    # Plot detailed view
    detailed_path = output_path.parent / f'{output_path.stem}_detailed.png'
    plot_onehot_detailed(
        all_samples, detailed_path,
        title='One-hot Prototype Details'
    )
    
    # Compute and print statistics
    print(f"\n[4] Prototype Statistics:")
    stats = compute_prototype_statistics(all_samples)
    
    print(f"    {'Proto':<6} {'FinalDist':<12} {'Heading':<12} {'Speed':<12}")
    print(f"    {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
    for proto_idx, s in sorted(stats.items()):
        print(f"    {proto_idx:<6} "
              f"{s['final_dist_mean']:.1f}±{s['final_dist_std']:.1f}m    "
              f"{s['final_heading_mean']:.0f}±{s['final_heading_std']:.0f}°    "
              f"{s['speed_mean']:.1f}±{s['speed_std']:.1f}")
    
    # Save individual prototype plots
    print(f"\n[5] Saving individual prototype plots...")
    for proto_idx, samples in all_samples.items():
        proto_path = output_path.parent / f'onehot_proto_{proto_idx}.png'
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, K))
        plot_prototype_trajectories(samples, proto_idx, ax, color=colors[proto_idx])
        plt.tight_layout()
        plt.savefig(proto_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f"    Saved {len(all_samples)} individual plots")
    
    print(f"\n[Done] Outputs saved to {output_path.parent}/")


if __name__ == '__main__':
    main()

