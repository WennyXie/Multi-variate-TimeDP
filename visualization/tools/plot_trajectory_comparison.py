#!/usr/bin/env python
"""Plot 2D trajectory comparison: real K-shot prompts vs generated samples."""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def load_model(ckpt_path, config_path, device='cuda'):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    return model


def sample_k_shot(model, prompt_data, n_samples=16, K=16, batch_size=32, device='cuda'):
    prompt_indices = np.random.choice(len(prompt_data), K, replace=False)
    prompts = prompt_data[prompt_indices]
    all_samples = []
    with torch.no_grad():
        prompt_tensor = torch.from_numpy(prompts).float().to(device)
        c_all, mask_all = model.get_learned_conditioning(prompt_tensor, return_mask=True)
        for i in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - i)
            indices = [(i + j) % K for j in range(bs)]
            c = c_all[indices]
            mask = mask_all[indices]
            samples = model.sample_log(cond=c, batch_size=bs, ddim=False, cfg_scale=1.0, mask=mask)[0]
            all_samples.append(samples.cpu().numpy())
    return np.concatenate(all_samples, axis=0), prompts, prompt_indices


def dx_dy_to_trajectory(dx, dy):
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y


def plot_and_save_single(name, gen_denorm, prompt_denorm, output_dir):
    """Plot and save immediately for one normalization method."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract dx, dy (first two channels), shape: (N, F, T)
    gen_dx = gen_denorm[:, 0, :]
    gen_dy = gen_denorm[:, 1, :]
    prompt_dx = prompt_denorm[:, 0, :]
    prompt_dy = prompt_denorm[:, 1, :]
    
    # Plot real prompts (green)
    for i in range(len(prompt_dx)):
        x, y = dx_dy_to_trajectory(prompt_dx[i], prompt_dy[i])
        label = 'Real Prompts' if i == 0 else None
        ax.plot(x, y, color='green', alpha=0.7, linewidth=1.5, label=label)
        ax.scatter(x[0], y[0], c='green', s=30, marker='o', zorder=5, alpha=0.7)
    
    # Plot generated (red)
    for i in range(len(gen_dx)):
        x, y = dx_dy_to_trajectory(gen_dx[i], gen_dy[i])
        label = 'Generated' if i == 0 else None
        ax.plot(x, y, color='red', alpha=0.5, linewidth=1.0, label=label)
        ax.scatter(x[0], y[0], c='red', s=20, marker='o', zorder=5, alpha=0.5)
    
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_title(f'{name}: Real vs Generated Trajectories (Unseen D3)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f'trajectory_{name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def process_one_model(name, cfg, K, N_gen, device, output_dir):
    """Process a single model and save its plot immediately."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model...")
    model = load_model(cfg['ckpt'], cfg['config'], device)
    
    # Load data
    print(f"Loading data from {cfg['data_dir']}...")
    unseen_prompt = np.load(os.path.join(cfg['data_dir'], 'unseen_prompt_pool.npy'))
    dataset = np.load(os.path.join(cfg['data_dir'], 'dataset.npz'), allow_pickle=True)
    
    # Check if raw data available
    has_raw = 'X_unseen_prompt_raw_NTD' in dataset
    if has_raw:
        unseen_prompt_raw = np.transpose(dataset['X_unseen_prompt_raw_NTD'], (0, 2, 1))
        print(f"  Raw data available")
    
    print(f"  Unseen prompt: {unseen_prompt.shape}")
    
    # Generate samples
    print(f"Generating {N_gen} samples with K={K} prompts...")
    np.random.seed(42)
    torch.manual_seed(42)
    gen_data, prompt_data, prompt_indices = sample_k_shot(model, unseen_prompt, n_samples=N_gen, K=K, device=device)
    print(f"  Generated: {gen_data.shape}")
    
    # Denormalize
    norm_stats_path = os.path.join(cfg['data_dir'], 'norm_stats.npz')
    pit_params_path = os.path.join(cfg['data_dir'], 'pit_params.json')
    
    if os.path.exists(norm_stats_path):
        # zscore
        norm_stats = np.load(norm_stats_path)
        mean, std = norm_stats['mean'], norm_stats['std']
        gen_denorm = gen_data * std[:, np.newaxis] + mean[:, np.newaxis]
        prompt_denorm = prompt_data * std[:, np.newaxis] + mean[:, np.newaxis]
        print(f"  Denorm: norm_stats.npz (zscore)")
    elif os.path.exists(pit_params_path):
        with open(pit_params_path, 'r') as f:
            pit_params = json.load(f)
        
        if 'mean' in pit_params and 'std' in pit_params and isinstance(pit_params['mean'], list):
            # zscore_winsor
            mean, std = np.array(pit_params['mean']), np.array(pit_params['std'])
            gen_denorm = gen_data * std[:, np.newaxis] + mean[:, np.newaxis]
            prompt_denorm = unseen_prompt_raw[prompt_indices] if has_raw else prompt_data * std[:, np.newaxis] + mean[:, np.newaxis]
            print(f"  Denorm: pit_params.json (zscore_winsor)")
        elif 'ecdf_x' in pit_params and 'ecdf_y' in pit_params:
            # centered_pit - proper inverse ECDF
            # ecdf_x[d] = sorted original values, ecdf_y[d] = ECDF values in [0,1]
            # normalized = 2 * ecdf_y - 1 (maps to [-1, 1])
            # inverse: ecdf_y = (normalized + 1) / 2, then interp to get original
            D = gen_data.shape[1]
            gen_denorm = np.zeros_like(gen_data)
            prompt_denorm_calc = np.zeros_like(prompt_data)
            
            for d in range(D):
                ecdf_x_d = np.array(pit_params['ecdf_x'][str(d)])  # sorted original values
                ecdf_y_d = np.array(pit_params['ecdf_y'][str(d)])  # ECDF values [0, 1]
                
                # Gen: [-1, 1] -> [0, 1] -> inverse ECDF -> original
                gen_u = np.clip((gen_data[:, d, :] + 1) / 2, 0, 1)  # map [-1,1] to [0,1]
                gen_denorm[:, d, :] = np.interp(gen_u.flatten(), ecdf_y_d, ecdf_x_d).reshape(gen_u.shape)
                
                # Prompt: same process
                prompt_u = np.clip((prompt_data[:, d, :] + 1) / 2, 0, 1)
                prompt_denorm_calc[:, d, :] = np.interp(prompt_u.flatten(), ecdf_y_d, ecdf_x_d).reshape(prompt_u.shape)
            
            # Use raw data for prompts if available (more accurate), otherwise use calculated
            prompt_denorm = unseen_prompt_raw[prompt_indices] if has_raw else prompt_denorm_calc
            print(f"  Denorm: pit_params.json (centered_pit, inverse ECDF)")
        else:
            prompt_denorm = unseen_prompt_raw[prompt_indices] if has_raw else prompt_data
            gen_denorm = gen_data
            print(f"  Denorm: unknown format, using raw/normalized")
    else:
        gen_denorm = gen_data
        prompt_denorm = unseen_prompt_raw[prompt_indices] if has_raw else prompt_data
        print(f"  Warning: no denorm params found")
    
    # Plot and save
    plot_and_save_single(name, gen_denorm, prompt_denorm, output_dir)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    print(f"✓ Done with {name}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = './figure'
    os.makedirs(output_dir, exist_ok=True)
    
    configs = {
        'zscore': {
            'ckpt': './logs/nuscenes_planA/nuscenes_planA_timedp/nuscenes_planA_timedp_32_nl_16_lr1.0e-04_bs32_pam_seed0/checkpoints/000123-0.1384.ckpt',
            'config': './configs/nuscenes_planA_timedp.yaml',
            'data_dir': 'data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0',
        },
        'centered_pit': {
            'ckpt': './logs/nuscenes_planA_pit/nuscenes_planA_centered_pit/nuscenes_planA_centered_pit_32_nl_16_lr1.0e-04_bs32_pam_seed0/checkpoints/000101-0.1442.ckpt',
            'config': './configs/nuscenes_planA_centered_pit.yaml',
            'data_dir': 'data/nuscenes_planA_T32_centered_pit_seed0',
        }
    }
    
    K = 16
    N_gen = 16
    
    for name, cfg in configs.items():
        try:
            process_one_model(name, cfg, K, N_gen, device, output_dir)
        except Exception as e:
            print(f"✗ Error with {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All done! Check ./figure/ for saved plots.")
    print("="*60)


if __name__ == '__main__':
    main()
