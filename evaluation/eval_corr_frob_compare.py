#!/usr/bin/env python
"""Quick comparison of MASKED Spearman corr_frob + lagged_corr_frob across 3 normalizations.
Now with PROPER denormalization for all methods including centered_pit (inverse ECDF).
"""

import os
import sys
import json
import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from evaluation.eval_planA import compute_corr_frob, compute_lagged_corr_frob, save_corr_diagnostics


def load_model(ckpt_path, config_path, device='cuda'):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    return model.to(device).eval(), config


def sample_k_shot(model, prompt_data, n_samples=128, K=16, batch_size=32, device='cuda'):
    prompt_indices = np.random.choice(len(prompt_data), K, replace=False)
    prompts = prompt_data[prompt_indices]
    all_samples = []
    with torch.no_grad():
        prompt_tensor = torch.from_numpy(prompts).float().to(device)
        c_all, mask_all = model.get_learned_conditioning(prompt_tensor, return_mask=True)
        for i in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - i)
            indices = [(i + j) % K for j in range(bs)]
            samples = model.sample_log(cond=c_all[indices], batch_size=bs, ddim=False, 
                                       cfg_scale=1.0, mask=mask_all[indices])[0]
            all_samples.append(samples.cpu().numpy())
    return np.concatenate(all_samples, axis=0)


def denormalize_data(data_norm, data_dir, method_name):
    """
    Denormalize data based on method type.
    
    Args:
        data_norm: normalized data, shape (N, F, T)
        data_dir: directory containing normalization params
        method_name: 'zscore', 'zscore_winsor', or 'centered_pit'
    
    Returns:
        denormalized data, shape (N, F, T)
    """
    norm_stats_path = os.path.join(data_dir, 'norm_stats.npz')
    pit_params_path = os.path.join(data_dir, 'pit_params.json')
    
    if os.path.exists(norm_stats_path):
        # zscore: simple linear inverse
        norm_stats = np.load(norm_stats_path)
        mean, std = norm_stats['mean'], norm_stats['std']
        data_denorm = data_norm * std[:, np.newaxis] + mean[:, np.newaxis]
        print(f"    Denorm: zscore (mean/std)")
        return data_denorm
    
    elif os.path.exists(pit_params_path):
        with open(pit_params_path, 'r') as f:
            pit_params = json.load(f)
        
        if 'mean' in pit_params and 'std' in pit_params and isinstance(pit_params['mean'], list):
            # zscore_winsor: linear inverse
            mean, std = np.array(pit_params['mean']), np.array(pit_params['std'])
            data_denorm = data_norm * std[:, np.newaxis] + mean[:, np.newaxis]
            print(f"    Denorm: zscore_winsor (mean/std from pit_params)")
            return data_denorm
        
        elif 'ecdf_x' in pit_params and 'ecdf_y' in pit_params:
            # centered_pit: inverse ECDF
            D = data_norm.shape[1]
            data_denorm = np.zeros_like(data_norm)
            
            for d in range(D):
                ecdf_x_d = np.array(pit_params['ecdf_x'][str(d)])  # sorted original values
                ecdf_y_d = np.array(pit_params['ecdf_y'][str(d)])  # ECDF values [0, 1]
                
                # normalized is in [-1, 1], map to [0, 1] for ECDF inverse
                data_u = np.clip((data_norm[:, d, :] + 1) / 2, 0, 1)
                data_denorm[:, d, :] = np.interp(data_u.flatten(), ecdf_y_d, ecdf_x_d).reshape(data_u.shape)
            
            print(f"    Denorm: centered_pit (inverse ECDF)")
            return data_denorm
    
    print(f"    Warning: No denorm params found, returning normalized data")
    return data_norm


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # === Sanity Check ===
    print("\n" + "="*70)
    print("SANITY CHECK")
    print("="*70)
    np.random.seed(42)
    test_data = np.random.randn(50, 32, 6)
    
    result1, diag1 = compute_corr_frob(test_data, test_data, diagnostics=True)
    print(f"corr_frob(x, x) = {result1:.8f} (expected ~0)")
    assert result1 < 1e-6, f"Sanity check failed: {result1}"
    
    result2 = compute_lagged_corr_frob(test_data, test_data, L=4)
    print(f"lagged_corr_frob(x, x) D_lags = {[f'{d:.6f}' for d in result2['D_lags']]}")
    for lag, d in enumerate(result2['D_lags']):
        assert d < 1e-6, f"Lagged sanity check failed at lag {lag}: {d}"
    
    print("✓ All sanity checks PASSED!\n")
    
    # Configuration
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
    
    results = {}
    K = 16
    L = 4
    output_dir = 'evaluation/corr_diagnostics' 
    
    for name, cfg in configs.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")
        
        # Load model
        print(f"  Loading model...")
        model, _ = load_model(cfg['ckpt'], cfg['config'], device)
        
        # Load normalized data
        print(f"  Loading data from {cfg['data_dir']}...")
        unseen_eval_norm = np.load(os.path.join(cfg['data_dir'], 'unseen_eval_pool.npy'))  # (N, F, T)
        unseen_prompt_norm = np.load(os.path.join(cfg['data_dir'], 'unseen_prompt_pool.npy'))
        
        # Also try to load raw data for real samples (more accurate)
        dataset = np.load(os.path.join(cfg['data_dir'], 'dataset.npz'), allow_pickle=True)
        has_raw = 'X_unseen_eval_raw_NTD' in dataset
        
        n_real = len(unseen_eval_norm)
        print(f"  Unseen eval: {unseen_eval_norm.shape}, prompt: {unseen_prompt_norm.shape}")
        
        # Generate samples
        print(f"  Generating {n_real} samples with K={K} prompts...")
        np.random.seed(42)
        torch.manual_seed(42)
        gen_data_norm = sample_k_shot(model, unseen_prompt_norm, n_samples=n_real, K=K, device=device)
        print(f"  Generated: {gen_data_norm.shape}")
        
        # === DENORMALIZE ===
        print(f"  Denormalizing...")
        
        # For real data: use raw if available, otherwise denormalize
        if has_raw:
            real_data_denorm = np.transpose(dataset['X_unseen_eval_raw_NTD'], (0, 2, 1))  # (N, T, D) -> (N, F, T)
            print(f"    Real: using raw data directly")
        else:
            real_data_denorm = denormalize_data(unseen_eval_norm, cfg['data_dir'], name)
        
        # For generated data: always denormalize
        gen_data_denorm = denormalize_data(gen_data_norm, cfg['data_dir'], name)
        
        # Transpose to (N, T, F) for corr_frob
        real_data_tf = np.transpose(real_data_denorm, (0, 2, 1))
        gen_data_tf = np.transpose(gen_data_denorm, (0, 2, 1))
        
        print(f"  Final shapes: real {real_data_tf.shape}, gen {gen_data_tf.shape}")
        
        # Print data statistics for sanity check
        print(f"  Data stats (denormalized):")
        print(f"    Real dx: mean={real_data_tf[:,:,0].mean():.3f}, std={real_data_tf[:,:,0].std():.3f}")
        print(f"    Gen  dx: mean={gen_data_tf[:,:,0].mean():.3f}, std={gen_data_tf[:,:,0].std():.3f}")
        print(f"    Real v:  mean={real_data_tf[:,:,2].mean():.3f}, std={real_data_tf[:,:,2].std():.3f}")
        print(f"    Gen  v:  mean={gen_data_tf[:,:,2].mean():.3f}, std={gen_data_tf[:,:,2].std():.3f}")
        
        # Compute metrics
        corr_frob_masked, diag = compute_corr_frob(real_data_tf, gen_data_tf, diagnostics=True)
        lagged_result = compute_lagged_corr_frob(real_data_tf, gen_data_tf, L=L, diagnostics=True)
        
        # Save diagnostics
        combined_diag = {**diag, 'lagged_corr_frob': lagged_result}
        save_corr_diagnostics(combined_diag, output_dir, name, 'D3_unseen', K, seed=42)
        
        results[name] = {
            'corr_frob_masked': corr_frob_masked,
            'lagged_D': lagged_result['D_lags'],
            'kept_pairs': lagged_result['kept_pairs'],
        }
        
        print(f"\n  Results for {name}:")
        print(f"    corr_frob (masked, lag=0): {corr_frob_masked:.4f}")
        print(f"    lagged_corr_frob D_lags:   {[f'{d:.4f}' for d in lagged_result['D_lags']]}")
        
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY: corr_frob (masked Spearman, lag=0) - WITH PROPER DENORMALIZATION")
    print("="*90)
    print(f"{'Method':<15} {'corr_frob':<12}")
    print("-"*30)
    for name, r in results.items():
        print(f"{name:<15} {r['corr_frob_masked']:<12.4f}")
    
    print("\n" + "="*90)
    print("SUMMARY: lagged_corr_frob (masked Spearman, lags 0-4)")
    print("="*90)
    header = f"{'Method':<15}" + "".join([f"{'lag'+str(i):<10}" for i in range(L+1)])
    print(header)
    print("-"*90)
    for name, r in results.items():
        row = f"{name:<15}" + "".join([f"{d:<10.4f}" for d in r['lagged_D']])
        print(row)
    
    print(f"\nDiagnostics saved to: {output_dir}/")
    print("\nDone!")


if __name__ == '__main__':
    main()
