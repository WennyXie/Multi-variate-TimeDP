#!/usr/bin/env python3
"""
Common utilities for prompt visualization scripts.

Provides:
- Model loading from checkpoint
- Prompt/mask extraction from data
- Data loading helpers
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def load_model(ckpt_path: str, config_path: str, device: str = 'cuda') -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        ckpt_path: Path to .ckpt file
        config_path: Path to .yaml config file
        device: 'cuda' or 'cpu'
        
    Returns:
        Loaded model in eval mode
    """
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


def get_model_info(model: nn.Module) -> Dict:
    """Get model configuration info."""
    info = {
        'has_cond_stage': hasattr(model, 'cond_stage_model') and model.cond_stage_model is not None,
    }
    
    if info['has_cond_stage']:
        cond_model = model.cond_stage_model
        info['cond_model_type'] = type(cond_model).__name__
        
        # Check for ProtoRouter or DomainUnifiedPrototyper
        if hasattr(cond_model, 'impl'):
            # ProtoRouter
            info['impl_type'] = type(cond_model.impl).__name__
            info['num_latents'] = cond_model.num_latents
            info['latent_dim'] = cond_model.latent_dim
        elif hasattr(cond_model, 'num_latents'):
            # Direct DomainUnifiedPrototyper
            info['impl_type'] = type(cond_model).__name__
            info['num_latents'] = cond_model.num_latents
            info['latent_dim'] = cond_model.latent_dim
        else:
            info['impl_type'] = 'unknown'
            info['num_latents'] = None
            info['latent_dim'] = None
    
    return info


@torch.no_grad()
def extract_prompts(
    model: nn.Module,
    data: np.ndarray,
    batch_size: int = 64,
    device: str = 'cuda',
    return_context: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract prompt masks from input data.
    
    Args:
        model: Loaded model
        data: Input data of shape (N, D, T) or (N, T, D) - will be auto-detected
        batch_size: Batch size for inference
        device: Device to use
        return_context: Whether to also return context vectors
        
    Returns:
        masks: (N, K) mask logits
        contexts: (N, K, latent_dim) context vectors if return_context=True, else None
    """
    # Auto-detect and fix shape: model expects (N, D, T) where D=6, T=32
    if data.shape[1] == 32 and data.shape[2] == 6:
        # Data is (N, T, D), transpose to (N, D, T)
        data = data.transpose(0, 2, 1)
    
    N = len(data)
    all_masks = []
    all_contexts = [] if return_context else None
    
    for i in range(0, N, batch_size):
        batch = data[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)
        
        # Get conditioning (context and mask)
        context, mask = model.get_learned_conditioning(batch_tensor, return_mask=True)
        
        all_masks.append(mask.cpu().numpy())
        if return_context:
            all_contexts.append(context.cpu().numpy())
    
    masks = np.concatenate(all_masks, axis=0)
    contexts = np.concatenate(all_contexts, axis=0) if return_context else None
    
    return masks, contexts


def load_domain_data(
    data_dir: Path,
    split: str = 'seen_train',
    domains: List[int] = None,
    n_per_domain: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data samples from specified domains.
    
    Args:
        data_dir: Path to data directory
        split: 'seen_train', 'seen_eval', or 'unseen_eval'
        domains: List of domain indices to load
        n_per_domain: Number of samples per domain
        
    Returns:
        data: (N_total, D, T) array
        domain_labels: (N_total,) array of domain indices
    """
    data_dir = Path(data_dir)
    
    all_data = []
    all_labels = []
    
    def fix_shape(arr):
        """Fix shape to (N, D, T) where D=6, T=32."""
        if arr.shape[1] == 32 and arr.shape[2] == 6:
            return arr.transpose(0, 2, 1)
        return arr
    
    if split == 'unseen_eval':
        # Unseen domain (domain 3)
        pool_path = data_dir / 'unseen_eval_pool.npy'
        if pool_path.exists():
            pool = fix_shape(np.load(pool_path))
            n = min(n_per_domain, len(pool))
            indices = np.random.choice(len(pool), n, replace=False)
            all_data.append(pool[indices])
            all_labels.append(np.full(n, 3))
    else:
        # Seen domains - try multiple file naming conventions
        pool = None
        domain_ids = None
        
        # Convention 1: train.npy + domain_train.npy
        train_path = data_dir / 'train.npy'
        domain_path = data_dir / 'domain_train.npy'
        if train_path.exists():
            pool = fix_shape(np.load(train_path))
            if domain_path.exists():
                domain_ids = np.load(domain_path)
            else:
                domain_ids = np.zeros(len(pool), dtype=int)
        
        # Convention 2: seen_train_pool.npy + seen_train_domain_ids.npy
        if pool is None:
            pool_name = 'seen_train_pool.npy' if split == 'seen_train' else 'seen_eval_pool.npy'
            pool_path = data_dir / pool_name
            if pool_path.exists():
                pool = fix_shape(np.load(pool_path))
                labels_path = data_dir / pool_name.replace('pool.npy', 'domain_ids.npy')
                if labels_path.exists():
                    domain_ids = np.load(labels_path)
                else:
                    domain_ids = np.zeros(len(pool), dtype=int)
        
        # Convention 3: dataset.npz
        if pool is None:
            npz_path = data_dir / 'dataset.npz'
            if npz_path.exists():
                npz = np.load(npz_path)
                if 'train' in npz:
                    pool = fix_shape(npz['train'])
                    domain_ids = npz.get('domain_train', np.zeros(len(pool), dtype=int))
        
        if pool is not None:
            if domains is None:
                domains = [0, 1, 2]  # Default seen domains
            
            for d in domains:
                if d == 3:
                    continue  # Unseen domain, handled separately
                mask = domain_ids == d
                domain_pool = pool[mask]
                if len(domain_pool) > 0:
                    n = min(n_per_domain, len(domain_pool))
                    indices = np.random.choice(len(domain_pool), n, replace=False)
                    all_data.append(domain_pool[indices])
                    all_labels.append(np.full(n, d))
    
    if len(all_data) == 0:
        raise ValueError(f"No data found for split={split}, domains={domains}")
    
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return data, labels


def sparsify_mask(mask: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Apply sparsification to mask logits (same as model does).
    
    Args:
        mask: (N, K) mask logits
        threshold: Values <= threshold become -inf
        
    Returns:
        Sparsified mask with -inf for inactive prototypes
    """
    sparse_mask = mask.copy()
    sparse_mask[sparse_mask <= threshold] = -1e9  # Use large negative instead of -inf for viz
    return sparse_mask


def softmax_mask(mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply softmax to mask logits.
    
    Args:
        mask: (N, K) mask logits
        temperature: Softmax temperature
        
    Returns:
        (N, K) normalized weights
    """
    mask = mask / temperature
    exp_mask = np.exp(mask - mask.max(axis=-1, keepdims=True))
    return exp_mask / exp_mask.sum(axis=-1, keepdims=True)

