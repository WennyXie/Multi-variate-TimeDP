#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick pipeline test for nuScenes multivariate TimeDP.
Verifies that forward + backward pass works without errors.
"""

import sys
import os
os.environ['DATA_ROOT'] = '/u/pyt9dp/TimeCraft/TimeDP/data'

import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

# Add TimeDP to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ldm.util import instantiate_from_config
from ldm.data.nuscenes_planA import NuScenesPlanADataModule


def test_data_loading():
    """Test data loading."""
    print("="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    dm = NuScenesPlanADataModule(
        data_dir='data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0',
        batch_size=16
    )
    dm.prepare_data()
    dm.setup()
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch 'context' shape: {batch['context'].shape}")
    print(f"Batch 'data_key': {batch['data_key']}")
    
    # TimeDP expects (B, T, F) input, get_input will convert to (B, F, T)
    # Note: batch size might be smaller if epoch_size is set
    assert batch['context'].shape[1:] == (32, 6), f"Expected (B, 32, 6), got {batch['context'].shape}"
    print("✓ Data loading works!\n")
    
    return batch


def test_model_forward(batch):
    """Test model forward pass."""
    print("="*60)
    print("TEST 2: Model Forward Pass")
    print("="*60)
    
    # Load config
    config = OmegaConf.load('configs/nuscenes_planA_timedp.yaml')
    
    # Override for quick test
    config.model.params.unet_config.params.seq_len = 32
    config.model.params.seq_len = 32
    config.model.params.channels = 6
    
    # Instantiate model
    print("Instantiating model...")
    model = instantiate_from_config(config.model)
    model.eval()
    
    print(f"Model type: {type(model)}")
    print(f"Channels: {model.channels}")
    print(f"Seq len: {model.seq_len}")
    
    # Prepare input - use get_input to convert (B, T, F) -> (B, F, T)
    x_raw = batch['context']  # (B, T, F) = (16, 32, 6)
    print(f"Raw input shape: {x_raw.shape}")
    
    # Simulate get_input: rearrange (B, T, F) -> (B, F, T)
    x = x_raw.permute(0, 2, 1)  # (B, F, T)
    print(f"After permute shape: {x.shape}")
    
    # Forward pass through first stage (identity)
    with torch.no_grad():
        z = model.encode_first_stage(x)
        print(f"Encoded shape: {z.shape}")
        
        # Get conditioning - cond_stage_model expects (B, F, T)
        c, mask = model.get_learned_conditioning(x, return_mask=True)
        print(f"Conditioning shape: {c.shape if c is not None else None}")
        print(f"Mask shape: {mask.shape if mask is not None else None}")
    
    print("✓ Forward pass works!\n")
    return model


def test_training_step(batch):
    """Test full training step with loss and backward."""
    print("="*60)
    print("TEST 3: Training Step (Forward + Backward)")
    print("="*60)
    
    # Load config
    config = OmegaConf.load('configs/nuscenes_planA_timedp.yaml')
    config.model.params.unet_config.params.seq_len = 32
    config.model.params.seq_len = 32
    config.model.params.channels = 6
    
    # Instantiate model
    print("Instantiating model...")
    model = instantiate_from_config(config.model)
    model.train()
    model.learning_rate = 1e-4
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    print(f"Device: {device}")
    print(f"Input shape: {batch['context'].shape}")
    
    # Training step
    print("Running training step...")
    loss, loss_dict = model.shared_step(batch)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Check loss is valid
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    
    # Backward pass
    print("Running backward pass...")
    loss.backward()
    
    # Check gradients
    total_params = 0
    params_with_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                if torch.isnan(param.grad).any():
                    print(f"⚠️ NaN gradient in {name}")
    
    print(f"Parameters with gradients: {params_with_grad}/{total_params}")
    print("✓ Training step works!\n")
    
    return loss


def test_sampling(model):
    """Test sampling (generation)."""
    print("="*60)
    print("TEST 4: Sampling (Generation)")  
    print("="*60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Generate samples
    print("Generating samples...")
    with torch.no_grad():
        # Shape: (batch_size, channels, seq_len)
        samples = model.sample(cond=None, batch_size=4, verbose=False)
    
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.2f}, {samples.max():.2f}]")
    
    assert samples.shape == (4, 6, 32), f"Expected (4, 6, 32), got {samples.shape}"
    assert not torch.isnan(samples).any(), "Samples contain NaN!"
    
    print("✓ Sampling works!\n")


def main():
    print("\n" + "="*60)
    print("NUSCENES MULTIVARIATE TIMEDP PIPELINE TEST")
    print("="*60 + "\n")
    
    # Test 1: Data loading
    batch = test_data_loading()
    
    # Test 2: Model forward
    model = test_model_forward(batch)
    
    # Test 3: Training step
    loss = test_training_step(batch)
    
    # Test 4: Sampling (optional, slower)
    # test_sampling(model)
    
    print("="*60)
    print("ALL TESTS PASSED! Pipeline is ready for training.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: bash scripts/run_planA_train.sh 1  (single GPU test)")
    print("  2. Check loss decreases over training steps")
    print("  3. Proceed to full training: bash scripts/run_planA_train.sh")


if __name__ == '__main__':
    main()

