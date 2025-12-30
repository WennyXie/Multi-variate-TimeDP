#!/usr/bin/env python3
"""
nuScenes Plan A DataModule with DDP-safe Balanced Domain Sampling.

Supports:
- Train domains {0, 1, 2} with balanced sampling
- Unseen domain {3} with separate prompt_pool and eval_pool
- DDP-safe sampling with per-rank RNG seeds
- Both zscore and centered_pit normalization
"""

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
import json


class DDPBalancedDomainSampler(Sampler):
    """
    DDP-safe Balanced Domain Sampler.
    
    For each sample:
    1. Uniformly sample a domain from available domains
    2. Uniformly sample an index from that domain
    
    DDP safety:
    - Each rank uses distinct RNG seed (base_seed + global_rank)
    - Prints domain histogram for verification
    """
    
    def __init__(
        self,
        domain_labels: np.ndarray,
        epoch_size: int = None,
        base_seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True
    ):
        """
        Args:
            domain_labels: Array of domain IDs for each sample
            epoch_size: Samples per epoch per rank. If None, auto-compute
            base_seed: Base random seed
            rank: DDP rank (0 if single GPU)
            world_size: Number of GPUs
            verbose: Print verification stats
        """
        self.domain_labels = domain_labels
        self.base_seed = base_seed
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.epoch = 0
        
        # Build index lists per domain
        self.idx_by_domain = defaultdict(list)
        for idx, domain in enumerate(domain_labels):
            self.idx_by_domain[int(domain)].append(idx)
        
        self.domains = sorted(self.idx_by_domain.keys())
        self.num_domains = len(self.domains)
        
        # Convert to numpy arrays
        self.idx_by_domain = {d: np.array(indices) for d, indices in self.idx_by_domain.items()}
        
        # Epoch size per rank
        if epoch_size is None:
            max_domain_len = max(len(indices) for indices in self.idx_by_domain.values())
            self.epoch_size = max_domain_len * self.num_domains
        else:
            self.epoch_size = epoch_size
        
        if self.verbose and self.rank == 0:
            print(f"\nDDPBalancedDomainSampler initialized:")
            print(f"  Domains: {self.domains}")
            for d in self.domains:
                print(f"    Domain {d}: {len(self.idx_by_domain[d])} samples")
            print(f"  Epoch size per rank: {self.epoch_size}")
            print(f"  World size: {self.world_size}")
            print(f"  Total samples per epoch: {self.epoch_size * self.world_size}")
    
    def __iter__(self):
        # DDP-safe: each rank gets unique seed
        seed = self.base_seed + self.epoch * 1000 + self.rank
        rng = np.random.RandomState(seed)
        
        indices = []
        domain_counts = {d: 0 for d in self.domains}
        
        for _ in range(self.epoch_size):
            # 1. Sample domain uniformly
            domain = rng.choice(self.domains)
            domain_counts[domain] += 1
            
            # 2. Sample index uniformly from that domain
            idx = rng.choice(self.idx_by_domain[domain])
            indices.append(idx)
        
        # Verify balance on first epoch
        if self.verbose and self.epoch == 0:
            total = sum(domain_counts.values())
            print(f"\n  Rank {self.rank} domain sampling (first {min(2000, total)} samples):")
            for d in sorted(domain_counts.keys()):
                pct = domain_counts[d] / total * 100
                print(f"    Domain {d}: {domain_counts[d]} ({pct:.1f}%)")
        
        self.epoch += 1
        return iter(indices)
    
    def __len__(self):
        return self.epoch_size
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducibility across epochs."""
        self.epoch = epoch


class NuScenesPlanADataset(Dataset):
    """Dataset for Plan A nuScenes trajectory data."""
    
    def __init__(
        self,
        data: np.ndarray,
        domain_labels: np.ndarray = None,
        name: str = 'train'
    ):
        """
        Args:
            data: shape (N, D, T) normalized data
            domain_labels: shape (N,) domain IDs (optional for unseen)
            name: Dataset name
        """
        self.data = data.astype(np.float32)
        self.domain_labels = domain_labels
        self.name = name
        
        self.n_samples, self.n_features, self.seq_len = self.data.shape
        
        print(f"NuScenesPlanADataset '{name}':")
        print(f"  Shape: {self.data.shape} (N, D, T)")
        if domain_labels is not None:
            print(f"  Domains: {np.unique(domain_labels)}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.data[idx])  # (D, T)
        x = x.T  # (T, D) - TimeDP get_input will permute back
        
        domain = int(self.domain_labels[idx]) if self.domain_labels is not None else 3
        
        return {
            'context': x,  # (T, D)
            'data_key': domain
        }


class NuScenesPlanADataModule(pl.LightningDataModule):
    """
    DataModule for Plan A with DDP-safe balanced sampling.
    
    Data structure:
    - train.npy: (N_train, D, T) from domains {0, 1, 2}
    - domain_train.npy: (N_train,) domain labels
    - unseen_prompt_pool.npy: (N_prompt, D, T) for K-shot prompts
    - unseen_eval_pool.npy: (N_eval, D, T) for evaluation
    """
    
    def __init__(
        self,
        data_dir: str,
        train_file: str = 'train.npy',
        domain_train_file: str = 'domain_train.npy',
        unseen_prompt_file: str = 'unseen_prompt_pool.npy',
        unseen_eval_file: str = 'unseen_eval_pool.npy',
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        input_channels: int = 6,
        window: int = 32,
        epoch_size: int = None,
        seed: int = 0,
        normalize: str = 'zscore',
        **kwargs
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_file = train_file
        self.domain_train_file = domain_train_file
        self.unseen_prompt_file = unseen_prompt_file
        self.unseen_eval_file = unseen_eval_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.input_channels = input_channels
        self.window = window
        self.epoch_size = epoch_size
        self.seed = seed
        self.normalize = normalize
        
        # Data placeholders
        self.train_data = None
        self.train_domains = None
        self.val_data = None
        self.val_domains = None
        self.unseen_prompt_data = None
        self.unseen_eval_data = None
        
        # For compatibility
        self.key_list = ['domain_0', 'domain_1', 'domain_2', 'domain_3']
        self.normalizer_dict = {}
        
        # DDP info
        self.rank = 0
        self.world_size = 1
    
    def prepare_data(self) -> None:
        """Verify data files exist."""
        required = [self.train_file, self.domain_train_file, 
                    self.unseen_prompt_file, self.unseen_eval_file]
        for f in required:
            if not (self.data_dir / f).exists():
                raise FileNotFoundError(f"Missing: {self.data_dir / f}")
        
        # Load normalizer
        if self.normalize == 'zscore':
            stats_path = self.data_dir / 'norm_stats.npz'
            if stats_path.exists():
                stats = np.load(stats_path)
                self.normalizer_dict['nuscenes'] = {
                    'mean': stats['mean'],
                    'std': stats['std']
                }
        elif self.normalize == 'centered_pit':
            pit_path = self.data_dir / 'pit_params.json'
            if pit_path.exists():
                with open(pit_path) as f:
                    self.normalizer_dict['nuscenes_pit'] = json.load(f)
        
        print(f"Data verified in {self.data_dir}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets."""
        # Get DDP info - check both dist and trainer
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        elif hasattr(self, 'trainer') and self.trainer is not None:
            self.rank = self.trainer.global_rank
            self.world_size = self.trainer.world_size
        
        # Load data
        full_train = np.load(self.data_dir / self.train_file).astype(np.float32)
        full_domains = np.load(self.data_dir / self.domain_train_file).astype(np.int64)
        self.unseen_prompt_data = np.load(self.data_dir / self.unseen_prompt_file).astype(np.float32)
        self.unseen_eval_data = np.load(self.data_dir / self.unseen_eval_file).astype(np.float32)
        
        if self.rank == 0:
            print(f"\nLoaded Plan A data:")
            print(f"  Train: {full_train.shape}, domains: {np.unique(full_domains)}")
            print(f"  Unseen prompt: {self.unseen_prompt_data.shape}")
            print(f"  Unseen eval: {self.unseen_eval_data.shape}")
        
        # Split train/val (stratified)
        np.random.seed(42)
        n_train = len(full_train)
        n_val = max(1, int(n_train * self.val_split))
        
        indices = np.random.permutation(n_train)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        self.train_data = full_train[train_idx]
        self.train_domains = full_domains[train_idx]
        self.val_data = full_train[val_idx]
        self.val_domains = full_domains[val_idx]
        
        if self.rank == 0:
            print(f"\nTrain/Val split:")
            print(f"  Train: {len(self.train_data)}")
            print(f"    Domain 0: {(self.train_domains == 0).sum()}")
            print(f"    Domain 1: {(self.train_domains == 1).sum()}")
            print(f"    Domain 2: {(self.train_domains == 2).sum()}")
            print(f"  Val: {len(self.val_data)}")
    
    def train_dataloader(self) -> DataLoader:
        dataset = NuScenesPlanADataset(
            self.train_data, self.train_domains, name='train'
        )
        
        sampler = DDPBalancedDomainSampler(
            self.train_domains,
            epoch_size=self.epoch_size,
            base_seed=self.seed,
            rank=self.rank,
            world_size=self.world_size,
            verbose=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        dataset = NuScenesPlanADataset(
            self.val_data, self.val_domains, name='val'
        )
        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(self.val_data)),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return unseen eval pool as test set."""
        dataset = NuScenesPlanADataset(
            self.unseen_eval_data, 
            np.full(len(self.unseen_eval_data), 3),
            name='unseen_eval'
        )
        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(self.unseen_eval_data)),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_domain_data(self, domain_id: int, split: str = 'train') -> np.ndarray:
        """Get data for a specific domain."""
        if split == 'train':
            mask = self.train_domains == domain_id
            return self.train_data[mask]
        elif split == 'val':
            mask = self.val_domains == domain_id
            return self.val_data[mask]
        elif split == 'unseen_prompt':
            return self.unseen_prompt_data
        elif split == 'unseen_eval':
            return self.unseen_eval_data
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def get_prompt_pool(self, domain_id: int, split: str = 'train') -> np.ndarray:
        """
        Get prompt pool for a domain.
        
        For seen domains (0,1,2): use train data with index-based disjoint split
        For unseen domain (3): use unseen_prompt_pool
        """
        if domain_id == 3:
            return self.unseen_prompt_data
        else:
            # For seen domains, use first half as prompt pool
            domain_data = self.get_domain_data(domain_id, 'train')
            n_prompt = len(domain_data) // 2
            return domain_data[:n_prompt]
    
    def get_eval_pool(self, domain_id: int) -> np.ndarray:
        """
        Get eval pool for a domain.
        
        For seen domains (0,1,2): use second half of train data (disjoint from prompt)
        For unseen domain (3): use unseen_eval_pool
        """
        if domain_id == 3:
            return self.unseen_eval_data
        else:
            domain_data = self.get_domain_data(domain_id, 'train')
            n_prompt = len(domain_data) // 2
            return domain_data[n_prompt:]
    
    def inverse_transform(self, data: np.ndarray, data_name: str = None) -> np.ndarray:
        """Inverse transform normalized data."""
        if 'nuscenes' in self.normalizer_dict:
            normalizer = self.normalizer_dict['nuscenes']
            mean = normalizer['mean']
            std = normalizer['std']
            
            if data.ndim == 3:
                if data.shape[1] == len(mean):  # (N, D, T)
                    return data * std[:, None] + mean[:, None]
                elif data.shape[2] == len(mean):  # (N, T, D)
                    return data * std + mean
            elif data.ndim == 2:
                if data.shape[0] == len(mean):  # (D, T)
                    return data * std[:, None] + mean[:, None]
                else:  # (T, D)
                    return data * std + mean
        return data

