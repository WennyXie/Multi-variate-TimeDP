#!/usr/bin/env python3
"""
nuScenes Preprocessing for Plan A - TimeDP Aligned
===================================================

Key differences from previous versions:
- ALL scenes used (467+183+115+85 = 850 total)
- NON-OVERLAPPING slicing: stride = T = 32
- Two normalization options: zscore (default) and centered_pit (TimeDP-style)
- Unseen domain D3 split into disjoint prompt_pool and eval_pool
- Normalization fitted ONLY on train domains {D0, D1, D2}

Domains:
- D0 (seen train): boston-seaport (467 scenes)
- D1 (seen train): singapore-onenorth (183 scenes)  
- D2 (seen train): singapore-queenstown (115 scenes)
- D3 (UNSEEN test): singapore-hollandvillage (85 scenes)

Usage:
    python nuscenes_preprocess_planA.py --normalize zscore
    python nuscenes_preprocess_planA.py --normalize centered_pit
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class NuScenesMetadataLoader:
    """Load nuScenes metadata from JSON files (no sensor blobs)."""
    
    def __init__(self, dataroot: str):
        self.dataroot = Path(dataroot)
        self.tables = {}
        self._load_tables()
        self._build_indexes()
    
    def _load_tables(self):
        """Load all required JSON tables."""
        table_names = ['scene', 'sample', 'sample_data', 'ego_pose', 'log', 
                       'calibrated_sensor', 'sensor']
        
        for name in table_names:
            path = self.dataroot / f'{name}.json'
            if path.exists():
                with open(path, 'r') as f:
                    self.tables[name] = json.load(f)
                print(f"  Loaded {name}: {len(self.tables[name])} records")
            else:
                raise FileNotFoundError(f"Table not found: {path}")
    
    def _build_indexes(self):
        """Build token -> record indexes for fast lookup."""
        self.idx = {}
        for name, records in self.tables.items():
            self.idx[name] = {r['token']: r for r in records}
        
        # Build sensor name index
        self.sensor_by_name = {}
        for s in self.tables['sensor']:
            self.sensor_by_name[s['channel']] = s['token']
        
        # Build calibrated_sensor index by sensor token
        self.cal_sensor_by_sensor = {}
        for cs in self.tables['calibrated_sensor']:
            self.cal_sensor_by_sensor[cs['sensor_token']] = cs
        
        # OPTIMIZATION: Build sample_token -> sample_data index for LIDAR_TOP keyframes
        # This eliminates O(n²) lookup in get_scene_trajectory
        print("  Building sample_data index (this speeds up processing 100x)...")
        lidar_sensor_token = self.sensor_by_name.get('LIDAR_TOP')
        self.lidar_sd_by_sample = {}  # sample_token -> sample_data record
        
        if lidar_sensor_token:
            for sd in self.tables['sample_data']:
                if sd['is_key_frame'] and sd['sample_token']:
                    cal_sensor = self.idx['calibrated_sensor'].get(sd['calibrated_sensor_token'])
                    if cal_sensor and cal_sensor['sensor_token'] == lidar_sensor_token:
                        self.lidar_sd_by_sample[sd['sample_token']] = sd
        
        print(f"  Indexed {len(self.lidar_sd_by_sample)} LIDAR_TOP keyframes")
    
    def get(self, table: str, token: str):
        return self.idx[table].get(token)


def quaternion_to_yaw(q: List[float]) -> float:
    """Convert quaternion [w,x,y,z] to yaw angle."""
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
    return r.as_euler('zyx')[0]


def get_scene_trajectory(loader: NuScenesMetadataLoader, scene_token: str) -> Optional[Dict]:
    """Extract ego trajectory from a scene using LIDAR_TOP keyframes.
    
    OPTIMIZED: Uses pre-built index for O(1) sample_data lookup instead of O(n) scan.
    """
    scene = loader.get('scene', scene_token)
    if not scene:
        return None
    
    positions = []
    yaws = []
    timestamps = []
    
    # Traverse sample chain
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = loader.get('sample', sample_token)
        if not sample:
            break
        
        # OPTIMIZED: O(1) lookup using pre-built index
        lidar_sd = loader.lidar_sd_by_sample.get(sample_token)
        
        if lidar_sd:
            ego_pose = loader.get('ego_pose', lidar_sd['ego_pose_token'])
            if ego_pose:
                x, y, _ = ego_pose['translation']
                yaw = quaternion_to_yaw(ego_pose['rotation'])
                ts = lidar_sd['timestamp'] / 1e6  # microseconds to seconds
                
                positions.append([x, y])
                yaws.append(yaw)
                timestamps.append(ts)
        
        sample_token = sample['next'] if sample['next'] else None
    
    if len(positions) < 2:
        return None
    
    return {
        'positions': np.array(positions),
        'yaws': np.array(yaws),
        'timestamps': np.array(timestamps),
        'scene_token': scene_token
    }


def extract_trajectory_features(traj: Dict) -> np.ndarray:
    """
    Extract 6-channel features from trajectory.
    
    Channels: [dx, dy, v, a, yaw_rate, curvature]
    - Ego-centric coordinates (relative to first frame)
    - Uses true dt for derivatives
    """
    positions = traj['positions']
    yaws = traj['yaws']
    timestamps = traj['timestamps']
    
    L = len(positions)
    if L < 2:
        return None
    
    # Ego-centric transformation
    origin = positions[0]
    yaw0 = yaws[0]
    
    # Rotation matrix for -yaw0
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    R = np.array([[c, -s], [s, c]])
    
    # Transform positions to ego frame
    positions_ego = (positions - origin) @ R.T
    
    # Align and unwrap yaws
    yaws_aligned = np.unwrap(yaws - yaw0)
    
    # Compute dt
    dt = np.diff(timestamps)
    dt = np.maximum(dt, 1e-6)  # Prevent division by zero
    
    # Compute features
    features = np.zeros((L, 6), dtype=np.float32)
    
    # dx, dy (displacements in ego frame)
    features[1:, 0] = np.diff(positions_ego[:, 0])  # dx
    features[1:, 1] = np.diff(positions_ego[:, 1])  # dy
    
    # v (speed)
    displacement = np.sqrt(features[1:, 0]**2 + features[1:, 1]**2)
    features[1:, 2] = displacement / dt
    
    # a (acceleration)
    features[2:, 3] = np.diff(features[1:, 2]) / dt[1:]
    
    # yaw_rate
    features[1:, 4] = np.diff(yaws_aligned) / dt
    
    # curvature = yaw_rate / max(v, 0.1)
    v_safe = np.maximum(features[1:, 2], 0.1)
    features[1:, 5] = features[1:, 4] / v_safe
    
    # t=0 already zero (initialized)
    
    return features


def create_windows_nonoverlap(features: np.ndarray, window_size: int = 32) -> List[np.ndarray]:
    """
    Create NON-OVERLAPPING windows with stride = window_size.
    
    This is the TimeDP-aligned approach.
    """
    L = len(features)
    windows = []
    
    for start in range(0, L - window_size + 1, window_size):
        window = features[start:start + window_size]
        windows.append(window)
    
    return windows


class CenteredPITNormalizer:
    """
    Centered Probability Integral Transform normalizer.
    
    Maps data to [-1, 1] using empirical CDF (TimeDP-style).
    """
    
    def __init__(self):
        self.fitted = False
        self.ecdf_x = {}  # Per-channel sorted values
        self.ecdf_y = {}  # Per-channel CDF values
    
    def fit(self, data: np.ndarray):
        """
        Fit ECDF per channel.
        
        Args:
            data: shape (N, T, D) or flattened per channel
        """
        if data.ndim == 3:
            N, T, D = data.shape
            data_flat = data.reshape(-1, D)
        else:
            data_flat = data
        
        D = data_flat.shape[1]
        
        for d in range(D):
            channel_data = data_flat[:, d]
            # Remove NaN/Inf
            valid = np.isfinite(channel_data)
            channel_data = channel_data[valid]
            
            # Compute ECDF
            sorted_data = np.sort(channel_data)
            ecdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            self.ecdf_x[d] = sorted_data
            self.ecdf_y[d] = ecdf_y
        
        self.fitted = True
        print(f"  CenteredPIT fitted on {len(data_flat)} samples, {D} channels")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted ECDF, map to [-1, 1]."""
        assert self.fitted, "Must call fit() first"
        
        original_shape = data.shape
        if data.ndim == 3:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data.copy()
        
        result = np.zeros_like(data_flat)
        
        for d in range(data_flat.shape[1]):
            # Use linear interpolation for CDF lookup
            result[:, d] = np.interp(data_flat[:, d], self.ecdf_x[d], self.ecdf_y[d])
        
        # Center to [-1, 1]
        result = result * 2 - 1
        
        return result.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform from [-1, 1] back to original scale."""
        assert self.fitted, "Must call fit() first"
        
        original_shape = data.shape
        if data.ndim == 3:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data.copy()
        
        # Uncentering: from [-1, 1] to [0, 1]
        data_flat = (data_flat + 1) / 2
        data_flat = np.clip(data_flat, 0, 1)
        
        result = np.zeros_like(data_flat)
        
        for d in range(data_flat.shape[1]):
            # Inverse CDF lookup
            result[:, d] = np.interp(data_flat[:, d], self.ecdf_y[d], self.ecdf_x[d])
        
        return result.reshape(original_shape)
    
    def get_params(self) -> Dict:
        """Get parameters for saving."""
        return {
            'ecdf_x': {str(k): v.tolist() for k, v in self.ecdf_x.items()},
            'ecdf_y': {str(k): v.tolist() for k, v in self.ecdf_y.items()}
        }


class ZScoreNormalizer:
    """Simple Z-score normalizer (mean/std per channel)."""
    
    def __init__(self, clip_percentile: float = None):
        """
        Args:
            clip_percentile: If set, winsorize at [p, 100-p] percentiles before computing stats
        """
        self.clip_percentile = clip_percentile
        self.mean = None
        self.std = None
        self.clip_min = None
        self.clip_max = None
        self.fitted = False
    
    def fit(self, data: np.ndarray):
        """Fit mean/std per channel, optionally with winsorization."""
        if data.ndim == 3:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data.copy()
        
        D = data_flat.shape[1]
        
        # Winsorization if requested
        if self.clip_percentile is not None:
            self.clip_min = np.zeros(D, dtype=np.float32)
            self.clip_max = np.zeros(D, dtype=np.float32)
            
            for d in range(D):
                self.clip_min[d] = np.nanpercentile(data_flat[:, d], self.clip_percentile)
                self.clip_max[d] = np.nanpercentile(data_flat[:, d], 100 - self.clip_percentile)
            
            # Clip data before computing mean/std
            data_flat = np.clip(data_flat, self.clip_min, self.clip_max)
            
            print(f"  Winsorization at {self.clip_percentile}% percentile:")
            channel_names = ['dx', 'dy', 'v', 'a', 'yaw_rate', 'curvature']
            for d in range(D):
                print(f"    {channel_names[d]}: [{self.clip_min[d]:.4f}, {self.clip_max[d]:.4f}]")
        
        self.mean = np.nanmean(data_flat, axis=0).astype(np.float32)
        self.std = np.nanstd(data_flat, axis=0).astype(np.float32)
        self.std = np.maximum(self.std, 1e-6)  # Prevent division by zero
        
        self.fitted = True
        print(f"  ZScore fitted: mean={self.mean}")
        print(f"                 std={self.std}")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.fitted, "Must call fit() first"
        
        # Clip if winsorization was used
        if self.clip_percentile is not None:
            data = np.clip(data, self.clip_min, self.clip_max)
        
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert self.fitted, "Must call fit() first"
        return data * self.std + self.mean
    
    def get_params(self) -> Dict:
        params = {'mean': self.mean.tolist(), 'std': self.std.tolist()}
        if self.clip_percentile is not None:
            params['clip_percentile'] = self.clip_percentile
            params['clip_min'] = self.clip_min.tolist()
            params['clip_max'] = self.clip_max.tolist()
        return params


def preprocess_nuscenes_planA(
    dataroot: str,
    output_dir: str,
    window_size: int = 32,
    normalize: str = 'zscore',
    seed: int = 0,
    unseen_prompt_ratio: float = 0.5
):
    """
    Main preprocessing function for Plan A.
    
    Args:
        dataroot: Path to nuScenes metadata
        output_dir: Output directory
        window_size: Window size T (also stride for non-overlap)
        normalize: 'zscore' or 'centered_pit'
        seed: Random seed
        unseen_prompt_ratio: Ratio of D3 windows for prompt pool (rest for eval)
    """
    print("="*60)
    print("nuScenes Plan A Preprocessing")
    print("="*60)
    print(f"  dataroot: {dataroot}")
    print(f"  output_dir: {output_dir}")
    print(f"  window_size (T): {window_size}")
    print(f"  stride: {window_size} (NON-OVERLAPPING)")
    print(f"  normalize: {normalize}")
    print(f"  seed: {seed}")
    print()
    
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("Loading nuScenes metadata...")
    loader = NuScenesMetadataLoader(dataroot)
    
    # Categorize scenes by location
    print("\nCategorizing scenes by location...")
    domain_scenes = {
        0: [],  # boston-seaport
        1: [],  # singapore-onenorth
        2: [],  # singapore-queenstown
        3: []   # singapore-hollandvillage (UNSEEN)
    }
    
    location_map = {
        'boston-seaport': 0,
        'singapore-onenorth': 1,
        'singapore-queenstown': 2,
        'singapore-hollandvillage': 3
    }
    
    for scene in loader.tables['scene']:
        log = loader.get('log', scene['log_token'])
        location = log['location']
        
        if location in location_map:
            domain_id = location_map[location]
            domain_scenes[domain_id].append(scene['token'])
    
    print("  Scene counts by domain:")
    for d, scenes in domain_scenes.items():
        loc_name = [k for k, v in location_map.items() if v == d][0]
        print(f"    D{d} ({loc_name}): {len(scenes)} scenes")
    
    # Process each domain
    def process_domain(scene_tokens: List[str], domain_id: int, domain_name: str):
        """Process all scenes for a domain."""
        all_windows = []
        scene_stats = []
        dt_values = []
        
        print(f"\n  Processing {domain_name} ({len(scene_tokens)} scenes)...")
        
        for i, scene_token in enumerate(scene_tokens):
            traj = get_scene_trajectory(loader, scene_token)
            if traj is None:
                continue
            
            features = extract_trajectory_features(traj)
            if features is None or len(features) < window_size:
                continue
            
            windows = create_windows_nonoverlap(features, window_size)
            all_windows.extend(windows)
            
            # Collect dt values
            dt = np.diff(traj['timestamps'])
            dt_values.extend(dt.tolist())
            
            scene_stats.append({
                'scene_token': scene_token,
                'trajectory_length': len(features),
                'n_windows': len(windows)
            })
            
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(scene_tokens)} scenes, {len(all_windows)} windows so far")
        
        print(f"    Total: {len(all_windows)} windows from {len(scene_stats)} valid scenes")
        
        return all_windows, scene_stats, dt_values
    
    # Process train domains (D0, D1, D2)
    train_windows = []
    train_domains = []
    all_scene_stats = {}
    all_dt_values = []
    
    domain_names = {
        0: 'boston-seaport',
        1: 'singapore-onenorth', 
        2: 'singapore-queenstown',
        3: 'singapore-hollandvillage'
    }
    
    for domain_id in [0, 1, 2]:
        windows, stats, dt_vals = process_domain(
            domain_scenes[domain_id], domain_id, domain_names[domain_id]
        )
        train_windows.extend(windows)
        train_domains.extend([domain_id] * len(windows))
        all_scene_stats[domain_names[domain_id]] = {
            'domain_id': domain_id,
            'n_scenes': len(domain_scenes[domain_id]),
            'n_valid_scenes': len(stats),
            'n_windows': len(windows),
            'scene_stats': stats
        }
        all_dt_values.extend(dt_vals)
    
    # Process unseen domain (D3)
    unseen_windows, unseen_stats, unseen_dt = process_domain(
        domain_scenes[3], 3, domain_names[3]
    )
    all_scene_stats[domain_names[3]] = {
        'domain_id': 3,
        'n_scenes': len(domain_scenes[3]),
        'n_valid_scenes': len(unseen_stats),
        'n_windows': len(unseen_windows),
        'scene_stats': unseen_stats
    }
    all_dt_values.extend(unseen_dt)
    
    # Convert to arrays
    X_train_raw = np.stack(train_windows, axis=0).astype(np.float32)  # [N_train, T, D]
    domain_train = np.array(train_domains, dtype=np.int64)
    X_unseen_raw = np.stack(unseen_windows, axis=0).astype(np.float32)  # [N_unseen, T, D]
    
    print(f"\nRaw data shapes:")
    print(f"  X_train: {X_train_raw.shape}")
    print(f"  X_unseen: {X_unseen_raw.shape}")
    
    # Split unseen into prompt_pool and eval_pool (disjoint)
    n_unseen = len(X_unseen_raw)
    n_prompt = int(n_unseen * unseen_prompt_ratio)
    
    np.random.seed(seed)
    unseen_indices = np.random.permutation(n_unseen)
    prompt_indices = unseen_indices[:n_prompt]
    eval_indices = unseen_indices[n_prompt:]
    
    X_unseen_prompt = X_unseen_raw[prompt_indices]
    X_unseen_eval = X_unseen_raw[eval_indices]
    
    print(f"\n  Unseen domain split (seed={seed}):")
    print(f"    Prompt pool: {len(X_unseen_prompt)} windows")
    print(f"    Eval pool: {len(X_unseen_eval)} windows")
    
    # Fit normalizer ONLY on train data
    print(f"\nFitting normalizer ({normalize}) on train data only...")
    
    if normalize == 'centered_pit':
        normalizer = CenteredPITNormalizer()
    elif normalize == 'zscore_winsor':
        normalizer = ZScoreNormalizer(clip_percentile=1.0)  # Winsorize at 1%/99%
    else:  # zscore
        normalizer = ZScoreNormalizer()
    
    normalizer.fit(X_train_raw)
    
    # Transform all data
    X_train = normalizer.transform(X_train_raw)
    X_unseen_prompt_norm = normalizer.transform(X_unseen_prompt)
    X_unseen_eval_norm = normalizer.transform(X_unseen_eval)
    
    # Compute denorm sanity stats
    def compute_channel_stats(data: np.ndarray, name: str) -> Dict:
        """Compute per-channel stats for v, a, curvature."""
        # Channels: [dx, dy, v, a, yaw_rate, curvature]
        v = data[:, :, 2]
        a = data[:, :, 3]
        curv = data[:, :, 5]
        
        return {
            'v_mean': float(np.mean(v)),
            'v_std': float(np.std(v)),
            'v_min': float(np.min(v)),
            'v_max': float(np.max(v)),
            'a_mean': float(np.mean(a)),
            'a_std': float(np.std(a)),
            'a_min': float(np.min(a)),
            'a_max': float(np.max(a)),
            'curvature_mean': float(np.mean(curv)),
            'curvature_std': float(np.std(curv)),
            'curvature_min': float(np.min(curv)),
            'curvature_max': float(np.max(curv))
        }
    
    denorm_stats = {
        'train': compute_channel_stats(X_train_raw, 'train'),
        'unseen_prompt': compute_channel_stats(X_unseen_prompt, 'unseen_prompt'),
        'unseen_eval': compute_channel_stats(X_unseen_eval, 'unseen_eval')
    }
    
    # Per-domain train stats
    for domain_id in [0, 1, 2]:
        mask = domain_train == domain_id
        domain_data = X_train_raw[mask]
        denorm_stats[f'train_domain{domain_id}'] = compute_channel_stats(domain_data, f'domain{domain_id}')
    
    # Save dataset.npz
    print("\nSaving dataset...")
    
    # Transpose to (N, D, T) format for model
    X_train_model = X_train.transpose(0, 2, 1)  # [N, D, T]
    X_unseen_prompt_model = X_unseen_prompt_norm.transpose(0, 2, 1)
    X_unseen_eval_model = X_unseen_eval_norm.transpose(0, 2, 1)
    
    npz_path = output_path / 'dataset.npz'
    np.savez(
        npz_path,
        # Model format (N, D, T)
        X_train=X_train_model,
        domain_train=domain_train,
        X_unseen_prompt_pool=X_unseen_prompt_model,
        X_unseen_eval_pool=X_unseen_eval_model,
        # Also save raw (N, T, D) for reference
        X_train_raw_NTD=X_train,
        X_unseen_prompt_raw_NTD=X_unseen_prompt_norm,
        X_unseen_eval_raw_NTD=X_unseen_eval_norm
    )
    print(f"  Saved: {npz_path}")
    
    # Save individual npy files for DataModule
    np.save(output_path / 'train.npy', X_train_model)
    np.save(output_path / 'domain_train.npy', domain_train)
    np.save(output_path / 'unseen_prompt_pool.npy', X_unseen_prompt_model)
    np.save(output_path / 'unseen_eval_pool.npy', X_unseen_eval_model)
    
    # Save normalizer params
    norm_params = normalizer.get_params()
    if normalize == 'zscore':
        np.savez(output_path / 'norm_stats.npz', 
                 mean=np.array(norm_params['mean']),
                 std=np.array(norm_params['std']))
    else:
        with open(output_path / 'pit_params.json', 'w') as f:
            json.dump(norm_params, f)
    
    # Save scenes_used.json
    scenes_used = {
        'seed': seed,
        'domains': {
            domain_names[d]: domain_scenes[d] for d in [0, 1, 2, 3]
        }
    }
    with open(output_path / 'scenes_used.json', 'w') as f:
        json.dump(scenes_used, f, indent=2)
    print(f"  Saved: scenes_used.json")
    
    # Save stats.json
    dt_array = np.array(all_dt_values)
    stats = {
        'preprocessing': {
            'window_size': window_size,
            'stride': window_size,
            'normalize': normalize,
            'seed': seed,
            'unseen_prompt_ratio': unseen_prompt_ratio
        },
        'counts': {
            'train_windows': len(X_train),
            'train_domain_0': int((domain_train == 0).sum()),
            'train_domain_1': int((domain_train == 1).sum()),
            'train_domain_2': int((domain_train == 2).sum()),
            'unseen_prompt_pool': len(X_unseen_prompt),
            'unseen_eval_pool': len(X_unseen_eval)
        },
        'dt_stats': {
            'min': float(dt_array.min()),
            'max': float(dt_array.max()),
            'mean': float(dt_array.mean()),
            'median': float(np.median(dt_array))
        },
        'denorm_stats': denorm_stats,
        'per_domain_stats': all_scene_stats,
        'normalizer_params': norm_params
    }
    
    with open(output_path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: stats.json")
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nData counts:")
    print(f"  Train total: {len(X_train)} windows")
    print(f"    D0 (boston): {(domain_train == 0).sum()}")
    print(f"    D1 (sg-onenorth): {(domain_train == 1).sum()}")
    print(f"    D2 (sg-queenstown): {(domain_train == 2).sum()}")
    print(f"  Unseen D3 (sg-hollandvillage):")
    print(f"    Prompt pool: {len(X_unseen_prompt)}")
    print(f"    Eval pool: {len(X_unseen_eval)}")
    
    print(f"\nNormalized data range:")
    print(f"  Train: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Unseen prompt: [{X_unseen_prompt_norm.min():.3f}, {X_unseen_prompt_norm.max():.3f}]")
    print(f"  Unseen eval: [{X_unseen_eval_norm.min():.3f}, {X_unseen_eval_norm.max():.3f}]")
    
    print(f"\nOutput directory: {output_path}")
    
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nuScenes Plan A Preprocessing')
    parser.add_argument('--dataroot', type=str, 
                        default='data/v1.0-trainval_meta',
                        help='Path to nuScenes metadata')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')
    parser.add_argument('--window_size', '-T', type=int, default=32,
                        help='Window size (also stride for non-overlap)')
    parser.add_argument('--normalize', type=str, default='zscore',
                        choices=['zscore', 'centered_pit'],
                        help='Normalization method')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--unseen_prompt_ratio', type=float, default=0.5,
                        help='Ratio of unseen windows for prompt pool')
    
    args = parser.parse_args()
    
    # Auto-generate output directory name
    if args.output_dir is None:
        args.output_dir = f'data/nuscenes_planA_allscenes_T{args.window_size}_stride{args.window_size}_{args.normalize}_seed{args.seed}'
    
    preprocess_nuscenes_planA(
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        window_size=args.window_size,
        normalize=args.normalize,
        seed=args.seed,
        unseen_prompt_ratio=args.unseen_prompt_ratio
    )

