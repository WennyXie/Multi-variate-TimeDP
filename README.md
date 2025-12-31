# TimeDP: Learning to Generate Multi-Domain Time Series with Domain Prompts

## Papers
- [TimeDP: Learning to Generate Multi-Domain Time Series with Domain Prompts](https://arxiv.org/pdf/2501.05403)

## Overview
Time series generation is a crucial task for many applications, including data augmentation, forecasting, and privacy-preserving data synthesis. Real-world scenarios often require generating time series data that aligns with given examples. **TimeDP** is designed to address this challenge by learning from multiple domains and generating realistic time series samples based on domain-specific characteristics.

## Introduction
Most existing time series generation models focus on single-domain data, limiting their ability to generalize across various real-world scenarios. **TimeDP** introduces a novel approach that enables multi-domain time series generation using a domain-prompting mechanism. By leveraging diverse time series datasets, TimeDP extracts fundamental patterns and applies them to generate high-quality, domain-specific synthetic time series data.

## Key Features
- **Multi-Domain Time Series Generation:** TimeDP integrates data from various domains and learns shared representations.
- **Time Series Basis Extraction:** A semantic prototype module constructs a basis library for time series data.
- **Few-Shot Domain Prompting:** Given a few example time series from a new domain, TimeDP can generate similar time series with high fidelity.
- **Diffusion-Based Model:** Built upon denoising diffusion probabilistic models (DDPM), ensuring robust and high-quality generation.
- **State-of-the-Art Performance:** TimeDP surpasses existing baselines in both in-domain and unseen domain generation tasks.

## Methodology
### 1. Input Processing
- Time series data from various domains are fed into TimeDP.
- The model extracts fundamental time series features (prototypes) that serve as elementary building blocks.

### 2. Learning Time Series Basis
- A **semantic prototype module** constructs a time series basis library.
- Each prototype represents elementary time series patterns such as trends and seasonality.
- A **prototype assignment module (PAM)** learns domain-specific prototype weights to construct domain prompts.

### 3. Generation with Domain Prompts
- During sampling, a small number of example time series from the target domain are provided.
- TimeDP extracts domain prompts and uses them as conditions for generating new time series.
- The generated time series maintains the statistical and temporal characteristics of the target domain.

## TimeDP framework overview
![TimeDP framework overview.](./figure/TimeDP_Overview.jpg)


## Quick Start

### Environment Setup

We recommend using conda as environment manager:
```bash
conda env create -f environment.yml
```

**Important**: Set the `DATA_ROOT` environment variable before running training or evaluation:
```bash
export DATA_ROOT=/path/to/TimeDP/data
```

### Data Preparation

**Option 1: Use pre-processed data**
The pre-processed nuScenes data is already available in the `data/` directory. Simply set the `DATA_ROOT` environment variable to point to the data directory:
```bash
export DATA_ROOT=/path/to/TimeDP/data
```

Two preprocessed datasets are available with different normalization methods:

- **`data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0/`**: Uses standard z-score normalization (zero mean, unit variance). This is the default normalization method.

- **`data/nuscenes_planA_T32_centered_pit_seed0/`**: Uses centered PIT (Permutation Invariant Transformation) normalization, which is the TimeDP-style normalization method that preserves permutation-invariant properties of the data.

**Option 2: Preprocess from raw nuScenes data**
Download nuScenes metadata and run preprocessing (see Usage section below).

### Command Line Arguments

The detailed descriptions about command line arguments for training are as follows:
| Parameter Name                    | Description                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `base` (`-b`)                     | Paths to base configuration files.                                                                                 |
| `train` (`-t`)                    | Boolean flag to enable training. (default: true)                                                                   |
| `debug` (`-d`)                    | Boolean flag to enter debug mode. (default: false)                                                                 |
| `seed` (`-s`)                     | Seed for initializing random number generators. (default: 23)                                                      |
| `logdir` (`-l`)                   | Directory for logging data. (default: ./logs)                                                                      |
| `seq_len` (`-sl`)                 | Sequence length for the model. (default: 24)                                                                       |
| `uncond` (`-uc`)                  | Boolean flag for unconditional generation.                                                                         |
| `use_pam` (`-up`)                 | Boolean flag to use the prototype assignment module.                                                               |
| `batch_size` (`-bs`)              | Batch size for training. (default: 128)                                                                            |
| `num_latents` (`-nl`)             | Number of latent variables. (default: 16)                                                                          |
| `overwrite_learning_rate` (`-lr`) | Learning rate to overwrite the config file. (default: None)                                                        |
| `gpus`                            | Comma-separated list of GPU ids to use for training.                                                               |
| `ckpt_name`                       | Checkpoint name to resume from for test or visualization. (default: last)                                          |


## Usage

### Complete Pipeline

The typical workflow consists of four main steps: preprocessing, training, evaluation, and visualization.

#### 1. Data Preprocessing

**Note**: If you are using the pre-processed data (Option 1 in Data Preparation section), you can skip this step and proceed directly to training.

Preprocess nuScenes dataset to extract trajectory features and create train/eval splits:

```bash
# Z-score normalization (default)
python preprocessing/nuscenes_preprocess_planA.py \
    --dataroot data/nuscenes_meta/v1.0-trainval \
    --normalize zscore

# Centered PIT normalization (TimeDP-style)
python preprocessing/nuscenes_preprocess_planA.py \
    --dataroot data/nuscenes_meta/v1.0-trainval \
    --normalize centered_pit
```

**Output**: Processed data saved to `data/nuscenes_planA_allscenes_T32_stride32_{normalize}_seed0/`

**Key parameters**:
- `--dataroot`: Path to nuScenes metadata directory
- `--normalize`: Normalization method (`zscore`, `centered_pit`)
- `--window_size` / `-T`: Window size (default: 32, also used as stride for non-overlapping windows)
- `--seed`: Random seed for unseen domain split (default: 0)

#### 2. Training

Train TimeDP model on nuScenes data:

```bash
# Single GPU training
python main_train.py \
    --base configs/nuscenes_planA_timedp.yaml \
    --gpus 0, \
    --logdir ./logs/nuscenes_planA \
    -sl 32 \
    -up \
    -nl 16 \
    --batch_size 32 \
    -lr 0.0001 \
    -s 0 \
    --max_steps 5000

# Multi-GPU training (recommended)
bash scripts/run_planA_train.sh
```

**Key parameters**:
- `--base` / `-b`: Path to config file
- `--gpus`: Comma-separated GPU IDs (e.g., `0,1,2,3`)
- `--logdir` / `-l`: Logging directory
- `-sl` / `--seq_len`: Sequence length (32 for nuScenes)
- `-up` / `--use_pam`: Enable prototype assignment module
- `-nl` / `--num_latents`: Number of prototypes (default: 16)
- `--batch_size` / `-bs`: Batch size per GPU
- `-lr` / `--overwrite_learning_rate`: Learning rate
- `-s` / `--seed`: Random seed
- `--max_steps`: Maximum training steps

**Checkpoints**: Saved in `logs/nuscenes_planA/{run_name}/checkpoints/`

#### 3. Evaluation

Evaluate trained model on seen and unseen domains:

```bash
python evaluation/eval_planA.py \
    --ckpt logs/nuscenes_planA/.../checkpoints/last.ckpt \
    --config configs/nuscenes_planA_timedp.yaml \
    --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
    --N_gen 128 \
    --K_values 1,4,16 \
    --modes no_prompt,k_shot,shuffled
```
**Key parameters**:
- `--ckpt`: Path to checkpoint file
- `--config`: Model config file
- `--data_dir`: Preprocessed data directory
- `--N_gen`: Number of samples to generate per evaluation
- `--K_values`: Comma-separated K values for few-shot prompting
- `--modes`: Evaluation modes (`no_prompt`, `k_shot`, `shuffled`, `minus_pam`)
- `--domains`: Domain IDs to evaluate (default: all domains)
- `--run_pit_diagnostics`: Run PIT diagnostics (for centered_pit normalization)
- `--save_corr_heatmaps`: Generate correlation heatmaps

**Output**: Results saved to `{data_dir}/fewshot_eval/summary.json`

#### 4. Visualization

Generate visualizations of evaluation results and prompt analysis:

**Metrics vs K** (plots evaluation metrics across K values):
```bash
python visualization/tools/plot_metrics_vs_k.py \
    --zscore_summary data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0/fewshot_eval/summary.json \
    --pit_summary data/nuscenes_planA_T32_centered_pit_seed0/fewshot_eval/summary.json \
    --outdir figures
```

**UMAP Comparison** (compares UMAP embeddings across different normalization methods):
```bash
python visualization/tools/compare_umap_methods.py \
    --zscore_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
    --centered_pit_dir data/nuscenes_planA_T32_centered_pit_seed0 \
    --out_dir visualization/umap_comparison
```

**Note**: This script requires denormalized samples from evaluation. Make sure to run evaluation with `--save_samples` flag first to generate the required sample files in `{data_dir}/fewshot_eval/samples/`.

**Noise Floor Heatmap** (demonstrates sampling noise floor by comparing real data split into two halves):
```bash
python visualization/tools/plot_real_real_heatmaps.py \
    --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
    --out_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0/noise_floor_heatmaps
```

**Prompt Heatmap** (shows prototype activation patterns):
```bash
python visualization/tools/viz_prompt_heatmap.py \
    --ckpt logs/.../checkpoints/last.ckpt \
    --config configs/nuscenes_planA_timedp.yaml \
    --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
    --domains 0,1,2 \
    --n_per_domain 50 \
    --output figures/prompt_heatmap.png
```

**Prompt UMAP** (shows prompt space structure):
```bash
python visualization/tools/viz_prompt_umap.py \
    --ckpt logs/.../checkpoints/last.ckpt \
    --config configs/nuscenes_planA_timedp.yaml \
    --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
    --domains 0,1,2,3 \
    --n_per_domain 100 \
    --output figures/prompt_umap.png \
    --simple
```

**One-hot Prototypes** (visualizes individual prototype semantics):
```bash
python visualization/tools/viz_onehot_prototypes.py \
    --ckpt logs/.../checkpoints/last.ckpt \
    --config configs/nuscenes_planA_timedp.yaml \
    --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
    --output figures/onehot_prototypes.png
```

**Prompt Sparsity** (analyzes active prototype counts):
```bash
python visualization/tools/plot_prompt_sparsity.py \
    --ckpt logs/.../checkpoints/last.ckpt \
    --config configs/nuscenes_planA_timedp.yaml \
    --data_dir data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0 \
    --outdir figures
```

## Citation
If you use TimeDP in your research, please cite:
```
@article{huang2025timedp,
  title={TimeDP: Learning to Generate Multi-Domain Time Series with Domain Prompts},
  author={Yu-Hao Huang, Chang Xu, Yueying Wu, Wu-Jun Li, Jiang Bian},
  journal={AAAI 2025},
  year={2025}
}
```


