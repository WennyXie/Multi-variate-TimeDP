#!/bin/bash
# ============================================================
# Plan A Training Script - 4 GPU DDP
# ============================================================
#
# Usage:
#   bash run_planA_train.sh         # 4 GPU DDP
#   bash run_planA_train.sh 1       # Single GPU (for testing)
#
# Requirements:
#   - 4 GPUs with at least 16GB VRAM each
#   - Preprocessed data in data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0/

set -e

# Activate environment
source /u/pyt9dp/.conda/envs/drive/etc/profile.d/conda.sh
conda activate timedp

cd /u/pyt9dp/TimeCraft/TimeDP

export DATA_ROOT=/u/pyt9dp/TimeCraft/TimeDP/data
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=${1:-4}
MAX_STEPS=${2:-5000}
BATCH_SIZE=${3:-32}

echo "============================================================"
echo "Plan A Training - TimeDP Aligned"
echo "============================================================"
echo "  GPUs: $NUM_GPUS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Effective batch size: $((NUM_GPUS * BATCH_SIZE))"
echo "  Max steps: $MAX_STEPS"
echo ""

# Check data exists
DATA_DIR="data/nuscenes_planA_allscenes_T32_stride32_zscore_seed0"
if [ ! -f "$DATA_DIR/train.npy" ]; then
    echo "ERROR: Data not found! Run preprocessing first:"
    echo "  python preprocessing/nuscenes_preprocess_planA.py --dataroot data/nuscenes_meta/v1.0-trainval --normalize zscore"
    exit 1
fi

# Training command
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    python main_train.py \
        --base configs/nuscenes_planA_timedp.yaml \
        --gpus 0, \
        --logdir ./logs/nuscenes_planA \
        -sl 32 \
        -up \
        -nl 16 \
        --batch_size $BATCH_SIZE \
        -lr 0.0001 \
        -s 0 \
        --max_steps $MAX_STEPS \
        --no_test
else
    # Multi-GPU DDP
    # Note: PyTorch Lightning handles DDP internally
    python main_train.py \
        --base configs/nuscenes_planA_timedp.yaml \
        --gpus 0,1,2,3 \
        --logdir ./logs/nuscenes_planA \
        -sl 32 \
        -up \
        -nl 16 \
        --batch_size $BATCH_SIZE \
        -lr 0.0001 \
        -s 0 \
        --max_steps $MAX_STEPS \
        --no_test
fi

echo ""
echo "Training complete! Checkpoints saved in logs/nuscenes_planA/"

