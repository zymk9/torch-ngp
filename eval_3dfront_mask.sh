#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=/disk1/yliugu/3dfront_sample_mask
WORK_SPACE=./workspace/3dfront_sample_instance

python3 main_nerf_mask.py \
${DATA_ROOT} \
--workspace ${WORK_SPACE} \
--iters 15000 \
--lr 1e-2 \
--bound 8 \
-O \
--test \
--ckpt latest \
--load_model_only \
--train_mask \
--num_rays $((4096)) \
--patch_size 16


# DATA_ROOT=/data/3dfront_sample_masks
# WORK_SPACE=./workspace/3dfront_sample_instance
# NERF_PRETRAIN_CKPT=./workspace/3dfront_sample_instance/checkpoints/ngp.pth