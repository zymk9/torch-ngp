#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=/disk1/yliugu/3dfront_sample_mask
WORK_SPACE=./workspace/3dfront_sample_instance
NERF_PRETRAIN_CKPT=./workspace/3dfront_sample/checkpoints/ngp_ep0012.pth

python3 main_nerf_mask.py \
${DATA_ROOT} \
--workspace ${WORK_SPACE} \
--iters 15000 \
--lr 1e-2 \
--bound 8 \
-O \
--label_regularization_weight 1.0 \
--ckpt ${NERF_PRETRAIN_CKPT} \
--load_model_only \
--train_mask \
--num_rays $((4096)) \
--patch_size 16
