#!/usr/bin/env bash

set -x
set -e

python3 main_nerf_mask.py \
/data/bhuai/temp/3dfront_0110_00 \
--workspace ./workspace/3dfront_0110_00 \
--iters 15000 \
--lr 1e-2 \
--bound 8 \
--gpu 5 \
-O \
--label_regularization_weight 1.0 \
--ckpt /data/bhuai/NeRF_RCNN/dependencies/torch-ngp/workspace/3dfront_0110_00/checkpoints/ngp_ep0102.pth \
--load_model_only \
--train_mask \
--num_rays $((4096)) \
--patch_size 8
# --test
# --mask3d /data/bhuai/temp/3dfront_0054_00_filtered.npy \
