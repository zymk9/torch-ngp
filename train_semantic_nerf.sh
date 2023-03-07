#!/usr/bin/env bash

set -x
set -e

python3 main_nerf_mask.py \
./instance_nerf_data/scene_data/3dfront_0004_00 \
--workspace ./workspace/semantic_nerf/3dfront_0004_00_rgb \
--iters 8000 \
--lr 1e-2 \
--bound 8 \
--gpu 1 \
-O \
--train_mask \
--train_semantic \
--semantic_loss_weight 0.0 \
--label_regularization_weight 0.0 \
--mask3d_loss_weight 0.0 \
--num_rays $((4096)) \
--patch_size 8
# --test
# --mask3d /data/bhuai/temp/3dfront_0054_00_filtered.npy \
