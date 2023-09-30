#!/usr/bin/env bash

set -x
set -e

python3 main_nerf_mask.py \
/data/bhuai/instance_nerf_data/instance_nerf_refined/3dfront_0089_00 \
--workspace /data/bhuai/instance_nerf_data/workspace/instance_nerf_refined/3dfront_0089_00 \
--iters 15000 \
--lr 1e-2 \
--bound 8 \
--gpu 7 \
-O \
--label_regularization_weight 0.0 \
--ckpt latest_model \
--load_model_only \
--test \
--save_path /data/bhuai/instance_nerf_data/video/3dfront_0089_00/inerf \
--train_mask 
# --num_rays $((4096)) \
# --patch_size 8
# --test
# --mask3d /data/bhuai/temp/3dfront_0054_00_filtered.npy \
