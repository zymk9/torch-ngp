#!/usr/bin/env bash

set -x
set -e

python3 main_nerf.py \
/disk1/yliugu/Grounded-Segment-Anything/data/3dfront_0089/train \
--workspace ./workspace/3dfront_0089_00 \
--iters 15000 \
--lr 1e-2 \
--bound 8 \
--gpu 5 \
-O \
--label_regularization_weight 1.0 \
--ckpt /disk1/yliugu/torch-ngp/workspace/3dfront_0089_00_t0/checkpoints/ngp_ep0106.pth \
--load_model_only \
--train_mask \
--num_rays $((4096)) \
--patch_size 8
# --test
# --mask3d /data/bhuai/temp/3dfront_0054_00_filtered.npy \
