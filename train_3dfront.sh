#!/usr/bin/env bash

set -x
set -e



DATA_ROOT=/disk1/yliugu/3dfront_sample/train
WORK_SPACE=./workspace/3dfront_sample

python main_nerf_mask.py \
${DATA_ROOT} \
--workspace ${WORK_SPACE} \
-O \
--iters 15000 \
--min_near 0.1 \
--upsample_steps 512 \
--bound 8


# DATA_ROOT=/data/3dfront_sample_masks
# WORK_SPACE=./workspace/3dfront_sample