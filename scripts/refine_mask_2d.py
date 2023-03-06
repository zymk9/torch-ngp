import cv2
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
img_root = '/data/yliugu/front3d_ngp/3dfront_0110_00/train/images'
mask_root = '/data/bhuai/temp/3dfront_0110_00/nerf_masks'
output_root = '/data/bhuai/temp/3dfront_0110_00/nerf_refined'

os.makedirs(output_root, exist_ok=True)

img_list = os.listdir(img_root)
img_list.sort()

refiner = refine.Refiner(device='cuda:0')

n_iter = 1

for i in tqdm(img_list):
    mask_file = os.path.join(mask_root, i.replace('.jpg', '.png'))
    img_file = os.path.join(img_root, i)

    mask = cv2.imread(mask_file)
    image = cv2.imread(img_file)

    mask = mask[..., 0]
    instance_list = np.unique(mask)

    for instance in instance_list:
        if instance == 0:
            continue

        instance_mask = (mask == instance).astype(np.uint8) * 255
        for _ in range(n_iter):
            instance_mask = refiner.refine(image, instance_mask, fast=False, L=900) 

        output_file = os.path.join(output_root, i.replace('.jpg', f'_{instance}.png'))
        cv2.imwrite(output_file, instance_mask)
