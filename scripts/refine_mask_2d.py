import cv2
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
from tqdm import tqdm

mask_root = '/data/bhuai/instance_nerf_data/workspace_new/instance_nerf_masks'
img_root = '/data/bhuai/instance_nerf_data/front3d_extra_data'
output_root = '/data/bhuai/instance_nerf_data/workspace_new/nerf_refined'

with open('/data/bhuai/instance_nerf_data/selected.txt', 'r') as f:
    scenes = f.readlines()
    scenes = [x.strip() for x in scenes]
    scenes = ['3dfront_' + x for x in scenes]

for scene in tqdm(scenes):
    mask_dir = os.path.join(mask_root, scene)
    img_dir = os.path.join(img_root, scene, 'train', 'images')
    output_dir = os.path.join(output_root, scene)

    os.makedirs(output_dir, exist_ok=True)

    img_list = os.listdir(img_dir)
    img_list.sort()

    refiner = refine.Refiner(device='cuda:0')

    n_iter = 1

    for i in tqdm(img_list):
        mask_file = os.path.join(mask_dir, i.replace('.jpg', '.png'))
        img_file = os.path.join(img_dir, i)

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

            output_file = os.path.join(output_dir, i.replace('.jpg', f'_{instance}.png'))
            cv2.imwrite(output_file, instance_mask)
