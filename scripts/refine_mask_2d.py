import cv2
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os
import argparse
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description='Refine 2D instance masks')
    parser.add_argument('--mask_root', type=str)
    parser.add_argument('--img_root', type=str)
    parser.add_argument('--output_root', type=str)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    mask_root = args.mask_root
    img_root = args.img_root
    output_root = args.output_root

    scenes = os.listdir(mask_root)
    scenes.sort()

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
