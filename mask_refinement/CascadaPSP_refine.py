import cv2
import segmentation_refinement as refine
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
import cv2
import time
import matplotlib.pyplot as plt
import os
import h5py
import json

Use_hack = False
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# img_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_sample_mask/validation'
# mask_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_sample_mask/validation'
# output_root = '/disk1/yliugu/3dfront_sample/refined_masks_2d'


scene_img_root = '/disk1/yichen/front3d_ngp/3dfront_0110_00/train/images'
scene_mask_root = '/disk1/yliugu/torch-ngp/workspace/3dfront_sample_1_mask/validation'
scene_output_root = '/disk1/yliugu/3dfront_sample_1/refined_masks_2d'
epoch = 19
scene_json_file = '/disk1/yliugu/3dfront_sample_1/transforms.json'

refiner = refine.Refiner(device='cuda:0') 

os.makedirs(scene_output_root, exist_ok=True)

batch_size = 5


im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])


with open(scene_json_file) as f:
    json_dict = json.load(f)
 


for i in tqdm(json_dict['frames']):
    # mask_file = os.path.join(scene_mask_root, i.replace('.jpg', '.png'))
    file_name = i['file_path'][9:-5]

    img_file = os.path.join(scene_img_root, f'{file_name}.jpg')
    mask_file = os.path.join(scene_mask_root, f'ngp_ep{epoch:04d}_{file_name}_mask.png')
    output_file = os.path.join(scene_output_root, f'{file_name}.hdf5')
    
    mask = cv2.imread(mask_file)[..., 0]
    image = cv2.imread(img_file)

    instance_list = np.unique(mask)
    instance_mask_list = [] 
    init_mask_list = []
    for instance in instance_list:
        if instance == 0:
            continue
        instance_mask_list.append((mask == instance).astype(np.uint8) * 255)
        # init_mask_list.append((sparse_mask == instance).astype(np.uint8) * 255)
    
    instance_masks = np.stack(instance_mask_list, axis=0)
    
    
    
    output = np.zeros_like(mask)
    for j in range(len(instance_list)):
        if instance == 0:
            continue
        instance_output = refiner.refine(image, instance_masks[j-1], fast=False, L=700) / 255
        
        output = (1- instance_output) * output + instance_list[j] * instance_output

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('cp_instance_id_segmaps', data = output.astype(int))
