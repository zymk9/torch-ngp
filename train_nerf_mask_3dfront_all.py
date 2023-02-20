import os
import json
import subprocess

with open('/data2/jhuangce/torch-ngp/3dfront_mask_data/dataset_split.json', 'r') as f:
    dataset_split = json.load(f)
scene_list = dataset_split['test']
scene_list = scene_list[3:]

for scene_name in scene_list:
    subprocess.run(['python3', 'main_nerf_mask.py',
                    f'/data2/jhuangce/torch-ngp/FRONT3D_render/finished/{scene_name}/train',
                    '--workspace', f'./workspace/3dfront_nerf/{scene_name}',
                    '--iters', '20000',
                    '--lr', '1e-2',
                    '--cuda_ray',
                    '--dataset_name', '3dfront',
                    '--wandb'])
    # subprocess.run(['python3', 'main_nerf_mask.py',
    #                 f'/data2/jhuangce/3dfront_mask_data/masks_2d/{scene_name}',
    #                 '--workspace', f'./workspace/3dfront_nerf_mask/{scene_name}',
    #                 '--iters', '10000',
    #                 '--lr', '1e-2',
    #                 '--ckpt', f'./workspace/3dfront_nerf/{scene_name}/checkpoints/ngp.pth',
    #                 '--load_model_only',
    #                 '--train_mask',
    #                 '--dataset_name', '3dfront',
    #                 '--wandb'])