import subprocess
import os
import sys
from tqdm import tqdm
import argparse
import glob


# num_part = 4
# scene_prefix = '/data/bhuai/instance_nerf_data/scene_data'
# data_root = '/data/bhuai/instance_nerf_data'

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--part', type=int, default=0)

#     args = parser.parse_args()
#     part = args.part

#     # scenes = os.listdir(scene_prefix)
#     # scenes = sorted(scenes)[part::num_part]
#     scenes = ['3dfront_0075_01']

#     for scene in tqdm(scenes):
#         print(f"Run instance_nerf for scene {scene}")
#         train_dir = os.path.join(scene_prefix, scene, 'train')
#         workspace_dir = os.path.join(data_root, 'workspace', scene)
#         os.makedirs(workspace_dir, exist_ok=True)

#         bashCommand = f'python3 main_nerf_mask.py ' \
#                       f'{train_dir} ' \
#                       f'--workspace {workspace_dir} ' \
#                       f'--iters 30000 ' \
#                       f'--lr 1e-2 ' \
#                       f'--bound 8 ' \
#                       f'--gpu {part + 1} ' \
#                       f'-O ' \
#                       f'--label_regularization_weight 0.0 ' \

#         process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=sys.stdout)
#         output, error = process.communicate()




scene_prefix = '/data/bhuai/instance_nerf_data/instance_nerf'
prev_workspace = '/data/bhuai/instance_nerf_data/workspace/original_nerf'
new_workspace = '/data/bhuai/instance_nerf_data/workspace/instance_nerf'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, default=0)

    args = parser.parse_args()
    part = args.part

    scenes = os.listdir(scene_prefix)
    scenes = sorted(scenes)
    # scenes = sorted(scenes)[part::num_part]

    for scene in tqdm(scenes):
        print(f"Run instance_nerf mask training for scene {scene}")
        train_dir = os.path.join(scene_prefix, scene)
        workspace_dir = os.path.join(new_workspace, scene)
        os.makedirs(workspace_dir, exist_ok=True)

        pattern = os.path.join(prev_workspace, scene, 'checkpoints', 'ngp_ep*.pth')
        checkpoint_list = sorted(glob.glob(pattern))
        if checkpoint_list:
            checkpoint = checkpoint_list[-1]
            print(f"Load checkpoint from {checkpoint}")
        else:
            raise ValueError(f"Checkpoint not found for scene {scene}")

        bashCommand = f'python3 main_nerf_mask.py ' \
                      f'{train_dir} ' \
                      f'--workspace {workspace_dir} ' \
                      f'--iters 25000 ' \
                      f'--lr 1e-2 ' \
                      f'--bound 8 ' \
                      f'--gpu 7 ' \
                      f'-O ' \
                      f'--label_regularization_weight 1.0 ' \
                      f'--ckpt {checkpoint} ' \
                      f'--load_model_only ' \
                      f'--train_mask ' \
                      f'--num_rays 4096 ' \
                      f'--patch_size 8 '

        process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=sys.stdout)
        output, error = process.communicate()
