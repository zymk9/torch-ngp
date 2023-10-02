import subprocess
import os
import sys
from tqdm import tqdm
import argparse
import glob
from tqdm.contrib.concurrent import process_map
from functools import partial
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd
import time

from multiprocessing import set_start_method
set_start_method("spawn", force=True)


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf-8')),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' MiB')))
    # idx = gpu_df['memory.free'].idxmax()
    # print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    # return idx, gpu_df.iloc[idx]['memory.free']

    id2free = {i: gpu_df.iloc[i]['memory.free'] for i in range(len(gpu_df))}
    return id2free


def train_rgb_single(train_dir, workspace_dir, gpu_id):
    print(f"Train instance_nerf rgb density for scene {train_dir}, using GPU {gpu_id}")
    bashCommand = f'python3 main_nerf_mask.py ' \
                  f'{train_dir} ' \
                  f'--workspace {workspace_dir} ' \
                  f'--iters 30000 ' \
                  f'--lr 1e-2 ' \
                  f'--bound 8 ' \
                  f'--gpu {gpu_id} ' \
                  f'-O ' \
                  f'--label_regularization_weight 0.0 '

    try:
        os.makedirs(workspace_dir, exist_ok=True)
        process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=subprocess.DEVNULL)
        output, error = process.communicate()
        
        print(f"Finish training rgb density for scene {train_dir}")
        return True

    except Exception as e:
        print(e)
        print(f"Error in training rgb density for scene {train_dir}")
        return False


def train_rgb(rank, world_size, train_dirs, workspace_dirs):
    for i in tqdm(range(rank, len(train_dirs), world_size)):
        train_dir = train_dirs[i]
        workspace_dir = workspace_dirs[i]
        train_rgb_single(train_dir, workspace_dir, rank)


def train_mask_single(scene, scene_prefix, new_workspace, prev_workspace, gpu_id, num_iters, reg_weight):
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
                  f'--iters {num_iters} ' \
                  f'--lr 1e-2 ' \
                  f'--bound 8 ' \
                  f'--gpu {gpu_id} ' \
                  f'-O ' \
                  f'--label_regularization_weight {reg_weight} ' \
                  f'--ckpt {checkpoint} ' \
                  f'--load_model_only ' \
                  f'--train_mask ' \
                  f'--num_rays 4096 ' \
                  f'--patch_size 8 '

    process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=sys.stdout)
    output, error = process.communicate()


def train_mask(rank, world_size, scenes, scene_prefix, new_workspace, prev_workspace, num_iters, reg_weight):
    for i in tqdm(range(rank, len(scenes), world_size)):
        scene = scenes[i]
        train_mask_single(scene, scene_prefix, new_workspace, prev_workspace, rank, num_iters, reg_weight)


def test_mask_single(scene, scene_prefix, workspace, gpu_id, reg_weight):
    print(f"Run instance_nerf mask testing for scene {scene}")
    train_dir = os.path.join(scene_prefix, scene)
    workspace_dir = os.path.join(workspace, scene)
    os.makedirs(workspace_dir, exist_ok=True)

    pattern = os.path.join(workspace, scene, 'checkpoints', 'ngp_ep*.pth')
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
                  f'--gpu {gpu_id} ' \
                  f'-O ' \
                  f'--label_regularization_weight {reg_weight} ' \
                  f'--ckpt {checkpoint} ' \
                  f'--load_model_only ' \
                  f'--train_mask ' \
                  f'--num_rays 4096 ' \
                  f'--patch_size 8 ' \
                  f'--test '

    process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=sys.stdout)
    output, error = process.communicate()


def test_mask(rank, world_size, scenes, scene_prefix, workspace, reg_weight):
    for i in tqdm(range(rank, len(scenes), world_size)):
        scene = scenes[i]
        test_mask_single(scene, scene_prefix, workspace, rank, reg_weight)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngp_option', type=str, default='train_rgb', choices=['train_rgb', 'train_mask', 'test_mask'], 
                        help='A dataset name for wandb logging')
    args = parser.parse_args()
    
    scene_prefix = '/data/bhuai/instance_nerf_data/instance_nerf_refined/front3d_new'
    prev_workspace = '/data/bhuai/instance_nerf_data/workspace_new/instance_nerf'
    new_workspace = '/data/bhuai/instance_nerf_data/workspace_new/instance_nerf_refined'
    data_root = '/data/bhuai/instance_nerf_data'

    scenes = os.listdir(scene_prefix)
    scenes = sorted(scenes)


    if args.ngp_option == 'train_rgb':
        train_dirs = []
        workspace_dirs = []
        for scene in scenes:
            train_dir = os.path.join(scene_prefix, scene, 'train')
            workspace_dir = os.path.join(data_root, 'workspace_new', scene)
            train_dirs.append(train_dir)
            workspace_dirs.append(workspace_dir)

        fn = partial(train_rgb, world_size=8, train_dirs=train_dirs, workspace_dirs=workspace_dirs)
    elif args.ngp_option == 'train_mask':
        fn = partial(train_mask, world_size=8, scenes=scenes, scene_prefix=scene_prefix, new_workspace=new_workspace, 
                    prev_workspace=prev_workspace, num_iters=20000, reg_weight=1.0)
    elif args.ngp_option == 'test_mask':
        fn = partial(test_mask, world_size=8, scenes=scenes, scene_prefix=scene_prefix, workspace=new_workspace, 
                 reg_weight=1.0)

    process_map(fn, range(8), max_workers=8)

