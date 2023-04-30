import os
import cv2
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation


import torch
from torch.utils.data import DataLoader

from .utils import get_rays, preprocess_feature
from .provider import nerf_matrix_to_ngp

class NeRFSAMDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test_per_pose=10, n_test_poses=2, feature_dim=256,
                 feature_size=64, sam_image_size= 1024, dataset_name='nerf'):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.augmentation = opt.augmentation

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.mask3d = opt.mask3d if self.training or self.type == 'val' else None
        self.feature_dim = feature_dim
        self.feature_size = feature_size
        self.sam_image_size = sam_image_size
        
        self.dataset_name = dataset_name


        if type == 'trainval' or type == 'test': # in test mode, interpolate new frames
            if dataset_name == '3dfront':
                self.root_path = os.path.join(self.root_path, 'train')
            with open(os.path.join(self.root_path, f'transforms_train_sam.json'), 'r') as f:
                transform = json.load(f)
            # with open(os.path.join(self.root_path, f'transforms_val_sam.json'), 'r') as f:
            #     transform_val = json.load(f)
            # transform['frames'].extend(transform_val['frames'])
        elif type == 'test_all':
            if dataset_name == '3dfront':
                self.root_path = os.path.join(self.root_path, 'val')
            with open(os.path.join(self.root_path, f'transforms_val_sam.json'), 'r') as f:
                transform = json.load(f)
        # only load one specified split
        else:
            if dataset_name == '3dfront':
                self.root_path = os.path.join(self.root_path, self.type)
            with open(os.path.join(self.root_path, f'transforms_{type}_sam.json'), 'r') as f:
                transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        if 'room_bbox' in transform:
            room_bbox = np.array(transform['room_bbox'])
            self.offset = -(room_bbox[0] + room_bbox[1]) * 0.5 * self.scale

        # read 3d mask constraints
        if self.mask3d is not None:
            assert 'room_bbox' in transform, '3d mask requires room_bbox in transforms.json!'

            mask3d = np.load(self.mask3d)
            if mask3d.max() >= self.num_instances:
                raise RuntimeError(f'3d mask has too many instances {mask3d.max()}, '
                    f'only {self.num_instances-1} instances are loaded!')

            assert len(mask3d.shape) == 3, f'3d mask should be [W, L, H], got {mask3d.shape}'

            res = mask3d.shape
            x = np.linspace(room_bbox[0][0], room_bbox[1][0], res[0]) * self.scale + self.offset[0]
            y = np.linspace(room_bbox[0][1], room_bbox[1][1], res[1]) * self.scale + self.offset[1]
            z = np.linspace(room_bbox[0][2], room_bbox[1][2], res[2]) * self.scale + self.offset[2]
            x, y, z = np.meshgrid(x, y, z, indexing='ij')
            coords = np.stack([x, y, z], -1).reshape(-1, 3)

            mask3d = mask3d.reshape(-1)
            keep = mask3d > 0
            mask3d = mask3d[keep]
            coords = coords[keep]
            
            self.mask3d_labels = torch.from_numpy(mask3d).to(torch.long).to(device)
            self.mask3d_coords = torch.from_numpy(coords).to(torch.float).to(device)
        
        # read images
        frames = transform["frames"]
        if self.type == 'val':
            frames = frames[10:]
        # frames = frames[:10]
        
        # for colmap, manually interpolate a test set.
        if type == 'test':
            # choose two random poses, and interpolate between.
            f = np.random.choice(frames, n_test_poses, replace=False)

            poses = [nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) for f0 in f]

            self.features = None
            self.poses = []
 
            for k in range(len(poses) - 1):
                pose0 = poses[k]
                pose1 = poses[k + 1]
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)
    
                for i in range(n_test_per_pose + 1):
                    ratio = np.sin(((i / n_test_per_pose) - 0.5) * np.pi) * 0.5 + 0.5
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)
        else:
            self.poses = []
            self.features = None
            self.images = None
            
            
            if not self.opt.online:
                if opt.load_feature:
                    self.features = []
                else:
                    self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])

                if not os.path.exists(f_path) :
                    continue
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
                self.poses.append(pose)
                if self.type == 'test_all':
                    continue
                # Online mode will not preload anything
                if self.opt.online:
                    continue
                if opt.load_feature:
                    with np.load(f_path) as data:
                        res = data['res']
                        feature = data['embedding']
                        feature = feature.reshape(res.tolist()).astype(np.float32)
                        feature = feature.transpose(1, 2, 0)

                    assert feature.shape[-1] == self.feature_dim , \
                        f'feature dimension does not match.'
                    assert feature.shape[0] == self.feature_size and feature.shape[1] == self.feature_size, \
                        f'feature size does not match.'
                    # resize the features to fit the image size
                    if feature.shape[0] != self.H or feature.shape[1] != self.W:
                        feature = preprocess_feature(feature, self.H, self.W)
                    
                    self.features.append(torch.from_numpy(feature))
                else:
                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                    if self.H is None or self.W is None:
                        self.H = image.shape[0] // downscale
                        self.W = image.shape[1] // downscale
                        
                    # add support for the alpha channel as a mask.
                    if image.shape[-1] == 3: 
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                    if image.shape[0] != self.H or image.shape[1] != self.W:
                        image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                        
                    image = image.astype(np.float32) / 255 # [H, W, 3/4]
                    self.images.append(image)

        self.num_data = len(self.poses)
        
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.type == 'test_all':
            self.features = None
            
        if self.features is not None:
            self.features = np.stack(self.features, axis=0)
            self.features = torch.from_numpy(self.features) # [N, H, W, C]
        if self.images is not None:
            self.images = np.stack(self.images, axis=0)
            self.images = torch.from_numpy(self.images) # [N, H, W, 3]
            
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        # print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.features.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None


        if self.preload:
            print('load poses to GPU')
            self.poses = self.poses.to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):
        B = len(index) # a list of length 1
        # random pose without gt images.
        
        scale = 1
        if self.opt.train_sam:
            scale = max(self.H, self.W) / 64

        if index[0] >= len(self.poses):
            perm = torch.randperm(self.poses.size(0))
            idx = perm[:B+1]
            poses = self.poses[idx].cpu().numpy()
            
            pose0 = poses[0]
            pose1 = poses[1]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            ratio = np.sin(((1 / 2) - 0.5) * np.pi) * 0.5 + 0.5
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = slerp(ratio).as_matrix()
            pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
            
            
            poses = torch.from_numpy(pose[None,...]).to(self.device) # [B, 4, 4]
            error_map = None if self.error_map is None else self.error_map[index]
            
            
            intrinsics = np.floor(self.intrinsics / scale)
            H = int(np.floor(self.H / scale))
            W = int(np.floor(self.W / scale))
            rays = get_rays(poses, intrinsics, H, W, self.num_rays, error_map, self.opt.patch_size)
            full_rays = get_rays(poses, self.intrinsics, self.H, self.W, -1, error_map, self.opt.patch_size)

            results = {
                'H': H,
                'W': W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                # 'index': rays['inds'],
                'full_H': self.H,
                'full_W': self.W,
                'full_rays_o': full_rays['rays_o'],
                'full_rays_d': full_rays['rays_d'],
                'poses': poses,
                'augment': True
            }
            return results
  
        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        
        intrinsics = np.floor(self.intrinsics / scale)
        H = int(np.floor(self.H / scale))
        W = int(np.floor(self.W / scale))
        rays = get_rays(poses, intrinsics, H, W, self.num_rays, error_map, self.opt.patch_size)
        full_rays = get_rays(poses, self.intrinsics, self.H, self.W, -1, error_map, self.opt.patch_size)

        results = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            # 'index': rays['inds'],
            'full_H': self.H,
            'full_W': self.W,
            'full_rays_o': full_rays['rays_o'],
            'full_rays_d': full_rays['rays_d'],
            'poses': poses,
            'augment': False
        }

        if self.features is not None:
            feature = self.features[index].to(self.device) # [B, H, W, C]
            if self.training:
                feature = torch.gather(feature.view(B, -1, self.feature_dim), 1, torch.stack(self.feature_dim * [rays['inds']], -1)) # [B, N, C]
            results['feature'] = feature
            
        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
            
            
        if self.mask3d is not None:
            results['mask3d_coords'] = self.mask3d_coords
            results['mask3d_labels'] = self.mask3d_labels 
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

                        
        return results

    def dataloader(self):
        size = self.num_data
        if self.training and self.augmentation > 0:
            size += round(size * self.augmentation) # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate,  num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.features is not None
        return loader
