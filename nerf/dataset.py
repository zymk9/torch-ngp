class NeRFMaskDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test_per_pose=10, n_test_poses=10):
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

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.mask3d = opt.mask3d if self.training or self.type == 'val' else None

        if type == 'trainval' or type == 'test': # in test mode, interpolate new frames
            with open(os.path.join(self.root_path, f'train_transforms.json'), 'r') as f:
                transform = json.load(f)
            with open(os.path.join(self.root_path, f'val_transforms.json'), 'r') as f:
                transform_val = json.load(f)
            transform['frames'].extend(transform_val['frames'])
        elif type == 'test_all':
            with open(os.path.join(self.root_path, f'train_transforms.json'), 'r') as f:
                transform = json.load(f)
        # only load one specified split
        else:
            with open(os.path.join(self.root_path, f'{type}_transforms.json'), 'r') as f:
                transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # find number of instances in this scene
        # if 'bounding_boxes' in transform:
        #     # TODO: instance id might not be consecutive.
        #     self.num_instances = len(transform['bounding_boxes']) + 1 # +1 for the background
        # else:
        #     raise RuntimeError('Failed to load number of instances, please check the transforms.json!')
        if 'num_instances' in transform:
            self.num_instances = transform['num_instances'] + 1 # +1 for the background
        else:
            raise RuntimeError('Failed to load number of instances, please check the transforms.json!')

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
        
        # for colmap, manually interpolate a test set.
        if type == 'test':
            # choose two random poses, and interpolate between.
            f = np.random.choice(frames, n_test_poses, replace=False)
            poses = [nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) for f0 in f]

            self.masks = None
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
            self.masks = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])

                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                with h5py.File(f_path, 'r') as mask_data:
                    mask = np.array(mask_data['cp_instance_id_segmaps'][:])

                assert mask.max() < self.num_instances, \
                    f'Instance id {mask.max()} exceeds the number of instances {self.num_instances - 1}'

                if self.H is None or self.W is None:
                    self.H = mask.shape[0] // downscale
                    self.W = mask.shape[1] // downscale

                if mask.shape[0] != self.H or mask.shape[1] != self.W:
                    mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_AREA)

                self.poses.append(pose)
                self.masks.append(mask)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.masks is not None:
            self.masks = torch.from_numpy(np.stack(self.masks, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.masks.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.masks is not None:
                self.masks = self.masks.to(torch.long).to(self.device)
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
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.masks is not None:
            masks = self.masks[index].to(self.device) # [B, H, W]
            if self.training:
                masks = torch.gather(masks.view(B, -1), 1, rays['inds']) # [B, N]
            results['masks'] = masks

        if self.mask3d is not None:
            results['mask3d_coords'] = self.mask3d_coords
            results['mask3d_labels'] = self.mask3d_labels 
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.masks is not None
        return loader
