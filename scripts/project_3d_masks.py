import os

# Util function for loading point clouds
import numpy as np
import cv2
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)


class PointsRenderer_NoDistWeight(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)

        weights[dists2 > 0] = 1.
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images


def grid2world(points, room_bbox, res):
    points /= np.array(res)[None, :]
    points *= room_bbox[1] - room_bbox[0]
    points += room_bbox[0]

    return points


def grid_pts_coord(mask_grid, room_bbox = None):
    x_ = np.linspace(0, mask_grid.shape[0]-1, mask_grid.shape[0])
    y_ = np.linspace(0, mask_grid.shape[1]-1, mask_grid.shape[1])
    z_ = np.linspace(0, mask_grid.shape[2]-1, mask_grid.shape[2])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    pts = np.stack([x, y, z], axis = -1)
    pts = pts.reshape([-1, 3])
    
    if room_bbox is not None:
        pts = grid2world(pts, room_bbox, mask_grid.shape)
    return pts


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def get_world_to_proj_matrix(pose, fl_y, w, h):
    view2proj = np.array([
        [fl_y, 0, w * 0.5],
        [0, -fl_y, h * 0.5],
        [0, 0, 1],
    ])

    world2view = np.linalg.inv(pose)

    return view2proj, world2view


def generate_predicted_grid(masks, scores, threshold = None):
    if threshold is None:
        threshold = masks.shape[-1]
    
    masks_sum = masks.sum(-1)
    
    # obj_index = masks_sum > 0 
    # obj_index = obj_index.reshape(-1)
    
    obj_overlap_index = masks_sum > 1  
    obj_overlap_index = obj_overlap_index.reshape(-1)

    
    instance_masks = masks.reshape([-1, threshold])
    overlap_masks = instance_masks[obj_overlap_index]
    overlap_score = overlap_masks * scores[None, :]
    overlap_score_max = np.max(overlap_score, -1)
    overlap_score_filter = overlap_score >= overlap_score_max[:, None]
    
    instance_masks[obj_overlap_index] = overlap_score_filter
    
    instance_id = np.arange(1, threshold+1)
    instance_masks = instance_masks * instance_id[None,  :]
    instance_masks = instance_masks.sum(-1)
    
    return instance_masks


def project_mask(obj_filename, pose_file, output_folder, grid_file, score_threshold):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    features = np.load(grid_file)
    resolution = features['resolution']
    rgbsigma = features['rgbsigma'].astype(np.float32)
    # alpha = density_to_alpha(rgbsigma[..., -1])
    # rgbsigma[..., -1] = alpha
    rgbsigma = rgbsigma.reshape(resolution[2], resolution[1], resolution[0], -1)
    rgbsigma = torch.from_numpy(rgbsigma)
    rgbsigma = rgbsigma[..., -1:].permute([3,0,1,2])
    rgbsigma = rgbsigma[None, ...] + 10000
    kernel_size = 5    
    weights = torch.ones([1,1, kernel_size,kernel_size,kernel_size]) / (kernel_size**3)

    rgbsigma = F.conv3d(rgbsigma, weights, padding='same')

    alpha = rgbsigma[0,0].reshape(-1)

    alpha = (alpha > 0.) * 255
   
    # alpha = alpha + alpha.min()-10000
    # alpha = alpha / alpha.max()
    # alpha = np.clip(alpha, 0., 1.)
    
    color_dict = np.random.randint(256, size=(40, 3))

    with open(pose_file, 'r') as f:
        data = json.load(f)
        w = int(data['w'])
        h = int(data['h'])
        camera_angle_x = data['camera_angle_x']
        camera_angle_y = data['camera_angle_y']
        fl_y = data['fl_y']
        frames = data['frames']
        room_bbox = np.array(data['room_bbox'])

    pointcloud = np.load(obj_filename)
    masks = pointcloud['masks']
    scores = pointcloud['scores']

    keep = scores > score_threshold
    print(f'keep {keep.sum()} masks: {scores[keep]}')
    print(f'drop {(~keep).sum()} masks: {scores[~keep]}')
    masks = masks[keep]
    scores = scores[keep]
    
    print(masks.shape)
    masks = masks.transpose([1,2,3,0])    
    rgb = generate_predicted_grid(masks, scores)
    
    index = rgb > 0
    verts = grid_pts_coord(masks[..., 0], room_bbox = room_bbox)
    
    rgb = rgb[index] 
    verts = verts[index]
    alpha = alpha[index]
    rgb = torch.from_numpy(rgb).unsqueeze(-1).expand([-1, 3]).float().to(device)
    
    verts = torch.from_numpy(verts).float().to(device)
    alpha = alpha.unsqueeze(-1).expand([-1, 3]).float().to(device)

    # alpha = torch.from_numpy(alpha).unsqueeze(-1).expand([-1, 3]).float().to(device)

    poses = []
    image_names = []
    for f in frames:
        pose = np.array(f['transform_matrix'])
        pose[:, 0] = - pose[:, 0]
        pose[:, 2] = - pose[:, 2]
        poses.append(pose)

        filename = f['file_path']
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        image_names.append(filename)

    poses = np.stack(poses, axis = 0)
    if len(poses.shape) < 3:
        poses = poses[None, ...]
    # poses = poses[:2]
    R = torch.from_numpy(poses[:, :3, :3])
    C = torch.from_numpy(poses[:, :3, 3])
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    focal_length = torch.ones([T.size(0), 1]) * fl_y
    focal_length = focal_length
   
    # verts = [verts for_ in range(T.size(0))]
    raster_settings = PointsRasterizationSettings(
        image_size=(h,w), 
        radius = 0.015,
        points_per_pixel = 1
    )

    batch_size = 100
    R_split = R.split(batch_size)
    T_split = T.split(batch_size)
    kernel = np.ones((7, 7), np.uint8)
    with torch.no_grad():
        images = []
        masks = []
        
        for R,T in tqdm(zip(R_split, T_split)):
            point_cloud = Pointclouds(points=[verts for _ in range(T.size(0))], features=[rgb for _ in range(T.size(0))])
            # point_cloud_alpha = Pointclouds(points=[verts for _ in range(T.size(0))], features=[alpha for _ in range(T.size(0))])
            
            cameras = FoVPerspectiveCameras(fov=camera_angle_y, degrees=False, device=device, R=R, T=T, znear=0.01)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            renderer = PointsRenderer_NoDistWeight(
                rasterizer=rasterizer,
                compositor=AlphaCompositor(background_color=(0, 0, 0))
            )
            
            images.append(renderer(point_cloud).cpu().numpy())
            # masks.append(renderer(point_cloud_alpha).cpu().numpy())
        images = np.concatenate(images, 0).astype(int)
        # masks = np.concatenate(masks, 0).astype(int)
        os.makedirs(output_folder, exist_ok=True)

        # print(np.unique(images.reshape(-1,3)))
        for i in tqdm(range(images.shape[0])):
        # for i in tqdm(range(masks.shape[0])):
            cv2.imwrite(os.path.join(output_folder, image_names[i]+'.png'), images[i])
            # cv2.imwrite(os.path.join(output_folder, image_names[i]), masks[i] )
            for j in np.unique(images[i].reshape(-1,3)):
                output = (images[i] == j) * 255
                # output = cv2.dilate((output * 255).astype(float), kernel, iterations=1)
                cv2.imwrite(os.path.join(output_folder, image_names[i]+f'_{j}.png'), output )


def get_parser():
    parser = argparse.ArgumentParser(description='Project 3D masks to 2D')
    parser.add_argument('--mask_dir', type=str, help='path to 3D masks')
    parser.add_argument('--scene_dir', type=str, help='path to scene data')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--feature_dir', type=str, help='path to feature directory')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    mask_dir = args.mask_dir
    scene_dir = args.scene_dir
    output_dir = args.output_dir
    feature_dir = args.feature_dir

    os.makedirs(output_dir, exist_ok=True)

    scenes = os.listdir(mask_dir)
    scenes.sort()

    for scene in tqdm(scenes):
        feature_path = os.path.join(feature_dir, scene+'.npz')
        mask_path = os.path.join(mask_dir, scene+'.npz')
        output_path = os.path.join(output_dir, scene)

        for s in scene_dir:
            pose_path = os.path.join(s, scene, 'train', 'transforms.json')
            if os.path.exists(pose_path):
                break

        if not os.path.exists(pose_path):
            print(f'pose path not found: {pose_path}')
            continue

        project_mask(mask_path, pose_path, output_path, feature_path, score_threshold=0.5)
        