
import os
import tqdm
import imageio
import json
import wandb

import numpy as np

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .utils import Trainer, preprocess_feature, postprocess_feature, print_shape, get_rays, linear_to_srgb, project_to_2d, project_to_3d, custom_meshgrid
from .helper import show_points, show_mask

class Cache:
    def __init__(self, size=100):
        self.size = size
        self.data = {}
        self.key = 0
    
    def full(self):
        return len(self.data) == self.size

    def insert(self, x):
        self.data[self.key] = x
        self.key = (self.key + 1) % self.size
    
    def get(self, key=None):
        if key is None:
            key = np.random.randint(0, len(self.data))
        return self.data[key]
    

class SAMTrainer(Trainer):
    def __init__(self, *args, feature_size=64, predictor = None, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: overflow issue, currently just use the same color map as gt visualization
        self.color_map = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8)

        self.feature_dim = self.model.feature_dim
        self.feature_size = feature_size
        self.predictor = predictor

        if self.opt.mask_loss_weight > 0:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()


        if self.opt.consistency_loss_weight > 0:
            self.consistency_criterion = torch.nn.BCEWithLogitsLoss()

        if predictor is not None:
            self.predictor.model.to(self.device)

        # freeze rgb and density
        self.model.encoder.requires_grad_(False)
        self.model.sigma_net.requires_grad_(False)
        self.model.encoder_dir.requires_grad_(False)
        self.model.color_net.requires_grad_(False)

        self.cache = Cache(self.opt.cache_size)

    ### ------------------------------

    def postprocess(self, features):
        target_feature_size = self.feature_size
        f_h, f_w = features.shape[1:]
        max_length = max(f_h, f_w)
        h, w = int(np.floor(target_feature_size * f_h / max_length)), int(np.floor(target_feature_size * f_w / max_length))
        features = cv2.resize(features.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
        features = features.transpose(2, 0, 1)
        features = torch.from_numpy(features[None, ...])
        
        padh = target_feature_size - h
        padw = target_feature_size - w
        return F.pad(features, (0, padw, 0, padh)).numpy()[0]

    def preprocess_masks(self, mask_input):
        B, H, W = mask_input.shape
        max_l = max(H, W)
        target_size = 256
        resized_size = (int(H / max_l * target_size), int(W / max_l * target_size))
        mask_input = F.interpolate(mask_input[:, None, ...], resized_size, mode="bilinear", align_corners=False)
        
        padh = target_size - resized_size[0]
        padw = target_size - resized_size[1]
        mask_input = F.pad(mask_input, (0, padw, 0, padh))
        mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
        return mask_input_torch

    def label_regularization(self, depth, pred_masks, H , W):
        '''
        depth: [B, N]
        pred_masks: [B, N, num_instances]
        '''
        pred_masks = pred_masks.view(-1, H, W)[None, ...].contiguous()

        diff_x = pred_masks[:, :, :, 1:] - pred_masks[:, :, :, :-1]
        diff_y = pred_masks[:, :, 1:, :] - pred_masks[:, :, :-1, :]


        depth = depth.view(-1, H, W) # [B, patch_size, patch_size]

        depth_diff_x = depth[:, :, 1:] - depth[:, :, :-1]
        depth_diff_y = depth[:, 1:, :] - depth[:, :-1, :]
        weight_x = torch.exp(-(depth_diff_x * depth_diff_x)).unsqueeze(1).expand_as(diff_x)
        weight_y = torch.exp(-(depth_diff_y * depth_diff_y)).unsqueeze(1).expand_as(diff_y)

        diff_x = diff_x * diff_x * weight_x
        diff_y = diff_y * diff_y * weight_y

        smoothness_loss = torch.sum(diff_x) / torch.sum(weight_x) + torch.sum(diff_y) / torch.sum(weight_y)

        return smoothness_loss

    def generate_prompt(self, H, W, padding=30):
        if self.opt.prompt_sample == 'uniform':
            y = torch.linspace(padding, H - padding, steps=self.opt.sample_step)
            x = torch.linspace(padding, W - padding, steps=self.opt.sample_step)
            
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            input_point = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], -1)
            input_label = torch.ones(input_point.shape[0])

        elif self.opt.prompt_sample == 'random':
            y = torch.randint(padding, H - padding, (self.opt.sample_step**2,))
            x = torch.randint(padding, W - padding, (self.opt.sample_step**2,))
            
            input_point = torch.stack([x, y], -1)
            input_label = torch.ones(input_point.shape[0])

        return input_point, input_label


    
    def sam_forward(self, features, H, W, point_coords=None, point_labels=None, mask_input=None):
        predictor = self.predictor
        sam = self.predictor.model
        if point_coords is not None and point_labels is not None:
            point_coords = predictor.transform.apply_coords_torch(point_coords, (H, W))
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=predictor.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
            coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
            points = (coords_torch, labels_torch)
        else:
            points = None
        
        if mask_input is not None:
            B, H, W = mask_input.shape
            max_l = max(H, W)
            target_size = 256
            resized_size = (int(H / max_l * target_size), int(W / max_l * target_size))
            mask_input = F.interpolate(mask_input[:, None, ...], resized_size, mode="bilinear", align_corners=False)
            
            padh = target_size - resized_size[0]
            padw = target_size - resized_size[1]
            mask_input = F.pad(mask_input, (0, padw, 0, padh))
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
        else:
            mask_input_torch = None
            

        # Embed prompts
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_input_torch,
        )

        # Predict masks
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=features,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        # Upscale the masks to the original image resolution
        masks = sam.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)

        return masks, iou_predictions, low_res_masks
    
    def sam_batch_predict(self, H, W, points=None, masks_input=None, image=None):

        if image is not None:
            self.predictor.set_image(image[0])
        transformed_points = self.predictor.transform.apply_coords(points, (H, W))
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)


        masks, iou_preds, low_res_masks = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            mask_input=masks_input,
            multimask_output=True,
            return_logits=False,
        )

        return masks, iou_preds, low_res_masks
    
    
    def predict_mask_from_feature(self, pred_features, H, W, num_batches, input_point=None, input_label=None, masks=None, points_per_batch=64):
        pred_features = postprocess_feature(pred_features, 64)
        nerf_masks = []
        for i in range(num_batches):
            batch_points = input_point[i*points_per_batch:(i+1)*points_per_batch, :] if input_point is not None else None
            batch_labels = input_label[i*points_per_batch:(i+1)*points_per_batch] if input_label is not None else None
            batch_masks = masks[i*points_per_batch:(i+1)*points_per_batch, ...] if masks is not None else None
            pred_masks, _, _ = self.sam_forward(
                pred_features, H, W,
                point_coords=batch_points, 
                point_labels=batch_labels,
                mask_input=batch_masks 
                
            )

            nerf_masks.append(pred_masks)

        nerf_masks = torch.cat(nerf_masks, dim=0)
        return nerf_masks

    def predict_gt_masks(self, image, H, W, num_batches, input_point, points_per_batch=64):
        gt_masks = []

        self.predictor.set_image(image)
        for i in range(num_batches):
            masks, _, _ = self.sam_batch_predict(
                H, W,
                input_point[i*points_per_batch:(i+1)*points_per_batch, :].numpy()
            )   # [B, 3, H, W]
            gt_masks.append(masks[:, -1])  # use the last level mask

        gt_masks = torch.cat(gt_masks, dim=0).to(torch.float32)
        return gt_masks
    def mask_loss(self, image, pred_features, H, W, points_per_batch=64):
        input_point, input_label = self.generate_prompt(H, W)
        num_batches = int(np.ceil(input_point.size(0) / points_per_batch))

        with torch.no_grad():
            gt_masks = []

            self.predictor.set_image(image)
            for i in range(num_batches):
                masks, _, _ = self.sam_batch_predict(
                    H, W,
                    input_point[i*points_per_batch:(i+1)*points_per_batch, :].numpy(), 
                )   # [B, 3, H, W]

                gt_masks.append(masks[:, -1].flatten(1))  # use the last level mask

            gt_masks = torch.cat(gt_masks, dim=0).to(torch.float32)

        pred_features = postprocess_feature(pred_features, 64)
        nerf_masks = []
        for i in range(num_batches):
            pred_masks, _, _ = self.sam_forward(
                pred_features, H, W,
                input_point[i*points_per_batch:(i+1)*points_per_batch, :].numpy(), 
                input_label[i*points_per_batch:(i+1)*points_per_batch].numpy(), 
                
            )

            nerf_masks.append(pred_masks[:, -1].flatten(1))

        nerf_masks = torch.cat(nerf_masks, dim=0)
        loss = self.mask_criterion(nerf_masks, gt_masks)
        return loss, nerf_masks
    
    
    def point_projection(self, point, data, depth, sec_depth, threshold = 0.1):
        
        H,W = depth.shape
        point = point.to(torch.long).to(self.device)
        pts_3D = project_to_3d(point, data['pose'], data['intrinsics'], depth)
        pts_2D, pts_depth = project_to_2d(pts_3D, data['sec_pose'], data['intrinsics'])
        im_depth = sec_depth[pts_2D[...,1].clamp(0, H-1).to(torch.long), pts_2D[..., 0].clamp(0, W-1).to(torch.long)]
        
        valid_0 = (pts_depth > 0.) & (torch.abs(im_depth- pts_depth) < threshold ) 
        valid_1 = (pts_2D[..., 0] >= 0) & (pts_2D[..., 0] < W)
        valid_2 = (pts_2D[..., 1] >= 0) & (pts_2D[..., 1] < H)
        return pts_2D, (valid_0 & valid_1) & valid_2
    
    def mask_projection(self, masks, data, depth, sec_depth, threshold = 0.05):
        B, H, W = masks.shape
        i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=self.device), torch.linspace(0, H-1, H, device=self.device)) # float
        # i = i[None,...].repeat(masks.shape[0],1,1)
        # j = j[None,...].repeat(masks.shape[0],1,1)


        i = i.t().reshape([-1])
        j = j.t().reshape([-1])
        

        depth = depth.to(self.device)
        sec_depth = sec_depth.to(self.device)
        
        masks_pts = torch.stack([i,j], -1).to(self.device).to(torch.long)
        
        pts_3D = project_to_3d(masks_pts, data['pose'], data['intrinsics'], depth)
        pts_2D, pts_depth = project_to_2d(pts_3D, data['sec_pose'], data['intrinsics'])

        # pts_2D = pts_2D.reshape([H, W, 2])
        # pts_depth = pts_depth.reshape([H, W])
        
        im_depth = sec_depth[pts_2D[...,1].clamp(0, H-1).to(torch.long), pts_2D[..., 0].clamp(0, W-1).to(torch.long)]
        valid_0 = (pts_depth > 0.) & (torch.abs(im_depth- pts_depth) < threshold ) 
        valid_1 = (pts_2D[..., 0] >= 0) & (pts_2D[..., 0] < W)
        valid_2 = (pts_2D[..., 1] >= 0) & (pts_2D[..., 1] < H)
        valid = (valid_0 & valid_1) & valid_2
        valid = valid[None,...].expand(B, -1)
        
        masks_binary = masks > 0.0
        masks_binary = masks_binary.reshape([B, -1])
        valid = masks_binary & valid
        
        pts_2D = pts_2D[None,...].expand(B, -1, -1).reshape([-1, 2])
        valid = valid.reshape(-1)
        pts_2D[torch.logical_not(valid)] = torch.tensor([W, H]).to(self.device)
        pts_2D = pts_2D.reshape([B, -1, 2])

        sec_masks = torch.zeros([B, H+1, W+1]) - 10.
        sec_masks[torch.arange(B)[:, None].expand(-1, H*W), pts_2D[..., 1], pts_2D[..., 0]] = 10.
        # sec_masks = sec_masks.reshape(-1)
        sec_masks = sec_masks[:, :H, :W]
        
        return sec_masks
        
    def get_sam_features_online(self, data):
        H_full, W_full = data['full_H'], data['full_W']
        H, W = data['H'], data['W'] # feature size

        rays_full_o = data['full_rays_o'] # [B, N, 3]
        rays_full_d = data['full_rays_d'] # [B, N, 3]

        is_training = self.model.training
        cuda_ray = self.model.cuda_ray
        self.model.cuda_ray = False # temporary fix
        self.model.eval()

        gt_features = []
        sec_gt_features = []
        with torch.no_grad():
            img_outputs = self.model.render(rays_full_o, rays_full_d, render_feature=False, staged=True, 
                                            perturb=False, force_all_rays=True, **vars(self.opt))
            imgs = img_outputs['image'].reshape(-1, H_full, W_full, 3).detach().cpu().numpy()
            imgs = (imgs * 255).astype(np.uint8) # [B, H_full, W_full, 3]
            cv2.imwrite('rgb.png', cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))  
            # temp_dir = os.path.join(self.workspace, 'sam_imgs')
            # os.makedirs(temp_dir, exist_ok=True)

            for i in range(imgs.shape[0]):
                # imageio.imsave(os.path.join(temp_dir, f'{i}.png'), imgs[i])
                self.predictor.set_image(imgs[i])
                emb = self.predictor.get_image_embedding().detach()
                if emb.shape[-2] != H or emb.shape[-1] != W:
                    emb = preprocess_feature(emb, H, W)

                emb = emb[0].permute(1, 2, 0) # [H, W, feature_dim]
                gt_features.append(emb)

            torch.cuda.empty_cache()
            
            if self.opt.consistency_loss_weight:
                rays_full_o = data['sec_full_rays_o'] # [B, N, 3]
                rays_full_d = data['sec_full_rays_d'] # [B, N, 3]
                sec_img_outputs = self.model.render(rays_full_o, rays_full_d, render_feature=False, staged=True, 
                                perturb=False, force_all_rays=True, **vars(self.opt))
                sec_imgs = sec_img_outputs['image'].reshape(-1, H_full, W_full, 3).detach().cpu().numpy()
                sec_imgs = (sec_imgs * 255).astype(np.uint8) # [B, H_full, W_full, 3]
                cv2.imwrite('sec_rgb.png', cv2.cvtColor(sec_imgs[0], cv2.COLOR_RGB2BGR))  
                for i in range(imgs.shape[0]):
                    # imageio.imsave(os.path.join(temp_dir, f'{i}.png'), imgs[i])
                    self.predictor.set_image(sec_imgs[i])
                    emb = self.predictor.get_image_embedding().detach()
                    if emb.shape[-2] != H or emb.shape[-1] != W:
                        emb = preprocess_feature(emb, H, W)
                    emb = emb[0].permute(1, 2, 0) # [H, W, feature_dim]
                    sec_gt_features.append(emb)
        gt_features = torch.stack(gt_features, dim=0).to(self.device) # [B, H, W, feature_dim]
        data['feature'] = gt_features
        data['images'] = imgs
        data['depth'] = img_outputs['t_depth'].reshape(-1, H_full, W_full).detach().cpu().numpy()
        data['sec_depth'] = sec_img_outputs['t_depth'].reshape(-1, H_full, W_full).detach().cpu().numpy()
        if is_training:
            self.model.train()
        self.model.cuda_ray = cuda_ray
        if self.opt.consistency_loss_weight:
            sec_gt_features = torch.stack(sec_gt_features, dim=0).to(self.device) # [B, H, W, feature_dim]
            data['sec_feature'] = sec_gt_features
            data['sec_images'] = sec_imgs
        return gt_features


    def generate_project_mask(self, data, nerf_point_masks, input_point, num_batches):
        with torch.no_grad():
            depth, sec_depth = torch.as_tensor(data['depth'][0]).to(self.device), torch.as_tensor(data['sec_depth'][0]).to(self.device)
            sec_input_point, valid = self.point_projection(input_point, data, depth, sec_depth )
            sec_input_point = sec_input_point[valid]
            nerf_point_masks = nerf_point_masks[valid]
            # nerf_point_masks = nerf_point_masks.reshape([-1, data['full_H'], data['full_W']])
            sec_project_masks = self.mask_projection(nerf_point_masks, data, depth, sec_depth)
            
            sec_project_masks = self.preprocess_masks(sec_project_masks)
            sec_project_masks = (sec_project_masks>0).to(torch.float) * 10 - 5

            sec_nerf_mask_masks, _, _ = self.sam_batch_predict(data['full_H'], data['full_W'], points=sec_input_point.detach().cpu().numpy(), 
                                                                masks_input=sec_project_masks, image=data['sec_images'])

        return sec_nerf_mask_masks, sec_input_point
    
    def train_step(self, data):
        bg_color = None

        use_cache = self.opt.cache_size > 0 and len(self.cache.data) > 0 and \
                    self.global_step % self.opt.cache_freq != 0
        if use_cache:
            data = self.cache.get()

        assert 'feature' in data or self.predictor is not None , \
            "Need features for training"
        
        if 'feature' in data:
            gt_feature = data['feature'] 
        else:
            gt_feature = self.get_sam_features_online(data)

        if not use_cache:
            self.cache.insert(data)
        
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        B, N, _ = rays_o.shape

        # rays_index = data['rays_index'][..., None]
        # rays_index = rays_index.expand(-1, -1, self.feature_dim)
        # gt_feature = torch.gather(gt_feature.view(B, -1, self.feature_dim), 1, rays_index) # [B, N]
        # gt_feature = gt_feature.to(self.device)

        self.model.train()

        outputs = self.model.render(rays_o, rays_d, render_feature=True, staged=False, bg_color=bg_color, perturb=True, 
                                    force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))

        pred_feature = outputs['sam_feature'].reshape(B, data['H'], data['W'], -1) # [B, N, feature_dim] -> [B, H, W, feature_dim]
        
        loss = self.criterion(pred_feature, gt_feature).mean(-1) # [B, N, feature_dim] --> [B, N]

        # patch-based rendering
        if self.opt.patch_size > 1:
            gt_feature = gt_feature.view(-1, self.opt.patch_size, self.opt.patch_size, self.feature_dim).permute(0, 3, 1, 2).contiguous()
            pred_feature = pred_feature.view(-1, self.opt.patch_size, self.opt.patch_size, self.feature_dim).permute(0, 3, 1, 2).contiguous()

            loss = loss + 1e-3 * self.criterion_lpips(pred_feature, gt_feature)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]
            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()


        if self.opt.consistency_loss_weight > 0:
            
            rays_o = data['sec_rays_o'] # [B, N, 3]
            rays_d = data['sec_rays_d'] # [B, N, 3]
            sec_outputs = self.model.render(rays_o, rays_d, render_feature=True, staged=False, bg_color=bg_color, perturb=True, 
                                    force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
    
            sec_pred_feature = sec_outputs['sam_feature'].reshape(B, data['H'], data['W'], -1) # [B, N, feature_dim] -> [B, H, W, feature_dim]
        


        input_point, input_label = self.generate_prompt(data['full_H'], data['full_W'])
        points_per_batch = self.opt.points_per_batch
        num_batches = int(np.ceil(input_point.size(0) / points_per_batch))
        if self.opt.mask_loss_weight > 0 or self.opt.consistency_loss_weight > 0:

            with torch.no_grad():
                gt_masks = self.predict_gt_masks(data['images'][0], data['H'], data['W'], num_batches, input_point, points_per_batch)
            
            if self.opt.mask_loss_weight > 0:
                loss_gt_masks = gt_masks.flatten(1)
                nerf_point_masks = self.predict_mask_from_feature(pred_feature, data['H'], data['W'], num_batches, input_point=input_point, 
                                                        input_label=input_label, points_per_batch=points_per_batch)
                nerf_point_masks = nerf_point_masks[:, -1].flatten(1)
                mask_loss = self.mask_criterion(nerf_point_masks, loss_gt_masks)
                mask_loss *= self.opt.mask_loss_weight
                # mask_loss, nerf_masks = self.mask_loss(data['images'][0], pred_feature, data['full_H'], data['full_W'])
                # mask_loss *= self.opt.mask_loss_weight
                loss = loss + mask_loss
                if self.opt.label_regularization_weight > 0:
                    loss = loss + self.label_regularization(data['depth'], nerf_point_masks, data['full_H'], data['full_W']) * self.opt.label_regularization_weight
            
                if self.opt.wandb:
                    wandb.log({
                        'mask_loss': mask_loss.item()
                    }, commit=False)
                    
                    
            if self.opt.consistency_loss_weight > 0:

                sec_nerf_mask_masks, sec_input_point = self.generate_project_mask(data, gt_masks, input_point, num_batches)
                
                sec_nerf_point_masks = self.predict_mask_from_feature(sec_pred_feature, data['H'], data['W'], num_batches, input_point=sec_input_point, 
                                                                    input_label=input_label[:sec_input_point.size(0)], points_per_batch=points_per_batch)

                sec_nerf_point_masks = sec_nerf_point_masks.flatten(2).to(torch.float)
                sec_nerf_mask_masks = sec_nerf_mask_masks.flatten(2).to(torch.float)
                consistency_loss = self.consistency_criterion(sec_nerf_point_masks, sec_nerf_mask_masks) * self.opt.consistency_loss_weight 
                
                loss = loss + consistency_loss
        # 3d mask constraints
        # if self.opt.mask3d_loss_weight > 0:
        #     mask3d_loss = self.mask3d_loss(data)
        #     loss = loss + mask3d_loss.mean() * self.opt.mask3d_loss_weight

        return pred_feature, gt_feature, loss
    
    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        assert 'feature' in data or self.predictor is not None , \
            "Need features for training"
        B, _, _ = rays_o.shape
        H, W = data['H'], data['W']

        # eval with fixed background color
        bg_color = None
        
        outputs = self.model.render(rays_o, rays_d, render_feature=True, staged=True, 
                                    bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_feature = outputs['sam_feature'].reshape(B, H, W, -1)
        
        if 'feature' in data:
            gt_feature = data['feature'] 
        else:
            gt_feature = self.get_sam_features_online(data)
                
        loss = self.criterion(pred_feature, gt_feature).mean()

        return pred_rgb, pred_depth, pred_feature, gt_feature, loss
    
    
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            if self.opt.consistency_loss_weight > 0:
                consistency_offset = 2
                for index in range(len(data['poses'])):
                    self.optimizer.zero_grad(set_to_none=True)
                    sec_index = (index + consistency_offset) % len(data['poses'])
                    train_data = {
                        'H': data['H'],
                        'W': data['W'],
                        'rays_o': data['rays'][index]['rays_o'].to(self.device),
                        'rays_d': data['rays'][index]['rays_d'].to(self.device),
                        'full_H': data['full_H'],
                        'full_W': data['full_W'],
                        'full_rays_o': data['full_rays'][index]['rays_o'].to(self.device),
                        'full_rays_d': data['full_rays'][index]['rays_d'].to(self.device),
                        'pose': data['poses'][index].to(self.device),
                        'index': data['index'],
                        # The following items are used for consistency loss.
                        'sec_rays_o': data['rays'][sec_index]['rays_o'].to(self.device),
                        'sec_rays_d': data['rays'][sec_index]['rays_d'].to(self.device),
                        'sec_full_rays_o': data['full_rays'][sec_index]['rays_o'].to(self.device),
                        'sec_full_rays_d': data['full_rays'][sec_index]['rays_d'].to(self.device),
                        'sec_pose': data['poses'][sec_index].to(self.device),
                        'intrinsics': data['intrinsics'].to(self.device)
                    }
                    
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, truths, loss = self.train_step(train_data)
                    self.scaler.scale(loss).backward()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scheduler_update_every_step:
                        self.lr_scheduler.step()

                    loss_val = loss.detach().item()
                    total_loss += loss_val
                    self.local_step += 1
                    self.global_step += 1
                pass
            else:
                self.local_step += 1
                self.global_step += 1

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.train_step(data)
                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()

                loss_val = loss.detach().item()
                total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)
            
                if self.opt.wandb:
                    wandb.log({
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'loss_val': loss_val,
                        'loss_avg': total_loss / self.local_step,
                        'epoch': self.epoch,
                        'local_step': self.local_step,
                        'global_step': self.global_step,
                    })
            del preds, truths, loss
            # print(torch.cuda.memory_summary())
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()
        self.log(f"  Average Loss: {average_loss}.  ")
        self.log(f"==> Finished Epoch {self.epoch}.")
    
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        B, H, W = rays_o.shape[0], data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, render_feature=True, staged=True, 
                                    bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_feature = outputs['sam_feature'].reshape(B, H, W, -1)


        rays_o = data['full_rays_o'] # [B, N, 3]
        rays_d = data['full_rays_d'] # [B, N, 3]
        B, H, W = rays_o.shape[0], data['full_H'], data['full_W']
        outputs = self.model.render(rays_o, rays_d, render_feature=False, staged=True, 
                                    bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)               

        pred_depth = outputs['depth'].reshape(-1, H, W)
        pred_t_depth = outputs['t_depth'].reshape(-1, H, W)
        return pred_rgb, pred_depth, pred_t_depth, pred_feature

    ### ------------------------------

    def test(self, loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_t_depth = []
            all_preds_feature = []
            all_poses = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, pred_t_depth, pred_feature = self.test_step(data)
                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                pred_t_depth = pred_t_depth[0].detach().cpu().numpy()
                
                pred_feature = pred_feature[0].detach().cpu().numpy()

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    all_preds_t_depth.append(pred_t_depth)
                    all_preds_feature.append(pred_feature)
                    all_poses.append(data['poses'][0].cpu().numpy().tolist())
                    
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    
                    np.save(os.path.join(save_path, f'{name}_{i:04d}_depth.npy'), pred_t_depth)
                    
                    # pred_feature = pred_feature.transpose(2,0,1)
                    # pred_feature = self.postprocess(pred_feature)
                    np.save(os.path.join(save_path, f'{name}_{i:04d}_feature.npy'), pred_feature)
                    
                pbar.update(loader.batch_size)
        
        if write_video:
            os.makedirs(os.path.join(save_path, 'frames'), exist_ok=True)
            for i, (r,d,f) in tqdm.tqdm(enumerate(zip(all_preds, all_preds_t_depth, all_preds_feature))):
                # f = f.transpose(2,0,1)
                # f = self.postprocess(f)
                np.save(os.path.join(save_path, 'frames', f'{name}_{i:04d}_feature.npy'), f)
                cv2.imwrite(os.path.join(save_path, 'frames', f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
                np.save(os.path.join(save_path, 'frames', f'{name}_{i:04d}_depth.npy'), d)
                    
            all_poses = {'poses': all_poses} 
            all_poses = json.dumps(all_poses, indent=4)
 
            # Writing to sample.json
            with open(os.path.join(save_path, 'poses.json'), "w") as file:
                file.write(all_poses)      
                        
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
       
        self.log(f"==> Finished Test.")

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth, pred_feature = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
            pred_feature = F.interpolate(pred_feature.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()
        pred_feature = pred_feature[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
            'feature': pred_feature,
        }

        return outputs

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_feature, gt_feature, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    preds_feature_list = [torch.zeros_like(preds_feature).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_feature_list, preds_feature)
                    preds_feature = torch.cat(preds_feature_list, dim=0)

                    gt_feature_list = [torch.zeros_like(gt_feature).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(gt_feature_list, gt_feature)
                    gt_feature = torch.cat(gt_feature_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds_feature, gt_feature)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    save_path_feature = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_feature.npz')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    pred_feature = preds_feature[0].detach().cpu().numpy()
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pred_feature = pred_feature.transpose(2,0,1)
                    res = np.array(pred_feature.shape)
                    pred_feature = pred_feature.reshape([res[0], -1])
                    np.savez(save_path_feature, res=res, embedding=pred_feature)
                    
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")
