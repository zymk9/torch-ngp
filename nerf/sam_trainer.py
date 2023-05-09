
import os
import tqdm
import imageio

import numpy as np

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributed as dist
from .utils import Trainer, preprocess_feature, postprocess_feature, print_shape, get_rays, linear_to_srgb
import json

class SAMTrainer(Trainer):
    def __init__(self, *args, feature_size=64, load_feature = True, predictor = None, **kwargs):
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
        if predictor is not None:
            self.predictor.model.to(self.device)
        self.load_feature = load_feature
        # freeze rgb and density
        self.model.encoder.requires_grad_(False)
        self.model.sigma_net.requires_grad_(False)
        self.model.encoder_dir.requires_grad_(False)
        self.model.color_net.requires_grad_(False)

    ### ------------------------------
    def postprocess(self, features):
        target_feature_size = self.feature_size
        f_h, f_w = features.shape[1:]
        max_length = max(f_h, f_w)
        h,w = int(np.floor(target_feature_size * f_h / max_length)), int(np.floor(target_feature_size * f_w / max_length))
        features = cv2.resize(features.transpose(1,2,0), (w,h), interpolation = cv2.INTER_LINEAR  )
        features = features.transpose(2,0,1)
        features = torch.from_numpy(features[None,...])
        
        padh = target_feature_size - h
        padw = target_feature_size - w
        return F.pad(features, (0, padw, 0, padh)).numpy()[0]



    def label_regularization(self, depth, pred_masks):
        '''
        depth: [B, N]
        pred_masks: [B, N, num_instances]
        '''
        pred_masks = pred_masks.view(-1, self.opt.patch_size, self.opt.patch_size, 
            self.num_instances).permute(0, 3, 1, 2).contiguous() # [B, num_instances, patch_size, patch_size]

        diff_x = pred_masks[:, :, :, 1:] - pred_masks[:, :, :, :-1]
        diff_y = pred_masks[:, :, 1:, :] - pred_masks[:, :, :-1, :]

        depth = depth.view(-1, self.opt.patch_size, self.opt.patch_size) # [B, patch_size, patch_size]

        depth_diff_x = depth[:, :, 1:] - depth[:, :, :-1]
        depth_diff_y = depth[:, 1:, :] - depth[:, :-1, :]
        weight_x = torch.exp(-(depth_diff_x * depth_diff_x)).unsqueeze(1).expand_as(diff_x)
        weight_y = torch.exp(-(depth_diff_y * depth_diff_y)).unsqueeze(1).expand_as(diff_y)

        diff_x = diff_x * diff_x * weight_x
        diff_y = diff_y * diff_y * weight_y

        smoothness_loss = torch.sum(diff_x) / torch.sum(weight_x) + torch.sum(diff_y) / torch.sum(weight_y)

        return smoothness_loss

    def generate_prompt(self, H, W):
        if self.opt.prompt_sample == 'uniform':
            y = torch.linspace(0, H, steps=self.opt.sample_step)
            x = torch.linspace(0, W, steps=self.opt.sample_step)
            
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            input_point = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], -1)
            input_label = torch.ones(input_point.shape[0])
            return input_point, input_label
        elif self.opt.prompt_sample == 'random':
            y = torch.randint(0, H, (self.opt.sample_step**2,))
            x = torch.randint(0, W, (self.opt.sample_step**2,))
            
            pass
            
    
    def mask_loss(self, pred_features, H, W):
        input_point, input_label = self.generate_prompt(H, W)
        with torch.no_grad():
            gt_masks = []
            for i in range(input_point.size(0)):
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point[i:i+1, :].numpy(),
                    point_labels=input_label[i:i+1].numpy(),
                    multimask_output=True,
                )

                gt_masks.append(masks[0].reshape(-1))
            gt_masks = np.stack(gt_masks)
            gt_masks = torch.from_numpy(gt_masks).to(torch.float32).to(self.device)

        target_feature_size = self.predictor.model.image_encoder.img_size // self.predictor.model.image_encoder.patch_size
        
        pred_features = postprocess_feature(pred_features, target_feature_size)

        self.predictor.set_torch_feature(pred_features)
        input_point = input_point.to(self.device)
        input_label = input_label.to(self.device)
        nerf_masks = []
        for i in range(input_point.size(0)):
            pred_masks, pred_scores, pred_logits = self.predictor.predict_torch(
                    point_coords=input_point[i:i+1, :][None, ...],
                    point_labels=input_label[i:i+1][None, ...],
                    return_logits=True
                )

            nerf_masks.append(pred_masks[0][0].reshape(-1))     

        nerf_masks = torch.stack(nerf_masks)
        loss = self.mask_criterion(nerf_masks, gt_masks)
        return loss
    
    
    def train_step(self, data):
        bg_color = None
        if 'feature' in data:
            gt_feature = data['feature'] 
        else:
            with torch.no_grad():
                if 'image' in data:
                    input_im = data['image']
                else:
                    rays_o = data['full_rays_o'] # [B, N, 3]
                    rays_d = data['full_rays_d'] # [B, N, 3]
                    self.model.eval()
                    input = self.model.render(rays_o, rays_d, render_feature=False, staged=True, bg_color=bg_color, 
                                                perturb=False, force_all_rays=True, use_cuda=False, **vars(self.opt))
                    input_im = input['image'].reshape(-1, data['full_H'], data['full_W'], 3)
                    input_im = (input_im[0].detach().cpu().numpy() * 255).astype(np.uint8)

                self.predictor.set_image(input_im)
                gt_feature = self.predictor.get_image_embedding().detach()
                
                if gt_feature.shape[0] != data['H'] or gt_feature.shape[1] != data['W']:
                    gt_feature = preprocess_feature(gt_feature, data['H'], data['W'])
    
                gt_feature = gt_feature.permute(0,2,3,1)

        
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        assert 'feature' in data or self.predictor is not None , \
            "Need features for training"

        B, N, _ = rays_o.shape
        rays_index = data['rays_index'][..., None]
        rays_index = rays_index.expand(-1, -1, self.feature_dim)
        gt_feature = torch.gather(gt_feature.view(B, -1, self.feature_dim), 1, rays_index) # [B, N]
        gt_feature = gt_feature.to(self.device)
        self.model.train()
        outputs = self.model.render(rays_o, rays_d, render_feature=True, staged=False, bg_color=bg_color, perturb=True, 
                                    force_all_rays=True, **vars(self.opt))
        pred_feature = outputs['sam_feature']
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

        if self.opt.label_regularization_weight > 0:
            loss = loss + self.label_regularization(outputs['depth'], pred_feature) * self.opt.label_regularization_weight
        
        if self.opt.mask_loss_weight > 0:
            pred_feature = pred_feature.reshape(B, data['H'], data['W'], -1)
            loss = loss + self.mask_loss(pred_feature, data['full_H'], data['full_W']) * self.opt.mask_loss_weight
        
        # 3d mask constraints
        if self.opt.mask3d_loss_weight > 0:
            mask3d_loss = self.mask3d_loss(data)
            loss = loss + mask3d_loss.mean() * self.opt.mask3d_loss_weight

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
            with torch.no_grad():
                if 'image' in data:
                    input_im = data['image']
                else:
                    input = self.model.render(data['full_rays_o'], data['full_rays_d'], render_feature=False, staged=True, bg_color=bg_color, 
                                                perturb=True, **vars(self.opt))
                    input_im = input['image'].reshape(-1, data['full_H'], data['full_W'], 3)
                    input_im = (input_im[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    
                self.predictor.set_image(input_im)
                gt_feature = self.predictor.get_image_embedding()
                
                if gt_feature.shape[0] != data['H'] or gt_feature.shape[1] != data['W']:
                    gt_feature = preprocess_feature(gt_feature, data['H'], data['W'])
                gt_feature = gt_feature.permute(0,2,3,1)
                
                
                
        loss = self.criterion(pred_feature, gt_feature).mean()

        if self.opt.label_regularization_weight > 0:
            loss = loss + self.label_regularization(outputs['depth'], pred_feature) * self.opt.label_regularization_weight

        if self.opt.mask3d_loss_weight > 0:
            mask3d_loss = self.mask3d_loss(data)
            loss = loss + mask3d_loss.mean() * self.opt.mask3d_loss_weight


        return pred_rgb, pred_depth, pred_feature, gt_feature, loss
    
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
                    
                    depth_shape = pred_t_depth.shape
                    pred_t_depth = pred_t_depth.reshape(-1).astype(np.float32)
                    np.savez(os.path.join(save_path, f'{name}_{i:04d}_depth.npz'), size=depth_shape, depth=pred_t_depth)
                    
                    pred_feature = pred_feature.transpose(2,0,1)
                    pred_feature = self.postprocess(pred_feature)
                    res = np.array(pred_feature.shape)
                    pred_feature = pred_feature.reshape(-1).astype(np.float32)
                    np.savez(os.path.join(save_path, f'{name}_{i:04d}_feature.npz'), res=res, embedding=pred_feature)

                    
                pbar.update(loader.batch_size)
        
        if write_video:
            os.makedirs(os.path.join(save_path, 'frames'), exist_ok=True)
            for i, (r,d,f) in tqdm.tqdm(enumerate(zip(all_preds, all_preds_t_depth, all_preds_feature))):
                f = f.transpose(2,0,1)
                f = self.postprocess(f)
                res = np.array(f.shape)
                f = f.reshape(-1).astype(np.float32)
                np.savez(os.path.join(save_path, 'frames', f'{name}_{i:04d}_feature.npz'), res=res, embedding=f)
                cv2.imwrite(os.path.join(save_path, 'frames', f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
                depth_shape = d.shape
                d = d.reshape(-1).astype(np.float32)
                np.savez(os.path.join(save_path, 'frames', f'{name}_{i:04d}_depth.npz'), size=depth_shape, depth=d)
                    
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
      