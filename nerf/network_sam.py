import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .sam_renderer import NeRFSAMRenderer


class NeRFNetwork(NeRFSAMRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_sam=3,
                 hidden_dim_sam=256,
                 num_layers_sam_dir=3,
                 hidden_dim_sam_dir=256,
                 feature_dim=256,       # sam embedding dim (256)
                 bound=1,
                 view_dependent=True,
                 **kwargs,
                 ):
        super().__init__(bound, feature_dim=feature_dim, **kwargs)

        self.view_dependent = view_dependent
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # sam feature network
        self.num_layers_sam = num_layers_sam
        self.hidden_dim_sam = hidden_dim_sam
        self.encoder_sam, self.in_dim_sam = get_encoder(encoding, num_levels=16, level_dim=8, 
                                                        log2_hashmap_size=19,
                                                        desired_resolution=512)

        sam_net = []
        for l in range(num_layers_sam):
            if l == 0:
                in_dim = self.in_dim_sam
            else:
                in_dim = hidden_dim_sam
            
            if l == num_layers_sam - 1:
                out_dim = hidden_dim_sam if self.view_dependent else feature_dim
            else:
                out_dim = hidden_dim_sam

            sam_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.sam_net = nn.ModuleList(sam_net)
        # self.norm_sam = nn.LayerNorm(hidden_dim_sam if self.view_dependent else feature_dim)

        if self.view_dependent:
            self.num_layers_sam_dir = num_layers_sam_dir
            self.hidden_dim_sam_dir = hidden_dim_sam_dir
            self.encoder_sam_dir, self.in_dim_sam_dir = get_encoder(encoding_dir, num_levels=16, level_dim=8, 
                                                                    log2_hashmap_size=19,
                                                                    desired_resolution=512)
        
            sam_dir_net = []
            for l in range(num_layers_sam_dir):
                if l == 0:
                    in_dim = self.in_dim_sam_dir + self.hidden_dim_sam
                else:
                    in_dim = hidden_dim_sam_dir
                
                if l == num_layers_sam_dir - 1:
                    out_dim = feature_dim
                else:
                    out_dim = hidden_dim_sam_dir

                sam_dir_net.append(nn.Linear(in_dim, out_dim, bias=True))

            self.sam_dir_net = nn.ModuleList(sam_dir_net)
            # self.norm_sam_dir = nn.LayerNorm(feature_dim)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    # TODO
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        s = self.encoder(x, bound=self.bound)

        h = s
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d_color = self.encoder_dir(d)
        h = torch.cat([d_color, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        # sam
        s_sam = self.encoder_sam(x, bound=self.bound)
        for l in range(self.num_layers_sam):
            s_sam = self.sam_net[l](s_sam)
            if l != self.num_layers_sam - 1:
                s_sam = F.relu(s_sam, inplace=True)
        # s_sam = self.norm_sam(s_sam)
        sam_f = s_sam
                
        if self.view_dependent:
            # s_sam = F.relu(s_sam, inplace=True)
            d_sam = self.encoder_sam_dir(d)
            h_sam = torch.cat([d_sam, s_sam], dim=-1)

            for l in range(self.num_layers_sam_dir):
                h_sam = self.sam_dir_net[l](h_sam)
                if l != self.num_layers_sam_dir - 1:
                    h_sam = F.relu(h_sam, inplace=True)
            # h_sam = self.norm_sam_dir(h_sam)
            sam_f = h_sam
            
        return sigma, color, sam_f

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow samed inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    def sam(self, x, d, mask=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        if mask is not None:
            output = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return output
            x = x[mask]
            d = d[mask]

        s_sam = self.encoder_sam(x, bound=self.bound)
        for l in range(self.num_layers_sam):
            s_sam = self.sam_net[l](s_sam)
            if l != self.num_layers_sam - 1:
                s_sam = F.relu(s_sam, inplace=True)                
        sam_f = s_sam
                
        if self.view_dependent:
            s_sam = F.relu(s_sam, inplace=True)         
            d_sam = self.encoder_sam_dir(d)
            h_sam = torch.cat([d_sam, s_sam], dim=-1)
            for l in range(self.num_layers_sam_dir):
                h_sam = self.sam_dir_net[l](h_sam)
                if l != self.num_layers_sam_dir - 1:
                    h_sam = F.relu(h_sam, inplace=True)
            sam_f = h_sam
            
        if mask is not None:
            output[mask] = sam_f.to(output.dtype) # fp16 --> fp32
        else:
            output = sam_f
            
        return output

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.encoder_sam.parameters(), 'lr': lr},
            {'params': self.sam_net.parameters(), 'lr': lr},
        ]
        if self.view_dependent:
            params.append({'params': self.encoder_sam_dir.parameters(), 'lr': lr})
            params.append({'params': self.sam_dir_net.parameters(), 'lr': lr})
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
