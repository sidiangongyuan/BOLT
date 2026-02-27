# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature


def attn_fuse(x, record_len, affine_matrix, align_corners):
    """Scaled dot-product attention fusion after warping to ego frame.
    Per-pixel self-attention across agents (no learnable params)."""
    _, C, H, W = x.shape
    B = affine_matrix.shape[0]
    split_x = regroup(x, record_len)
    sqrt_c = C ** 0.5
    out = []
    for b in range(B):
        N = record_len[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        feature_in_ego = warp_affine_simple(split_x[b],
                                            t_matrix[0, :, :, :],
                                            (H, W), align_corners=align_corners)
        # feature_in_ego: (N, C, H, W)
        feat_flat = feature_in_ego.view(N, C, H * W).permute(2, 0, 1)  # (H*W, N, C)
        score = torch.bmm(feat_flat, feat_flat.transpose(1, 2)) / sqrt_c  # (H*W, N, N)
        attn = torch.softmax(score, dim=-1)  # (H*W, N, N)
        fused = torch.bmm(attn, feat_flat)  # (H*W, N, C)
        # take ego (index 0)
        out.append(fused[:, 0, :].permute(1, 0).view(C, H, W))
    return torch.stack(out)


def max_fuse(x, record_len, affine_matrix, align_corners):
    """Element-wise max fusion after warping to ego frame."""
    _, C, H, W = x.shape
    B = affine_matrix.shape[0]
    split_x = regroup(x, record_len)
    out = []
    for b in range(B):
        N = record_len[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        feature_in_ego = warp_affine_simple(split_x[b],
                                            t_matrix[0, :, :, :],
                                            (H, W), align_corners=align_corners)
        out.append(torch.max(feature_in_ego, dim=0)[0])
    return torch.stack(out)


def mean_fuse(x, record_len, affine_matrix, align_corners):
    """Element-wise mean fusion after warping to ego frame."""
    _, C, H, W = x.shape
    B = affine_matrix.shape[0]
    split_x = regroup(x, record_len)
    out = []
    for b in range(B):
        N = record_len[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        feature_in_ego = warp_affine_simple(split_x[b],
                                            t_matrix[0, :, :, :],
                                            (H, W), align_corners=align_corners)
        out.append(torch.mean(feature_in_ego, dim=0))
    return torch.stack(out)


def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3) 
    """
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                    scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        self.stage = model_cfg["stage"]
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        self.fuse_method = model_cfg.get('fuse_method', 'weighted')  # weighted | max | mean | v2xvit
        print('Align corners: ', self.align_corners)
        print('Fuse method: ', self.fuse_method)

        if self.fuse_method == 'v2xvit':
            from opencood.models.fuse_modules.fusion_in_one import V2XViTFusion
            v2xvit_cfg = model_cfg.get('v2xvit', {})
            self.v2xvit_levels = nn.ModuleList()
            split_attn_map = {64: 'split_attn64', 128: 'split_attn128', 256: 'split_attn'}
            for dim in self.model_cfg['num_filters']:
                level_cfg = {'transformer': {'encoder': {
                    'num_blocks': v2xvit_cfg.get('num_blocks', 1),
                    'depth': v2xvit_cfg.get('depth', 1),
                    'use_roi_mask': True,
                    'use_RTE': False, 'RTE_ratio': 0,
                    'cav_att_config': {
                        'dim': dim, 'use_hetero': True,
                        'use_RTE': False, 'RTE_ratio': 0,
                        'heads': max(dim // 32, 2), 'dim_head': 32, 'dropout': 0.1,
                    },
                    'pwindow_att_config': {
                        'dim': dim,
                        'heads': [max(dim // 16, 1), max(dim // 32, 1), max(dim // 64, 1)],
                        'dim_head': [16, 32, 64],
                        'dropout': 0.1,
                        'window_size': [4, 8, 16],
                        'relative_pos_embedding': True,
                        'fusion_method': split_attn_map.get(dim, 'split_attn'),
                    },
                    'feed_forward': {'mlp_dim': dim, 'dropout': 0.1},
                    'sttf': {'downsample_rate': 1, 'voxel_size': [0.4, 0.4], 'use_adaptor': False},
                }}}
                self.v2xvit_levels.append(V2XViTFusion(level_cfg))

        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}


        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            if self.fuse_method == 'max':
                fused_feature_list.append(max_fuse(feature_list[i], record_len, affine_matrix, self.align_corners))
            elif self.fuse_method == 'mean':
                fused_feature_list.append(mean_fuse(feature_list[i], record_len, affine_matrix, self.align_corners))
            elif self.fuse_method == 'attn':
                fused_feature_list.append(attn_fuse(feature_list[i], record_len, affine_matrix, self.align_corners))
            elif self.fuse_method == 'v2xvit':
                fused_feature_list.append(self.v2xvit_levels[i](feature_list[i], record_len, affine_matrix))
            else:
                fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        return fused_feature, occ_map_list 
    
    def forward(self, spatial_features, record_len=None, affine_matrix=None, agent_modality_list=None, cam_crop_info=None):
        """
        Unified forward method to switch between 'single' and 'collab' mode.
        If in 'single' mode, only spatial_features is required.
        If in 'collab' mode, additional parameters are needed.
        """
        if self.stage == "single":
            return self.forward_single(spatial_features)
        elif self.stage == "collab":
            if record_len is None or affine_matrix is None:
                raise ValueError("record_len and affine_matrix are required for forward_collab()")
            return self.forward_collab(spatial_features, record_len, affine_matrix, agent_modality_list, cam_crop_info)