import os
import sys

sys.path.append("/content/drive/MyDrive/SOLO-implementation/utils")


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.conv import ConvModule
from utils import normal_init, bias_init_with_prob, multi_apply


INF = 1e8


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


class SoloHead(nn.Module):
    def __init__(self,
                 num_classes=81,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=7,
                 cate_down_pos=0,
                 grid_num=[40,36,24,16,12],
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), **kwargs):
        super(SoloHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.cate_down_pos = cate_down_pos

        self.grid_num = grid_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        self.cate_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.solo_cate = nn.ModuleList([
                nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1) for _ in self.grid_num
        ])
        self.solo_mask = nn.ModuleList([
            nn.Conv2d(self.feat_channels, num**2, 1, padding=0) for num in self.grid_num
        ])


    def init_weights(self):
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_mask = bias_init_with_prob(0.01)
        for m in self.solo_mask:
            normal_init(m, std=0.01, bias=bias_mask)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)


    def forward(self, feats):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)

        mask_pred, cate_pred = multi_apply(self.forward_single, new_feats,
                                          list(range(len(self.grid_num))),
                                          upsampled_size=upsampled_size)
        return mask_pred, cate_pred


    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5,
                              mode='bilinear', align_corners=True,
                              recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:],
                              mode='bilinear', align_corners=True))


    def forward_single(self, x, idx, upsampled_size=None):
        mask_feat = x
        cate_feat = x

        ##############
        # ins branch #
        ##############
        # concat coord to
        x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
        y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([mask_feat.shape[0], 1, -1, -1])
        x = x.expand([mask_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        mask_feat = torch.cat([mask_feat, coord_feat], 1)

        for i, mask_layer in enumerate(self.mask_convs):
            mask_feat = mask_layer(mask_feat)

        mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='bilinear')
        mask_pred = self.solo_mask[idx](mask_feat)


        ###############
        # cate branch #
        ###############
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.grid_num[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate[idx](cate_feat)
        if not self.training:
            mask_pred = F.interpolate(mask_pred.sigmoid(), size=upsampled_size, mode='bilinear')
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return mask_pred, cate_pred