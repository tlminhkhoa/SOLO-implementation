import sys
import os.path as osp

from torch.utils import data
add_p = [osp.dirname(osp.abspath(osp.dirname(__file__)))]
add_p.append(f'{osp.sep}'.join(add_p[-1].split(osp.sep)[:-1]))
add_p.append(f'{osp.sep}'.join(add_p[-1].split(osp.sep)[:-1]))
sys.path.extend(add_p)


import torch
import torch.nn.functional as F

import sys
sys.path.append("/content/drive/MyDrive/SOLO-implementation/utils")
sys.path.append("/content/drive/MyDrive/SOLO-implementation")
from target_gen import Solo_targetGEN
import dice_loss, focal_loss
from dice_loss import dice_loss
from focal_loss import FocalLoss
from utils import multi_apply


class SoloLoss():
    def __init__(self,
                 num_classes=81,
                 sigma=0.1,
                 mask_loss_weight=3,
                 strides=(4, 8, 16, 32, 64),
                 grid_num=[40,36,24,16,12],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))):

        self.cate_out_channels = num_classes - 1
        self.sigma = sigma
        self.strides = strides
        self.grid_num = grid_num
        self.scale_ragnes = scale_ranges
        self.mask_loss_weight = mask_loss_weight
        self.loss_cate = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
        self.target_generator = Solo_targetGEN(sigma, strides, grid_num, scale_ranges)

    def __call__(self,
                mask_preds,
                cate_preds,
                gt_bbox_list,
                gt_label_list,
                gt_mask_list):

        featmap_sizes = [featmap.size()[-2:] for featmap in mask_preds]
        mask_label_ls, cate_label_ls, mask_ind_label_ls = \
            multi_apply(self.target_generator.gen_target,
                        gt_bbox_list,
                        gt_label_list,
                        gt_mask_list,
                        featmap_sizes=featmap_sizes
            )
        print(mask_label_ls)

        # ins
        mask_labels = [torch.cat([mask_labels_level_img[ins_ind_labels_level_img, ...]
                                    for mask_labels_level_img, ins_ind_labels_level_img in
                                    zip(mask_labels_level, ins_ind_labels_level)], 0)
                        for mask_labels_level, ins_ind_labels_level in zip(zip(*mask_label_ls), zip(*mask_ind_label_ls))]

        mask_preds = [torch.cat([mask_preds_level_img[ins_ind_labels_level_img, ...]
                                for mask_preds_level_img, ins_ind_labels_level_img in
                                zip(mask_preds_level, ins_ind_labels_level)], 0)
                        for mask_preds_level, ins_ind_labels_level in zip(mask_preds, zip(*mask_ind_label_ls))]


        mask_ind_labels = [
            torch.cat([mask_ind_labels_level_img.flatten()
                        for mask_ind_labels_level_img in mask_ind_labels_level])
            for mask_ind_labels_level in zip(*mask_ind_label_ls)
        ]
        flatten_mask_ind_labels = torch.cat(mask_ind_labels)
        num_mask = flatten_mask_ind_labels.sum()

        # dice loss
        loss_mask = []
        for input, target in zip(mask_preds, mask_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_mask.append(dice_loss(input, target))
        loss_mask = torch.cat(loss_mask).mean()
        loss_mask = loss_mask * self.mask_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                        for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_ls)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_mask + 1)
        return dict(loss_mask=loss_mask, loss_cate=loss_cate, loss_total=loss_mask + loss_cate)