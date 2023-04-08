import os
import sys

add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F

import modules
from src.eval import get_masks


class Solo_v1(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(Solo_v1, self).__init__()

        self.backbone = getattr(modules, cfg['backbone'])(**cfg.get('backbone_args', {}))
        self.neck = modules.FPN(**cfg.get('fpn_args', {}))
        self.head = getattr(modules, cfg['head'])(**cfg.get('head_args', {}))
        self.post_process = getattr(modules, cfg['post_process'])(cfg['post_process_args'],
                                     grid_num=cfg['head_args']['grid_num'],
                                     strides=cfg['head_args']['strides'],
                                     num_classes=cfg['head_args']['num_classes'])
        self.solo_loss = getattr(modules, cfg['loss'])(**cfg.get('loss_args', {}))


    def forward(self, inp, targets=None, img_metas=None):
        if isinstance(inp, tuple):
            inp = torch.stack(inp)

        backbone_out = self.backbone(inp)
        neck_out = self.neck(backbone_out)
        mask_preds, cate_preds = self.head(neck_out)
        if self.training:
            gt_bboxes_list = [target['boxes'] for target in targets]
            gt_labels_list = [target['labels'] for target in targets]
            gt_masks_list = [target['masks'] for target in targets]
            losses = self.solo_loss(mask_preds, cate_preds, gt_bboxes_list, gt_labels_list, gt_masks_list)
            return losses
        else:
            return self.post_process(mask_preds, cate_preds, img_metas=img_metas)