# https://github.com/WXinlong/SOLO/blob/master/mmdet/models/losses/focal_loss.py

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sigmoid_focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss
from torchvision.ops import sigmoid_focal_loss as _sigmoid_focal_loss
def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

# def sigmoid_focal_loss(pred,
#                        target,
#                        weight=None,
#                        gamma=2.0,
#                        alpha=0.25,
#                        reduction='mean',
#                        avg_factor=None):
#     # Function.apply does not accept keyword arguments, so the decorator
#     # "weighted_loss" is not applicable
#     loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
#     # TODO: find a proper way to handle the shape of weight
#     if weight is not None:
#         weight = weight.view(-1, 1)
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss

# This method is only for debugging
def sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    # with open("./target.txt", 'w') as f:
    #   for s in target:
    #       f.write(str(s) + '\n')
    # target_before = target
    # print(target_before.shape)
    # print(type(target_before))
    try:
      target_onehot = F.one_hot(target.long(),  pred_sigmoid.size(-1)).type_as(pred)
    except Exception as error:
      import pandas as pd
      df = pd.DataFrame(target.numpy())
      df.to_csv("./target.csv")
      print(error)


    # target_after = target_onehot
    # with open("./target_after.txt", 'w') as f:
    #   for s in target:
    #       f.write(str(s) + '\n')
    # print(target_after.shape)
    # print(type(target_after))

    # print(torch.eq(target_before, target_after))
    # target_onehot = target.view(1, -1).type_as(pred)


    pt = (1 - pred_sigmoid) * target_onehot + pred_sigmoid * (1 - target_onehot)
    focal_weight = (alpha * target_onehot + (1 - alpha) *
                    (1 - target_onehot)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target_onehot, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

# # This method is only for debugging
# def py_sigmoid_focal_loss(pred,
#                           target,
#                           weight=None,
#                           gamma=2.0,
#                           alpha=0.25,
#                           reduction='mean',
#                           avg_factor=None):
#     pred_sigmoid = pred.sigmoid()
#     target = target.type_as(pred)
#     pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
#     focal_weight = (alpha * target + (1 - alpha) *
#                     (1 - target)) * pt.pow(gamma)
#     loss = F.binary_cross_entropy_with_logits(
#         pred, target, reduction='none') * focal_weight
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss


# def sigmoid_focal_loss(pred,
#                        target,
#                        weight=None,
#                        gamma=2.0,
#                        alpha=0.25,
#                        reduction='mean',
#                        avg_factor=None):
#     # Function.apply does not accept keyword arguments, so the decorator
#     # "weighted_loss" is not applicable
#     loss = py_sigmoid_focal_loss(pred, target, gamma, alpha)
#     # TODO: find a proper way to handle the shape of weight
#     if weight is not None:
#         weight = weight.view(-1, 1)
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss


# def sigmoid_focal_loss(pred,
#                        target,
#                        weight=None,
#                        gamma=2.0,
#                        alpha=0.25,
#                        reduction='mean',
#                        avg_factor=None):
#     # Function.apply does not accept keyword arguments, so the decorator
#     # "weighted_loss" is not applicable
#     loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
#     # TODO: find a proper way to handle the shape of weight
#     if weight is not None:
#         weight = weight.view(-1, 1)
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss


class FocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls