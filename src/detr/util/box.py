import torch
from torch import Tensor, as_tensor, zeros
from typing import Optional, Union, Tuple


def box_area(boxes: Tensor):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Out: (N, M, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    # Out: (N, M)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / union, union


def box_giou(boxes1: Tensor, boxes2: Tensor):
    iou, union = box_iou(boxes1, boxes2)

    # Out: (N, M, 2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_giou_loss(out_boxes: Tensor, tgt_boxes: Tensor):
    return (1 - torch.diag(box_giou(box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes)))).sum()


def box_cxcywh_to_xyxy(boxes: Tensor):
    x_c, y_c, w, h = boxes.unbind(-1)

    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor):
    x0, y0, x1, y1 = boxes.unbind(-1)

    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_rel_to_abs(boxes: Tensor, shapes: Tensor):
    # TODO: Different shapes.
    _, img_h, img_w = shapes.unbind(-1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    return boxes


def crop(boxes: Tensor, top: int, left: int, height: int, width: int):
    shift = as_tensor([left, top, left, top])
    maxes = as_tensor([width, height, width, height])
    return boxes.sub_(shift).clamp_(min=zeros(4), max=maxes)


def hflip(boxes: Tensor, width: int):
    negat = as_tensor([-1, 1, -1, 1])
    shift = as_tensor([width, 0, width, 0])
    return boxes.mul_(negat).add_(shift)[..., (2, 1, 0, 3)]


def resize(boxes: Tensor, height_ratio: float, width_ratio: float):
    ratios = as_tensor([width_ratio, height_ratio, width_ratio, height_ratio])
    return boxes.mul_(ratios)
