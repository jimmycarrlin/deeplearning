import torch
import torchmetrics

from torch import nn, Tensor
from torchmetrics.detection import mean_ap
from typing import Tuple, Dict, Any

from src.detr.data_classes.output import ObjectDetectionOutput
from src.detr.data_classes.target import ObjectDetectionTarget

from torchvision.ops import box_area


class MeanAP(torchmetrics.detection.mean_ap.MeanAveragePrecision):

    def update(self, output: ObjectDetectionOutput, target: ObjectDetectionTarget):
        self.__update_output(output)
        self.__update_target(target)

    def __update_output(self, output: ObjectDetectionOutput):
        output = output.detach().cpu()
        detections, scores, labels = output.unnormalized_bboxes, output.scores, output.labels

        for d, l, s in zip(detections, labels, scores):
            self.detections.append(d)
            self.detection_labels.append(l)
            self.detection_scores.append(s)

    def __update_target(self, target: ObjectDetectionTarget):
        target = target.detach().cpu()
        groundtruths, labels = target.iter_bboxes(), target.iter_labels()

        for gt, l in zip(groundtruths, labels):
            self.groundtruths.append(gt)
            self.groundtruth_labels.append(l)

    def _MeanAveragePrecision__evaluate_image_gt_no_preds(
        self, gt: Tensor, gt_label_mask: Tensor, area_range: Tuple[int, int], nb_iou_thrs: int
    ) -> Dict[str, Any]:
        """Some GT but no predictions."""
        # GTs
        gt = gt[gt_label_mask]
        nb_gt = len(gt)
        areas = box_area(gt)
        ignore_area = (areas < area_range[0]) | (areas > area_range[1])
        gt_ignore, _ = torch.sort(ignore_area.to(torch.uint8))
        gt_ignore = gt_ignore.to(torch.bool).to(self.device)

        # Detections
        nb_det = 0
        det_ignore = torch.zeros((nb_iou_thrs, nb_det), dtype=torch.bool, device=self.device)

        return {
            "dtMatches": torch.zeros((nb_iou_thrs, nb_det), dtype=torch.bool, device=self.device),
            "gtMatches": torch.zeros((nb_iou_thrs, nb_gt), dtype=torch.bool, device=self.device),
            "dtScores": torch.zeros(nb_det, dtype=torch.float32, device=self.device),
            "gtIgnore": gt_ignore,
            "dtIgnore": det_ignore,
        }


class COCOPostProcess(nn.Module):

    @torch.no_grad()
    def forward(self, output: ObjectDetectionOutput):
        bboxes, scores, labels = output.unnormalized_bboxes(), output.scores(), output.labels()
        return [{"boxes": b, "scores": s, "labels": l} for b, s, l in zip(bboxes, scores, labels)]
