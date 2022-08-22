import os
import requests
from typing import List, Tuple

import torch
from torch import Tensor
from torchvision.datasets import CocoDetection

from PIL import Image

from src.detr.data_classes.target import ObjectDetectionTarget
from src.detr.masked_tensor.masked_tensor import MaskedTensor


class CocoDetectionRemote(CocoDetection):
    # TODO: make more safe.
    def _load_image(self, id: int) -> Image.Image:
        path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
        return Image.open(requests.get(path, stream=True).raw).convert("RGB")

    def __len__(self):
        return 100


def coco_collate_fn(list_of_samples: List[Tuple[Tensor, dict]]):
    images, annotations = zip(*list_of_samples)

    tensor = MaskedTensor(images, channels_last=False)
    target = ObjectDetectionTarget(
        labels=torch.tensor([obj["category_id"] for annotation in annotations for obj in annotation]),
        bboxes=torch.stack([obj["bbox"] for annotation in annotations for obj in annotation]),
        sizes=tuple(len(annotation) for annotation in annotations),
    )
    return tensor, target
