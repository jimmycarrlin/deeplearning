import os
from pathlib import Path
from datetime import datetime


from torch.optim import AdamW
from torch.utils.data import DataLoader

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.torchmodel.torchmodel import TorchModel
from src.detr.modules.detr import DETR
from src.detr.modules.criterion.criterion import SetCriterion
from src.detr.modules.metrics.map import MeanAP
from src.callback import CompositeCallback, ObjDetReporter, Profiler, Saver
from src.detr.util.transformations import get_coco_transforms
from src.detr.util.data import coco_collate_fn, CocoDetectionRemote
from src.detr.util.transformations import Compose, PolyToMask, ToTensor, MaxResize, Normalize


def main():
    path2imgs = "http://images.cocodataset.org/"
    path2anns = "/Users/pavelkiselev/PycharmProjects/cocoapi/data/annotations/"

    trn_tsfms = Compose([PolyToMask(), ToTensor(), MaxResize(1333), Normalize()])
    val_tsfms = Compose([PolyToMask(), ToTensor(), MaxResize(1333), Normalize()])

    trn_set = CocoDetectionRemote(path2imgs + "train2017", path2anns + "instances_train2017.json", transforms=trn_tsfms)
    val_set = CocoDetectionRemote(path2imgs + "val2017", path2anns + "instances_val2017.json", transforms=val_tsfms)

    trn_loader = DataLoader(trn_set, batch_size=1, shuffle=False, collate_fn=coco_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=coco_collate_fn)

    model = DETR.default(num_classes=91)
    criterion = SetCriterion(num_classes=91)
    optimizer = AdamW(params=model.parameters_to_optimize())
    metrics = MeanAP()

    model_repr = model.extra_repr()
    time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = Path(os.getcwd()) / "runs" / model_repr / time
    print(log_dir)
    callback = CompositeCallback([ObjDetReporter(log_dir), Profiler.default(log_dir), Saver(log_dir)])

    torchmodel = TorchModel(model, optimizer, criterion, metrics, callback)
    torchmodel.train(trn_loader, val_loader, epochs=100)


if __name__ == "__main__":
    main()
