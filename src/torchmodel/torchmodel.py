from functools import partial

from src.detr.util.notebook import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

import matplotlib

import torch
from torch import nn
from torch.nn.functional import softmax

from torch.backends import mps
from torch import cuda

import torchmetrics
from torchmetrics import Metric, Accuracy, MetricCollection

from src.plotting import plot_rand_preds
from src.callback import DefaultCallback
from src.detr.masked_tensor.masked_tensor import MaskedTensor
from src.detr.data_classes.output import ObjectDetectionOutput
from src.detr.data_classes.target import ObjectDetectionTarget
from src.callback import CompositeCallback, ObjDetReporter, Profiler, Saver


etqdm = partial(trange, unit="epoch", desc="Epoch loop")
btqdm = partial(tqdm, unit="batch", desc="Batch loop", leave=False)


class TorchModel:
    """Class-wrapper for torch models."""
    def __init__(self, model, optimizer, criterion, metrics=None, callback=None):
        if hasattr(criterion, "reduction"):
            criterion.reduction = "sum"

        if metrics is None:
            metrics = MetricCollection([])
        else:
            if isinstance(metrics, (tuple, list)):
                metrics = MetricCollection(metrics)
                for name, metric in metrics.items():
                    if hasattr(metric, "top_k"):
                        metric.top_k = metric.top_k or 1

        if cuda.is_available():
            self.device = torch.device("cuda:0")
            if cuda.device_count() > 1:
                model = nn.DataParallel(model)
        # elif mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.optimizer = optimizer
        self.criterion = criterion
        self.callback = callback if callback is not None else DefaultCallback()

        self.model = self.to_device(model)
        self.metrics = self.to_device(metrics)
        self.loss = self.to_device(torch.zeros(2))

    def train(self, trn_loader, val_loader, epochs=1):
        with self.callback as callback:

            for epoch in etqdm(epochs):
                trn_info = self._train(trn_loader)
                val_info = self._validate(val_loader)
                callback(self, trn_info, val_info, epoch)

        return trn_info, val_info

    def _train(self, loader):
        self.model.train()  # criterion.train()?

        for input, target in btqdm(loader):
            input, target = self.to_device((input, target))

            self.model.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self._update_loss(loss)
            self._update_metrics(output, target)

        loss = self._compute_and_reset_loss()
        metrics = self._compute_and_reset_metrics()

        return loss, metrics

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()

        for input, target in loader:
            input, target = self.to_device((input, target))

            output = self.model(input)
            loss = self.criterion(output, target)

            self._update_loss(loss)
            self._update_metrics(output, target)

        loss = self._compute_and_reset_loss()
        metrics = self._compute_and_reset_metrics()

        return self.from_device((loss, metrics))

    def _update_loss(self, loss):
        self.loss += torch.stack([loss, torch.tensor(1.0).to(loss.device)])

    def _reset_loss(self):
        self.loss = torch.zeros(2, device=self.device)

    def _compute_and_reset_loss(self):
        value = self.loss[0] / self.loss[1]
        self._reset_loss()
        return value

    def _update_metrics(self, output, target):
        self.metrics.update(output, target)

    def _reset_metrics(self):
        self.metrics.reset()

    def _compute_and_reset_metrics(self):
        values = self.metrics.compute()
        self._reset_metrics()
        return values

    def test(self, tst_loader):
        return self._validate(tst_loader)

    @torch.no_grad()
    def predict_proba(self, input):
        self.model.eval()

        output = self.model(input)
        probas = softmax(output, dim=1)

        return probas

    @torch.no_grad()
    def predict(self, input):
        self.model.eval()

        output = self.model(input)
        _, preds = torch.max(output, dim=1)

        return preds

    def state_dict(self):
        modules = [
            "model.module" if isinstance(self.model, nn.DataParallel) else "model",
            "optimizer",
        ]
        return {module: getattr(self, module).state_dict() for module in modules}

    def load_model(self, path_to_model):
        self.model.load_state_dict(path_to_model)
        return self

    def load_optimizer(self, path_to_optimizer):
        self.optimizer.load_state_dict(path_to_optimizer)
        return self

    def to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return type(obj)(self.to_device(x) for x in obj)

        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return obj.to(self.device)

        if isinstance(obj, (MaskedTensor, ObjectDetectionOutput, ObjectDetectionTarget)):
            return obj.to(self.device)

        raise TypeError(f"Got: {type(obj)}")

    def from_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return type(obj)(self.from_device(x) for x in obj)

        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return obj.detach().cpu()

        if isinstance(obj, (MaskedTensor, ObjectDetectionOutput, ObjectDetectionTarget)):
            return obj.detach().cpu()

        if isinstance(obj, (matplotlib.figure.Figure, torchmetrics.detection.mean_ap.BaseMetricResults)):
            return obj

        raise TypeError(f"Got: {type(obj)}")