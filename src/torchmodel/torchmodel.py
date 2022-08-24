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
    """The class providing train/inference interface for Torch models."""
    def __init__(self, model, optimizer, criterion, metrics=None, callback=None, profiler=None):
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
        else:
            self.device = torch.device("cpu")

        self.optimizer = optimizer
        self.criterion = criterion
        self.callback = callback if callback is not None else DefaultCallback()
        self.profiler = profiler

        self.model = self.to_device(model)
        self.metrics = self.to_device(metrics)

    def train(self, trn_loader, val_loader, epochs=1):
        with self.callback as C:

            for epoch in etqdm(epochs):
                trn_info = self._train(trn_loader)
                val_info = self._validate(val_loader)
                C(self, trn_info, val_info, epoch)

        return trn_info, val_info

    def _train(self, loader):
        with self.profiler as P:

            self.model.train()  # criterion.train()?

            for input, target in btqdm(loader):
                input, target = self.to_device((input, target))

                self.model.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                self._update_loss(loss.detach())
                self._update_metrics(output.detach(), target.detach())
                P.step()

            loss = self._compute_and_reset_loss()
            metrics = self._compute_and_reset_metrics()

        return {"loss": loss, "metrics": metrics}

    @torch.no_grad()
    def _validate(self, loader):
        with self.profiler as P:

            self.model.eval()

            for input, target in loader:
                input, target = self.to_device((input, target))

                output = self.model(input)
                loss = self.criterion(output, target)

                self._update_loss(loss.detach())
                self._update_metrics(output.detach(), target.detach())
                P.step()

            loss = self._compute_and_reset_loss()
            metrics = self._compute_and_reset_metrics()

        return {"loss": loss, "metrics": metrics}

    def _update_loss(self, loss):
        if not hasattr(self, "loss"):
            self.loss = torch.empty(0).to(self.device)
        torch.cat((self.loss, loss.unsqueeze(0)))

    def _compute_and_reset_loss(self):
        value = self.loss.mean().cpu()
        self.loss = torch.empty(0)
        return value

    def _update_metrics(self, output, target):
        self.metrics.update(output, target)

    def _compute_and_reset_metrics(self):
        values = self.metrics.compute()
        self.metrics.reset()
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

    def load_state_dict(self, path):
        for module_name, module_state_dict in torch.load(path):
            getattr(self, module_name).load_state_dict(module_state_dict)
        return self

    def to_device(self, obj):
        # TODO: *args.
        if isinstance(obj, (tuple, list)):
            return type(obj)(self.to_device(x) for x in obj)

        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return obj.to(self.device)

        if isinstance(obj, (MaskedTensor, ObjectDetectionOutput, ObjectDetectionTarget)):
            return obj.to(self.device)

        raise TypeError(f"Got: {type(obj)}")

    def from_device(self, obj):
        # TODO: *args.
        if isinstance(obj, (tuple, list)):
            return type(obj)(self.from_device(x) for x in obj)

        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return obj.detach().cpu()

        if isinstance(obj, (MaskedTensor, ObjectDetectionOutput, ObjectDetectionTarget)):
            return obj.detach().cpu()

        if isinstance(obj, (matplotlib.figure.Figure, torchmetrics.detection.mean_ap.BaseMetricResults)):
            return obj

        raise TypeError(f"Got: {type(obj)}")