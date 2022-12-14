import os

import torch

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ray import tune


class Callback(metaclass=ABCMeta):
    """Base class for all callbacks. Implement Composite pattern."""
    @abstractmethod
    def __call__(self, torchmodel, trn_info, val_info, epoch):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class CompositeCallback(Callback, Iterable):
    """Composite callback. Container for primitive callbacks."""
    def __init__(self, children):
        self.children = children

    def __call__(self, torchmodel, trn_info, val_info, epoch):
        for child in self:
            child.__call__(torchmodel, trn_info, val_info, epoch)

    def __enter__(self):
        for child in self:
            child.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for child in self:
            child.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return iter(self.children)


class ClassificationReporter(SummaryWriter, Callback):
    # TODO: metric iteration.
    """Primitive callback. Report losses and metrics."""
    def __call__(self, torchmodel, trn_info, val_info, epoch):
        self.add_scalar("Loss/trn", trn_info[0], epoch)
        self.add_scalar("Loss/val", val_info[0], epoch)

        for i, metric in enumerate(torchmodel.metrics):
            metric_name = type(metric).__name__
            self.add_scalar(f"{metric_name}Top{metric.top_k}/trn", trn_info[1][i], epoch)
            self.add_scalar(f"{metric_name}Top{metric.top_k}/val", val_info[1][i], epoch)

        # self.add_figure("Random predictions", val_info[2], epoch)

    def report(self, torchmodel, tst_info):
        self.add_scalar(f"Loss/tst", tst_info[0])

        for i, metric in enumerate(torchmodel.metrics):
            metric_name = type(metric).__name__
            self.add_scalar(f"{metric_name}Top{metric.top_k}/tst", tst_info[1][i])

        # self.add_figure("Random predictions", tst_info[2])


class ObjDetReporter(SummaryWriter, Callback):

    def __call__(self, torchmodel, trn_info, val_info, epoch):
        self.add_scalar("Loss/trn", trn_info["loss"], epoch)
        self.add_scalar("Loss/val", val_info["loss"], epoch)

        for (name, trn_value), (_, val_value) in zip(trn_info["metrics"].items(), val_info["metrics"].items()):
            self.add_scalar(f"{name}/trn", trn_value, epoch)
            self.add_scalar(f"{name}/val", val_value, epoch)

        self.flush()

        # self.add_figure("Random predictions", val_info[2], epoch)

    def report(self, torchmodel, tst_info):
        self.add_scalar(f"Loss/tst", tst_info[0])

        for name, value in tst_info[1]:
            self.add_scalar(f"{name}/tst", value)

        # self.add_figure("Random predictions", tst_info[2])


class Profiler(torch.profiler.profile, Callback):
    """Primitive callback. Profile CPU, GPU and memory usage."""
    def __call__(self, *_):
        self.step()

    @classmethod
    def default(cls, log_dir):
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=3)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(log_dir)
        profile_memory = True

        return cls(schedule=schedule, on_trace_ready=on_trace_ready, profile_memory=profile_memory)


class Saver(Callback):
    # TODO: lr_scheduler.state_dict().
    """Primitive callback. Save last and best models."""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_val_loss = float("inf")

    def __call__(self, torchmodel, _, val_info, __):
        val_loss = val_info["loss"]
        state_dict = torchmodel.state_dict()
        os.makedirs(self.save_dir, exist_ok=True)

        torch.save(state_dict, self.save_dir / "torchmodel_last.pt")
        if val_loss < self.best_val_loss:
            torch.save(state_dict, self.save_dir / "torchmodel_best.pt")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Tuner(Callback):
    """Primitive callback. Used only in parallel with ray tune."""
    def __call__(self, torchmodel, trn_info, val_info, epoch):
        with tune.checkpoint_dir(epoch) as checkpoint_dir:

            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(torchmodel.model.state_dict(), torchmodel.optimizer.state_dict(), path)

        tune.report(loss=val_info["loss"], accuracy=val_info["metrics"][0])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
