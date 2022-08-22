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

    def add(self, child):
        self.children.append(child)

    def remove(self, child):
        self.children.remove(child)

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
        self.add_scalar("Loss/trn", trn_info[0], epoch)
        self.add_scalar("Loss/val", val_info[0], epoch)

        for (name, trn_value), (_, val_value) in zip(trn_info[1].items(), val_info[1].items()):
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
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=2)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(log_dir)
        profile_memory = True

        return cls(schedule=schedule, on_trace_ready=on_trace_ready, profile_memory=profile_memory)


class Saver(Callback):
    # TODO: making a directory to save.
    # TODO: optimizer.state_dict(), lr_scheduler.state_dict().
    """Primitive callback. Save last and best models."""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_val_loss = float("inf")

    def __call__(self, torchmodel, _, val_info, __):
        val_loss, *_ = val_info
        self._backup(torchmodel)
        self._best(torchmodel, val_loss)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _backup(self, torchmodel):
        for module_name, module_state_dict in torchmodel.state_dict().items():
            torch.save(module_state_dict, self.save_dir / f"backup_{module_name}.pt")

    def _best(self, torchmodel, val_loss):
        if val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss
        for module_name, module_state_dict in torchmodel.state_dict().items():
            torch.save(module_state_dict, self.save_dir / f"best_{module_name}.pt")


class Tuner(Callback):
    """Primitive callback. Used only in parallel with ray tune."""
    def __call__(self, torchmodel, trn_info, val_info, epoch):
        with tune.checkpoint_dir(epoch) as checkpoint_dir:

            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(torchmodel.model.state_dict(), torchmodel.optimizer.state_dict(), path)

        tune.report(loss=val_info[0], accuracy=val_info[1][0])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DefaultCallback(Callback):
    """Primitive callback. Do nothing."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, torchmodel, train_info, val_info, epoch):
        pass
