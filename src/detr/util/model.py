import copy

from torch.nn import ModuleList, Conv2d


def _get_clones(module, N):
    return ModuleList(copy.deepcopy(module) for _ in range(N))


def cnn_out_channels(cnn):
    C = 3

    for name, mod in cnn.named_modules():
        if isinstance(mod, Conv2d) and ("downsample" not in name):
            C = mod.out_channels if mod.in_channels == C else C

    return C