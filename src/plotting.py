import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms


matplotlib.rcParams["font.family"] = ["monospace"]


tsfm_inv = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])


def matplotlib_imshow(tensorimg, one_channel=False):
    """Normalize and plot tensor-like-images."""
    if one_channel:
        tensorimg = tensorimg.mean(dim=0)

    tensorimg = tsfm_inv(tensorimg)
    npimg = tensorimg.numpy()

    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_rand_preds(torchmodel, loader, n_max=5):
    """Plot Top-5 predictions from model for random sample of size 'n_max'."""
    classes = loader.classes
    input, labels = iter(loader).next()
    input, labels = torchmodel.to_device((input[:n_max], labels[:n_max]))
    probs = torchmodel.from_device(torchmodel.predict_proba(input))

    k = min(len(classes), 5)              # for topk()
    max_len = len(max(classes, key=len))  # for pretty print

    fig = plt.figure(figsize=(18, 36 / len(input)))
    for i, (probs_i, preds_i) in enumerate(zip(*torch.topk(probs, k, dim=1))):
        ax = fig.add_subplot(1, len(input), i+1, xticks=[], yticks=[])
        matplotlib_imshow(input[i])

        title = []
        for prob_i, pred_i in zip(probs_i, preds_i):
            title.append(
                f"{classes[pred_i]:<{max_len}} {prob_i:.2f}"
            )
        title.append(f"\nlabel: {classes[labels[i]]}")
        title = "\n".join(title)

        ax.set_title(title, loc="left")

    return fig