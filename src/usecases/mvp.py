from pathlib import Path
from typing import List
from typing import Union

from fastai.vision import Image
from fastai.vision import ImageDataBunch
from fastai.vision import ImageList
from fastai.vision import cnn_learner
from fastai.vision import models
from fastai.vision import imagenet_stats
from fastai.torch_core import flatten_model
import matplotlib.pyplot as plt
import torch
from torch import nn


# TODO refactor all types into a separate file
# define some useful types (inspired by fastai)
PathOrStr = Union[Path, str]


def data_for_activations(data_path: PathOrStr, bs: int = 16, img_size: int = 160, pct_partial: float = 1.0,
                         num_workers: int = 0, seed: int = 42) -> ImageDataBunch:
    """
    Create data object from Imagenet-style directory structure.

    This is a wrapper around fastai's Data Block API. The purpose is to automate and package together datasets and
    dataloaders, transforms, splitting the data, etc.

    :param data_path: path to data in Imagenet-style folder structure.
    :param bs: batch size
    :param img_size: target image size
    :param pct_partial: proportion of all data to use
    :param num_workers: number of workers used to parallelize data transformations when feeding into the model
    :param seed:
    :return: data object containing data set and data loader (in PyTorch sense)

    .. note:: more on Data Block API here: https://docs.fast.ai/data_block.html
    .. note:: Imagenet-style directory structure: https://docs.fast.ai/vision.data.html#ImageDataBunch.from_folder
    .. note:: `num_workers` anything from 0 crashes on my laptop, ideally, should equal the number of cores of your CPU
    .. note:: all of the data will be used as training set, even images in `valid` folder
    """
    data = (ImageList.from_folder(data_path)            # -> ImageList
            .use_partial_data(pct_partial, seed=seed)   # -> ImageList
            .split_none()                               # -> ItemLists: train and valid ItemList
            .label_from_folder()                        # -> LabelLists: train and valid LabelList
            .transform(size=img_size)                   # -> LabelLists: train and valid LabelList
            .databunch(bs=bs, num_workers=num_workers)  # -> ImageDataBunch
            .normalize(imagenet_stats))                 # -> ImageDataBunch

    # we want the order of images to not be shuffled to be able to find the right images easily
    data.train_dl = data.train_dl.new(shuffle=False)
    return data


class VisualSearchEngine:
    def __init__(self, data: ImageDataBunch):
        # Create data and learner
        self.data = data
        self.learner = cnn_learner(self.data, models.resnet18)
        self.last_layer: nn.Module = flatten_model(self.learner.model)[-2]

        # Precompute data activations as part of initialization
        # TODO refactor computation into a separate method?
        self.activations_list = []
        self.last_layer.register_forward_hook(self.hook)
        _ = self.learner.get_preds(data.train_ds)
        self.data_activations = torch.cat(self.activations_list)

        # This will store activations for query image
        self.query_act = None

    def hook(self, module, input, output) -> None:
        """
        Hook for collecting layer activations.
        This hook is invoked for each batch forward pass.

        :param module: layer to which the hook is attached
        :param input: batch input into the layer
        :param output: batch output of the layer
        :return: None
        """
        self.activations_list.append(output)

    def find_closest_images(self, img: Image, num=16) -> List[Image]:
        """
        Return `num` closest images to `img` in activations space.

        :param img: query image
        :param num: number of closest images to return
        :return: list of `num` closest images
        """
        # Compute activation for query image
        self.activations_list = []
        _ = self.learner.predict(img)
        self.query_act = self.activations_list[0]
        # Find distances, sort them
        # TODO calculate distances in a separate method?
        distances = (self.data_activations - self.query_act).pow(2).sum(dim=1).numpy()
        closest_idxs = distances.argsort()[:num]
        result_itemlist = self.data.train_ds[closest_idxs]
        # Only return images, ignore the labels
        return [img for img, label in result_itemlist]


def plot_results(imgs: List[Image]) -> None:
    """
    Plot closest images.

    .. note:: code inspired by: https://docs.fast.ai/vision.image.html#The-fastai-Image-classes

    :param imgs: list of closest images
    :return: None
    """
    # TODO handle any number of images, not only 16
    _, axs = plt.subplots(4, 4, figsize=(8, 8))
    for img, ax in zip(imgs, axs.flatten()):
        img.show(ax=ax)


if __name__ == '__main__':
    pass
