import os
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

from fastai.vision import Image
from fastai.vision import ImageDataBunch
from fastai.vision import ImageList
from fastai.vision import cnn_learner
from fastai.vision import models
from fastai.vision import imagenet_stats
from fastai.torch_core import flatten_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import rasterfairy
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch import Tensor


# TODO refactor all types and errors into a separate file
# define some useful types (inspired by fastai)
PathOrStr = Union[Path, str]


class NonExistentPathError(Exception):
    pass


class EmptyFolderError(Exception):
    pass


class PathIsNotFolderError(Exception):
    pass


class InvalidImageError(Exception):
    pass


def get_data(data_path: PathOrStr, bs: int = 16, img_size: int = 160, pct_partial: float = 1.0,
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

    data: ImageDataBunch = (ImageList.from_folder(data_path)            # -> ImageList
                            .use_partial_data(pct_partial, seed=seed)   # -> ImageList
                            .split_none()                               # -> ItemLists: train and valid ItemList
                            .label_from_folder()                        # -> LabelLists: train and valid LabelList
                            .transform(size=img_size)                   # -> LabelLists: train and valid LabelList
                            .databunch(bs=bs, num_workers=num_workers)  # -> ImageDataBunch
                            .normalize(imagenet_stats))                 # -> ImageDataBunch

    # we want the order of images to not be shuffled to be able to find the right images easily
    data.train_dl = data.train_dl.new(shuffle=False)
    data.img_size = img_size  # data object needs to know its image size
    return data


class VisualSearchEngine:
    """
    Deep-learning based visual similarity search and clustering.

    Visual similarity is defined as L2 distance in embedding space of last fully connected
    layer of a neural network.

    Supported use cases:
        - given a query image, find visually most similar images
        - given a set of images, arrange them into a collage where images are grouped in
          terms of visual similarity
    """

    def __init__(self, data_path: str, **kwargs):
        # Create data and learner
        self._validate_path(data_path)
        self.data: ImageDataBunch = get_data(data_path, **kwargs)
        self.learner = cnn_learner(self.data, models.resnet18)
        self.last_layer: nn.Module = flatten_model(self.learner.model)[-2]

        # Precompute data activations as part of initialization
        # TODO refactor computation into a separate method?
        self.activations_list: List[Tensor] = []
        self.last_layer.register_forward_hook(self.hook)
        _ = self.learner.get_preds(self.data.train_ds)
        self.data_activations = torch.cat(self.activations_list)

        # This will store activations for query image
        self.query_act = None

    def _validate_image(self, image_path: str):
        """
        Validate image, raise exception if any problem.
        Potential problems:
         - corrupt images
         - non-image files
         - non-image files disguised as images

        :param image_path: path to the image
        :return: None
        """

        try:
            img = PILImage.open(image_path)
            img.verify()
        except Exception:
            raise InvalidImageError(f"Corrupt image / non-image file: {image_path}\nAborting.")

    def _validate_path(self, path: str):
        """
        Validate the path and images have no problems.

        Handles and raises in the following cases:
        - non-existent path
        - path is not a directory
        - empty folder
        - folder only has hidden images
        - images are corrupt
        - non-image files are present
        - non-image files are present that have image extension

        :param path: path to the folder with images to validate
        """

        # TODO: handle recursive Imagenet-style folder structure validation (irrelevant for current use case)
        # Handle non-existent path
        if not os.path.exists(path):
            raise NonExistentPathError(f"Provided path doesn't exist: {path}\nCan't initialize the engine.")

        # Handle path not a dir case
        if not os.path.isdir(path):
            raise PathIsNotFolderError(f"Provided path is not a folder: {path}\nCan't initialize the engine.")

        # Collect filenames, ignore hidden files (starting with the dot)
        file_names = [name for name in os.listdir(path) if not name.startswith('.') and os.path.isfile(os.path.join(path, name))]

        # Handle empty folder path, or folder with only hidden files
        if not file_names:
            raise EmptyFolderError(f"Provided folder is empty: {path}\nCan't initialize the engine.")

        # Validate images, throw error if any problem
        for file_name in file_names:
            full_path = os.path.join(path, file_name)
            self._validate_image(full_path)

    def hook(self, module, input, output) -> None:
        """
        Hook for collecting layer activations.
        This hook is invoked for each batch's forward pass.

        :param module: layer to which the hook is attached
        :param input: batch input into the layer
        :param output: batch output of the layer
        :return: None
        """

        self.activations_list.append(output)

    @property
    def data_size(self) -> int:
        """
        Calculate number of images in the dataset.

        :return: number of images in the dataset
        """

        return len(self.data.train_ds)

    @property
    def collage_grid_size(self) -> int:
        """
        Calculate size of collage grid.

        Collage is a square grid of n by n images.
        If the size of dataset used for collage is not exactly
        a square of some integer, then some of the images will
        not be used in the collage.

        :return: size of the collage grid
        """

        return int(np.floor(np.sqrt(self.data_size)))

    @property
    def collage_data_size(self) -> int:
        """
        Calculate number of images needed for collage.

        A square collage of size n needs n**2 images.

        :return: number of images needed for the collage
        """

        return self.collage_grid_size**2

    def _get_array_from_image(self, img: Image) -> np.array:
        """
        Convert an Image object into numpy array.

        :param img: fastai Image with dimensions: (channels, height, width)
        :return: numpy array with dimensions: (height, width, channels)
        """

        img = img.data.numpy()  # (channels, height, width)
        img = img.transpose(1, 2, 0)  # (height, width, channels)
        return img

    def make_collage(self) -> np.array:
        """
        Make collage based on activation space visual similarity.

        This method:
        1. computes t-SNE embeddings
        2. computes the square grid based on embeddings
        3. creates the collage array and fills it with images from dataset

        :return: 3-channel array representing square collage image
        """

        collage_activations: Tensor = self.data_activations[:self.collage_data_size]

        # edge case: only 1 image uploaded -> return the image itself
        if len(collage_activations) == 1:
            return self._get_array_from_image(self.data.train_ds[0][0])

        # prepare the grid
        tsne: np.array = TSNE().fit_transform(collage_activations)
        grid_arrangement: Tuple[int, int] = (self.collage_grid_size, self.collage_grid_size)
        grid_xy, _ = rasterfairy.transformPointCloud2D(tsne, target=grid_arrangement)

        # create the collage
        collage_size = self.collage_grid_size * self.data.img_size
        collage = np.zeros([collage_size, collage_size, 3])  # empty collage
        for i in range(self.collage_data_size):
            row, col = map(int, grid_xy[i])  # grid has floats, but need ints to index
            up = row * self.data.img_size
            down = (row + 1) * self.data.img_size
            left = col * self.data.img_size
            right = (col + 1) * self.data.img_size
            collage[up:down, left:right] = self._get_array_from_image(self.data.train_ds[i][0])
        return collage

    def find_closest_images(self, img: Image, num: int = 16) -> List[Image]:
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
