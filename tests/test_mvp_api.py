import os
from functools import partial
from typing import Tuple
import pytest
from tempfile import mkdtemp
from tempfile import NamedTemporaryFile
from uuid import uuid4

import imageio
import numpy as np

from sjaandi import VisualSearchEngine
from sjaandi.usecases.mvp import EmptyFolderError
from sjaandi.usecases.mvp import InvalidImageError
from sjaandi.usecases.mvp import PathIsNotFolderError
from sjaandi.usecases.mvp import NonExistentPathError
from sjaandi.usecases.mvp import InvalidParameterValueError


def save_random_image(full_name: str,
                      resolution: Tuple[int, int] = (300, 300),
                      channels: int = 3):
    """
    Save synthetic image of given resolution and extension to folder.

    :param full_name: filename, including a valid extension
    :param resolution: height and width of the image
    :param channels: channels in the image
    :return:
    """

    supported_channels = [1, 3, 4]
    if channels not in supported_channels:
        raise InvalidParameterValueError(f"Images that have {channels} channels are not supported.")

    img = np.random.randint(0, high=256, size=(*resolution, channels), dtype=np.uint8)
    try:
        imageio.imsave(full_name, img)
    except ValueError:
        print(f"Filename wrong extension: {full_name}")
    except FileNotFoundError:
        print(f"Directory doesn't exist: {full_name}")


def save_random_file(file_path: str):
    """
    Create a file at given path, fill with random string, save.

    :param file_path: full path to the file to be created.
    :return:
    """
    with open(file_path, 'w') as f:
        f.write(str(uuid4()))
        f.close()


def make_folder_with_files(extension: str = '.jpg', n_files=5, file_type: str = 'random', **kwargs) -> str:
    """
    Make a temp folder with n files of given extension.

    .. note:: kwargs are passed through to file_creating_function and are function-specific

    :param extension: file extension, either with dot or not
    :param n_files: number of files to generate
    :param file_type: which type of file to create:
        - 'random': file with random string contents
        - 'image': random valid image
    :return: path to the folder with files
    """

    # assign file creating function
    if file_type == 'random':
        file_creating_function = partial(save_random_file, **kwargs)
    elif file_type == 'image' or file_type == 'images':
        file_creating_function = partial(save_random_image, **kwargs)
    else:
        raise InvalidParameterValueError(f"File type '{file_type}' is not supported.")

    # create files
    temp_dir = mkdtemp()
    for i in range(n_files):
        file_path = os.path.join(temp_dir, f"some-file-{i}.{extension.strip('.')}")
        try:
            file_creating_function(file_path)
        except TypeError:
            raise InvalidParameterValueError(f"Function '{file_creating_function.func.__name__}' cannot accept arguments {kwargs}.")

    return temp_dir


@pytest.fixture()
def empty_folder_path() -> str:
    """
    Create an empty folder.
    :return: path to folder
    """

    return mkdtemp()


@pytest.fixture()
def temp_file_path() -> str:
    """
    Create a temp file.
    :return: path to the file
    """

    return NamedTemporaryFile().name


@pytest.fixture()
def nonexistent_path() -> str:
    """
    Create path that doesn't exits.
    :return: non-existent path
    """

    return str(uuid4())


@pytest.fixture()
def non_images_path() -> str:
    """
    Make a folder with 5 random PDF files.
    :return: path to the folder
    """

    return make_folder_with_files('.pdf')


@pytest.fixture()
def evil_images_path() -> str:
    """
    Make a folder with 5 invalid JPG files.
    :return: path to the folder
    """

    return make_folder_with_files('.jpg')


@pytest.fixture()
def valid_large_random_images() -> str:
    """
    Make a folder with 10 valid images that are larger than size accepted by neural net.

    Data should properly resize them down.

    :return: path to the folder
    """

    return make_folder_with_files('.jpg', file_type='image', resolution=(200, 200), n_files=10)


@pytest.fixture()
def valid_small_random_images() -> str:
    """
    Make a folder with 6 valid images that are smaller than size accepted by neural net.

    Data should properly resize them up.

    :return: path to the folder
    """

    return make_folder_with_files('.jpg', file_type='image', resolution=(100, 100), n_files=6)


@pytest.fixture()
def valid_random_four_channel_images() -> str:
    """
    Make a folder with 5 valid images that have 4 channels.

    :return: path to the folder
    """

    # use .png because that supports 4 channels
    return make_folder_with_files('.png', file_type='image', resolution=(300, 300), n_files=6, channels=4)


@pytest.fixture()
def valid_random_one_channel_images() -> str:
    """
    Make a folder with 5 valid images that have one channel.

    :return: path to the folder
    """

    return make_folder_with_files('.jpg', file_type='image', resolution=(300, 300), n_files=6, channels=1)


@pytest.fixture()
def valid_random_one_image() -> str:
    """
    Make a folder with one image.

    :return: path to the folder
    """

    return make_folder_with_files('.png', file_type='image', resolution=(300, 300), n_files=1)


class TestInvalidDataPath:
    def test_api_throws_non_existent_path_error_with_nonexistent_path(self, nonexistent_path):
        """
        VisualSearchEngine should raise an error when passed a path
        that doesn't exist.
        """

        with pytest.raises(NonExistentPathError):
            engine = VisualSearchEngine(nonexistent_path)

    def test_api_throws_non_folder_error_with_file_path(self):
        """
        VisualSearchEngine should raise an error when passed a path
        that is not a folder.
        """

        with NamedTemporaryFile() as temp:
            with pytest.raises(PathIsNotFolderError):
                engine = VisualSearchEngine(temp.name)

    def test_api_throws_empty_folder_error_with_empty_folder_path(self, empty_folder_path):
        """
        VisualSearchEngine should raise an error when passed a path
        to a folder that is empty.
        """

        with pytest.raises(EmptyFolderError):
            engine = VisualSearchEngine(empty_folder_path)

    def test_api_throws_error_for_path_with_non_images(self, non_images_path):
        """
        VisualSearchEngine should raise an error when passed a path
        to a folder with files that are not images.
        """

        with pytest.raises(InvalidImageError):
            engine = VisualSearchEngine(non_images_path)

    def test_api_throws_error_for_path_with_evil_images(self, evil_images_path):
        """
        VisualSearchEngine should raise an error when passed a path
        to a folder with files with images extensions but that are
        not images initially.
        """

        with pytest.raises(InvalidImageError):
            engine = VisualSearchEngine(evil_images_path)


class TestWorksWithActualImages:
    def test_api_works_with_actual_images_of_greater_size(self, valid_large_random_images):
        collage = VisualSearchEngine(valid_large_random_images).make_collage()

    def test_api_works_with_actual_images_of_smaller_size(self, valid_small_random_images):
        collage = VisualSearchEngine(valid_small_random_images).make_collage()

    def test_api_works_with_four_channel_images(self, valid_random_four_channel_images):
        collage = VisualSearchEngine(valid_random_four_channel_images).make_collage()

    def test_api_works_with_one_channel_images(self, valid_random_one_channel_images):
        collage = VisualSearchEngine(valid_random_one_channel_images).make_collage()

    def test_api_works_with_one_image(self, valid_random_one_image):
        collage = VisualSearchEngine(valid_random_one_image).make_collage()
