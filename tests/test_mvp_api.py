import os
from typing import Tuple
import pytest
from tempfile import mkdtemp
from tempfile import NamedTemporaryFile
from uuid import uuid4

from sjaandi import VisualSearchEngine
from sjaandi.usecases.mvp import EmptyFolderError
from sjaandi.usecases.mvp import InvalidImageError
from sjaandi.usecases.mvp import PathIsNotFolderError
from sjaandi.usecases.mvp import NonExistentPathError


def save_image(folder: str,
               resolution: Tuple[int, int] = (300, 300),
               channels: int = 3,
               extension: str = '.jpg'):
    """
    Save synthetic image of given resolution and extension to folder.

    :param folder: folder where to save the image
    :param resolution: height and width of the image
    :param channels: channels in the image
    :param extension:
    :return:
    """

    pass


def make_folder_with_files(extension: str = '.jpg', n_files=5) -> str:
    """
    Make a temp folder with n files of given extension.

    Files are filled with a random string.

    :param extension: file extension, either with dot or not
    :param n_files: number of files to generate
    :return: path to the folder with files
    """

    temp_dir = mkdtemp()
    for i in range(n_files):
        file_path = os.path.join(temp_dir, f"some-file-{i}.{extension.strip('.')}")
        with open(file_path, 'w') as f:
            f.write(str(uuid4()))
            f.close()
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
