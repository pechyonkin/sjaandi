from typing import Tuple
import pytest
from uuid import uuid4
from tempfile import NamedTemporaryFile
from tempfile import mkdtemp

from sjaandi import VisualSearchEngine
from sjaandi.usecases.mvp import NonExistentPathError
from sjaandi.usecases.mvp import EmptyFolderError
from sjaandi.usecases.mvp import PathIsNotFolderError


TEST_DATA_PATH = 'data/raw/test-data'
NONEXISTENT_PATH = str(uuid4())
FILE_PATH = NamedTemporaryFile().name
print(FILE_PATH)
EMPTY_FOLDER_PATH = mkdtemp()
PCT_PARTIAL = 0.05  # percentage of full data to use


def save_image(folder: str,
               resolution: Tuple[int, int] = (300,300),
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


@pytest.fixture()
def empty_data_path():
    pass


def test_api_throws_non_existent_path_error_with_nonexistent_path():
    """
    VisualSearchEngine should raise an error when passed a path
    that doesn't exist.
    """
    with pytest.raises(NonExistentPathError):
        engine = VisualSearchEngine(NONEXISTENT_PATH,
                                    pct_partial=PCT_PARTIAL)


def test_api_throws_non_folder_error_with_file_path():
    """
    VisualSearchEngine should raise an error when passed a path
    that is not a folder.
    """
    temp = NamedTemporaryFile()
    with pytest.raises(PathIsNotFolderError):
        engine = VisualSearchEngine(temp.name,
                                    pct_partial=PCT_PARTIAL)
    temp.close()


def test_api_throws_empty_folder_error_with_empty_folder_path():
    """
    VisualSearchEngine should raise an error when passed a path
    to a folder that is empty.
    """
    with pytest.raises(EmptyFolderError):
        engine = VisualSearchEngine(EMPTY_FOLDER_PATH,
                                    pct_partial=PCT_PARTIAL)





