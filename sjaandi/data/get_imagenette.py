# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from fastai.datasets import untar_data
from fastai.datasets import URLs


def main() -> Path:
    """
    Download and untar Imagenette 160 dataset.

    Read more about the dataset:
    https://github.com/fastai/imagenette

    :return: path to data.
    """
    logger = logging.getLogger(__name__)
    logger.info('getting Imagenette from the internet')

    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / 'data' / 'raw'

    path_to_data = untar_data(URLs.IMAGENETTE_160, dest=raw_data_dir)
    return path_to_data


if __name__ == '__main__':
    # Cookiecutter boilerplate
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
