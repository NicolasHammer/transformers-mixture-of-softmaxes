import os
import urllib
import logging

logger = logging.getLogger(__name__)


def wmt_dataset(directory: str = "/", train: bool = False, dev: bool = False,
                test: bool = False, train_filename: str = "train.tok.clean.bpe.32000", dev_filename: str = "newstest2013.tok.bpe.32000",
                test_filename: str = "newstest2014.tok.bpe.32000", check_files: list = ["train.tok.clean.bpe.32000.en"],
                url: str = "https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8"):
    """
    Based off of https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/wmt.html
    """
    extract_file(url=url, directory=directory, check_files=check_files, filename='wmt16_en_de.tar.gz')


def extract_file(url: str, directory: str, filename: str = None, extension: str = None, check_files: list = []):
    """
    Based off of https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/download.html#download_file_maybe_extract
    """
    filepath = os.path.join(directory, filename)
    check_files = [os.path.join(directory, str(f)) for f in check_files]

    if len(check_files) > 0 and all([os.path.isfile(filepath) for filepath in filepaths]):
        return filepath

    if not os.path.isdir(directory):
        os.makedirs(directory)

    logger.info(f"Downloading {filename}")

    # Download
    if "drive.google.come" in url:
