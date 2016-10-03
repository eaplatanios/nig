from __future__ import absolute_import

import os
import tarfile

from nig.data.loaders import utilities

__author__ = 'eaplatanios'

SOURCE_URL = 'https://googledrive.com/host/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2S' \
             'EpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M/dbpedia_csv.tar.gz'
VALIDATION_SIZE = 5000


def maybe_download(working_dir):
    train_path = os.path.join(working_dir, 'dbpedia_csv/train.csv')
    test_path = os.path.join(working_dir, 'dbpedia_csv/test.csv')
    if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
        archive_path = utilities.maybe_download('dbpedia_csv.tar.gz',
                                                working_dir, SOURCE_URL)
        tar_file = tarfile.open(archive_path, 'r:*')
        tar_file.extractall(working_dir)


def load(working_dir):
    data_dir = os.path.join(working_dir, 'dbpedia')
    maybe_download(data_dir)
    train_data_file = os.path.join(data_dir, 'dbpedia_csv', 'train.csv')
    test_data_file = os.path.join(data_dir, 'dbpedia_csv', 'test.csv')
    train_data = utilities.load_csv(train_data_file, 0, False)
    val_data = train_data[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:]
    test_data = utilities.load_csv(test_data_file, 0, False)
    return train_data, val_data, test_data
