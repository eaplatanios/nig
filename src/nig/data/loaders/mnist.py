from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np

from nig.data.loaders import utilities
from nig.utilities import logger

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
VALIDATION_SIZE = 5000


def _read32(bytestream):
    dtype = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dtype)[0]


def extract_images(filename):
    logger.info('Extracting ' + filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filename))
        num_images = _read32(bytestream)
        num_rows = _read32(bytestream)
        num_cols = _read32(bytestream)
        buffer = bytestream.read(num_rows * num_cols * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)
        return data


def extract_labels(filename):
    logger.info('Extracting ' + filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, filename))
        num_items = _read32(bytestream)
        buffer = bytestream.read(num_items)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels


def load(working_dir, float_images=True):
    local_file = utilities.maybe_download(TRAIN_IMAGES, working_dir,
                                          SOURCE_URL + TRAIN_IMAGES)
    train_images = extract_images(local_file)
    local_file = utilities.maybe_download(TRAIN_LABELS, working_dir,
                                          SOURCE_URL + TRAIN_LABELS)
    train_labels = extract_labels(local_file)
    local_file = utilities.maybe_download(TEST_IMAGES, working_dir,
                                          SOURCE_URL + TEST_IMAGES)
    test_images = extract_images(local_file)
    local_file = utilities.maybe_download(TEST_LABELS, working_dir,
                                          SOURCE_URL + TEST_LABELS)
    test_labels = extract_labels(local_file)
    train_images = train_images.reshape(train_images.shape[0],
                                        train_images.shape[1] *
                                        train_images.shape[2])
    test_images = test_images.reshape(test_images.shape[0],
                                      test_images.shape[1] *
                                      test_images.shape[2])
    if float_images:
        train_images = train_images.astype(np.float32)
        test_images = test_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    val_images = train_images[:VALIDATION_SIZE]
    val_labels = train_labels[:VALIDATION_SIZE]
    train_data = np.column_stack([train_images[:]] + [train_labels])
    val_data = np.column_stack([val_images[:]] + [val_labels])
    test_data = np.column_stack([test_images[:]] + [test_labels])
    return train_data, val_data, test_data
