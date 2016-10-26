# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import gzip
import logging
import numpy as np
import os

from . import utilities

__author__ = 'eaplatanios'

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
VALIDATION_SIZE = 5000
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

logger = logging.getLogger(__name__)


def _read32(bytestream):
    dtype = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dtype)[0]


def extract_images(filename):
    logger.info('Extracting ' + filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s.'
                             % (magic, filename))
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
            raise ValueError('Invalid magic number %d in MNIST label file: %s.'
                             % (magic, filename))
        num_items = _read32(bytestream)
        buffer = bytestream.read(num_items)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels


def load(working_dir, float_images=True):
    working_dir = os.path.join(working_dir, 'mnist')
    if float_images:
        train_data_file = os.path.join(working_dir, 'train_data.npy')
        test_data_file = os.path.join(working_dir, 'test_data.npy')
    else:
        train_data_file = os.path.join(working_dir, 'train_data_float.npy')
        test_data_file = os.path.join(working_dir, 'test_data_float.npy')
    if not (os.path.isfile(train_data_file) and os.path.isfile(test_data_file)):
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
        train_data = np.column_stack([train_images[:]] + [train_labels])
        test_data = np.column_stack([test_images[:]] + [test_labels])
        np.save(train_data_file, train_data)
        np.save(test_data_file, test_data)
    else:
        train_data = np.load(train_data_file)
        test_data = np.load(test_data_file)
    val_data = train_data[:VALIDATION_SIZE]
    train_data = train_data[:VALIDATION_SIZE]
    return train_data, val_data, test_data
