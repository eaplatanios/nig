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

import os
import numpy as np

from . import utilities

__author__ = 'alshedivat'

SOURCE_URL = 'https://dl.dropboxusercontent.com/u/17460940/data/delicious.npz'

TRAIN_SIZE = 12920
# VALIDATION_SIZE = 3000
TEST_SIZE = 3185
NB_FEATURES = 500
NB_LABELS = 983


def maybe_download(working_dir):
    path = os.path.join(working_dir, 'delicious/data.npz')
    if not os.path.isfile(path):
        archive_path = utilities.maybe_download('data.npz',
                                                working_dir, SOURCE_URL)


def load(working_dir):
    data_dir = os.path.join(working_dir, 'delicious')
    # Load the data
    maybe_download(data_dir)
    data = np.load(os.path.join(data_dir, 'data.npz'))
    # Sanity checks
    assert data['X_train'].shape == (TRAIN_SIZE, NB_FEATURES)
    assert data['Y_train'].shape == (TRAIN_SIZE, NB_LABELS)
    assert data['X_test'].shape == (TEST_SIZE, NB_FEATURES)
    assert data['Y_test'].shape == (TEST_SIZE, NB_LABELS)
    # Split the data
    # train_data = (data['X_train'][:-VALIDATION_SIZE],
    #               data['Y_train'][:-VALIDATION_SIZE])
    # val_data = (data['X_train'][-VALIDATION_SIZE:],
    #             data['Y_train'][-VALIDATION_SIZE:])
    train_data = (data['X_train'], data['Y_train'])
    test_data = (data['X_test'], data['Y_test'])
    labels_order = data['labels_order']
    # return train_data, val_data, test_data, labels_order
    return train_data, test_data, labels_order
