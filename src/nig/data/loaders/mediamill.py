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

import numpy as np

from . import utilities

__author__ = 'alshedivat'

SOURCE_URL = 'https://dl.dropboxusercontent.com/u/17460940/data/mediamill.npz'

TRAIN_SIZE = 35907
VALIDATION_SIZE = 3000
TEST_SIZE = 5000
NB_FEATURES = 120
NB_LABELS = 101


def maybe_download(working_dir):
    path = os.path.join(working_dir, 'mediamill/data.npz')
    if not os.path.isfile(path):
        archive_path = utilities.maybe_download('data.npz',
                                                working_dir, SOURCE_URL)


def load(working_dir):
    data_dir = os.path.join(working_dir, 'mediamill')
    # Load the data
    maybe_download(data_dir)
    data = np.load(os.path.join(data_dir, 'data.npz'))
    # Sanity checks
    assert (data['X'].shape ==
            (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, NB_FEATURES))
    assert (data['Y'].shape ==
            (TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE, NB_LABELS))
    # Split the data
    train_data = (data['X'][:TRAIN_SIZE],
                  data['Y'][:TRAIN_SIZE])
    val_data = (data['X'][TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE],
                data['Y'][TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE])
    test_data = (data['X'][-TEST_SIZE:],
                 data['Y'][-TEST_SIZE:])
    labels_order = data['labels_order']
    return train_data, val_data, test_data, labels_order
