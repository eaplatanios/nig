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
import tarfile

from . import utilities

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
    maybe_download(working_dir)
    train_data_file = os.path.join(working_dir, 'dbpedia_csv', 'train.csv')
    test_data_file = os.path.join(working_dir, 'dbpedia_csv', 'test.csv')
    train_data = utilities.load_csv(train_data_file, 0, False)
    val_data = train_data[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:]
    test_data = utilities.load_csv(test_data_file, 0, False)
    return train_data, val_data, test_data
