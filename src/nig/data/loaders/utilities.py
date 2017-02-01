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

import csv
import logging
import numpy as np
import os

from six.moves import urllib

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)


def load_csv(filename, target_column=-1, has_header=True):
    with open(filename, 'r', encoding='utf-8') as csv_file:
        data_file = csv.reader(csv_file)
        if has_header:
            header = next(data_file)
            num_samples = int(header[0])
            num_features = int(header[1])
            data = np.empty([num_samples, num_features + 1])
            for i, ir in enumerate(data_file):
                data[i, -1] = np.asarray(ir.pop(target_column))
                data[i, 0:-1] = np.asarray(ir)
        else:
            data = []
            for ir in data_file:
                label = np.asarray(ir.pop(target_column))[np.newaxis]
                data.append(np.concatenate([np.asarray(ir), label], axis=0))
            data = np.array(data)
    return data


def maybe_download(filename, working_dir, source_url):
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    filepath = os.path.join(working_dir, filename)
    if not os.path.isfile(filepath):
        urllib.request.urlretrieve(source_url, filepath)
        size = os.path.getsize(filepath)
        logger.info('Successfully downloaded ' + filename +
                    ' (' + str(size) + ' bytes).')
    return filepath
