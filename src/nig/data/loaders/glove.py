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

import codecs
import numpy as np

from nig.data.utilities import deserialize_data, serialize_data

__author__ = 'eaplatanios'


def load_map(path):
    ext = path.split('.')[-1]
    if ext == 'txt':
        with codecs.open(path, 'r', encoding='UTF-8') as file:
            lines = [line.rstrip().split(' ') for line in file.readlines()]
        return {l[0]: np.array([float(v) for v in l[1:]]) for l in lines}
    elif ext == 'bin':
        return deserialize_data(path)
    raise ValueError('Unsupported file extension %s.' % ext)


def save_map(path, mapping):
    ext = path.split('.')[-1]
    if ext == 'txt':
        with codecs.open(path, 'w', encoding='UTF-8') as file:
            for k, v in mapping.items():
                file.write(str(k) + ' ' + ' '.join(v.tolist()))
    elif ext == 'bin':
        serialize_data(mapping, path)
    else:
        raise ValueError('Unsupported file extension %s.' % ext)


def l2_normalize_word_vectors(mapping):
    return {k: v / np.linalg.norm(v, ord=2) for k, v in mapping.items()}
