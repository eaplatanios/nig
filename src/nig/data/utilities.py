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

import logging
import six.moves.cPickle as pickle

__author__ = 'eaplatanios'

__all__ = ['serialize_data', 'deserialize_data']

logger = logging.getLogger(__name__)


def serialize_data(data, path):
    """Serializes the provided data using cPickle.

    Args:
        data: Python data structure to serialize.
        path (str): Path to the serialized file to create.
    """
    logger.info('Serializing data to file %s.' % path)
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=2)
    logger.info('Done serializing data to file %s.' % path)


def deserialize_data(path):
    """Deserializes the provided file using cPickle.

    Args:
        path (str): Path to the serialized file.

    Returns:
        Deserialized Python data structure.
    """
    logger.info('Deserializing data from file %s.' % path)
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data
