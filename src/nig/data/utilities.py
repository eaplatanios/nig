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
import os
import six.moves.cPickle as pickle
import yaml

from collections import OrderedDict

__author__ = 'eaplatanios'

__all__ = ['serialize_data', 'deserialize_data', 'save_yaml', 'load_yaml',
           'yaml_ordered_dump', 'yaml_ordered_load']

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


def save_yaml(data, path, ordered=False, append=True):
    if not os.path.isfile(path):
        append = False
    with open(path, 'at' if append else 'wt') as f:
        if ordered:
            yaml_ordered_dump(data, f.read(), loader=yaml.SafeLoader)
        else:
            yaml.safe_dump(data, f.read())


def load_yaml(path, ordered=False):
    with open(path, 'rt') as f:
        if ordered:
            data = yaml_ordered_load(f.read(), loader=yaml.SafeLoader)
        else:
            data = yaml.safe_load(f.read())
    return data


def yaml_ordered_dump(data, stream=None, dumper=yaml.Dumper, **kwargs):
    class OrderedDumper(dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)


def yaml_ordered_load(
        stream, loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(loader):
        pass

    def _construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping)
    return yaml.load(stream, OrderedLoader)
