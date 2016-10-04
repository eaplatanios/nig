from __future__ import absolute_import

import logging
import six.moves.cPickle as pickle

__author__ = 'eaplatanios'

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
