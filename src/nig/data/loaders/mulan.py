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

import arff
import glob
import logging
import numpy as np
import os
import patoolib
import six
import xml.etree.ElementTree

from . import utilities
from ..utilities import serialize_data, deserialize_data

__author__ = 'eaplatanios'

__all__ = ['dataset_info', 'load']

SOURCE_URL = 'http://sourceforge.net/projects/mulan/files/datasets/'

DATASETS = {
    'bibtex': 'bibtex.rar',
    'birds': 'birds.rar',
    'bookmarks': 'bookmarks.rar',
    'cal500': 'CAL500.rar',
    'corel5k': 'corel5k.rar',
    'corel16k': 'corel16k.rar',
    'delicious': 'delicious.rar',
    'emotions': 'emotions.rar',
    'enron': 'enron.rar',
    # 'flags': 'flags.zip',
    'genbase': 'genbase.rar',
    'mediamill': 'mediamill.rar',
    'medical': 'medical.rar',
    # 'nuswide_cvlad_plus': 'nuswide-cVLADplus.rar',
    # 'nuswide_bow': 'nuswide-bow.rar',
    'rcv1v2_subset_1': 'rcv1subset1.rar',
    'rcv1v2_subset_2': 'rcv1subset2.rar',
    'rcv1v2_subset_3': 'rcv1subset3.rar',
    'rcv1v2_subset_4': 'rcv1subset4.rar',
    'rcv1v2_subset_5': 'rcv1subset5.rar',
    'scene': 'scene.rar',
    'tmc2007': 'tmc2007.rar',
    'yahoo': 'yahoo.rar',
    'yeast': 'yeast.rar'
}

dataset_info = {
    # TODO: Add information for more datasets.
    'delicious': {
        'num_features': 500,
        'num_labels': 983},
    'emotions': {
        'num_features': 72,
        'num_labels': 6},
    'mediamill': {
        'num_features': 120,
        'num_labels': 101},
    'rcv1v2': {
        'num_features': 47236,
        'num_labels': 101},
    'scene': {
        'num_features': 294,
        'num_labels': 6},
    'yahoo': {
        'arts1': {
            'num_features': 23146,
            'num_labels': 26},
        'business1': {
            'num_features': 21924,
            'num_labels': 30},
        'computers1': {
            'num_features': 34099,
            'num_labels': 30},
        'education1': {
            'num_features': 27537,
            'num_labels': 30},
        'entertainment1': {
            'num_features': 32001,
            'num_labels': 21},
        'health1': {
            'num_features': 30607,
            'num_labels': 30},
        'reference1': {
            'num_features': 39682,
            'num_labels': 30},
        'science1': {
            'num_features': 37197,
            'num_labels': 30},
        'social1': {
            'num_features': 52359,
            'num_labels': 30},
        'society1': {
            'num_features': 31802,
            'num_labels': 27}
    }
}

logger = logging.getLogger(__name__)


def _extract_data(filename, dataset_part_name=None):
    def separate_labels(data, labels):
        label_indices = [i[0].replace('\\\'', '\'') for i in data['attributes']]
        label_indices = [i in labels for i in label_indices]
        data_indices = [not i for i in label_indices]
        data = np.array(data['data'], dtype=np.float32)
        return data[:, data_indices], data[:, label_indices]

    logger.info('Extracting ' + filename)
    directory = os.path.dirname(filename)
    patoolib.extract_archive(
        archive=filename, outdir=directory, interactive=False, verbosity=0)
    xml_files = glob.glob(os.path.join(directory, '*.xml'))
    if len(xml_files) > 0:
        xml_header_root = xml.etree.ElementTree.parse(xml_files[0]).getroot()
        labels = set(label.attrib['name'] for label in xml_header_root
                     if label.tag.endswith('label'))
    else:
        raise ValueError('Missing the XML header file.')
    dataset_files = dict()
    for filename in os.listdir(directory):
        if filename.endswith('-train.arff'):
            dataset_name = filename[:-11].lower()
            data_file = os.path.join(directory, filename)
            if dataset_name not in dataset_files:
                dataset_files[dataset_name] = dict()
            dataset_files[dataset_name]['train'] = data_file
        if filename.endswith('-test.arff'):
            dataset_name = filename[:-10].lower()
            data_file = os.path.join(directory, filename)
            if dataset_name not in dataset_files:
                dataset_files[dataset_name] = dict()
            dataset_files[dataset_name]['test'] = data_file
    if dataset_part_name is None:
        datasets = dict()
        for name, files in six.iteritems(dataset_files):
            train_data = arff.load(open(files['train'], 'r'))
            test_data = arff.load(open(files['test'], 'r'))
            train_data = separate_labels(train_data, labels)
            test_data = separate_labels(test_data, labels)
            datasets[name] = (train_data, test_data)
        if len(datasets) == 1:
            datasets = six.next(six.itervalues(datasets))
        return datasets
    if dataset_part_name not in dataset_files:
        raise ValueError('Dataset part "%s" not found.' % dataset_part_name)
    files = dataset_files[dataset_part_name]
    train_data = arff.load(open(files['train'], 'r'))
    test_data = arff.load(open(files['test'], 'r'))
    train_data = separate_labels(train_data, labels)
    test_data = separate_labels(test_data, labels)
    return train_data, test_data


def load(working_dir, dataset, dataset_part_name=None):
    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError('Unsupported dataset name %s.' % dataset)
    if dataset_part_name is None:
        train_data_file = os.path.join(working_dir, 'train_data.npy')
        test_data_file = os.path.join(working_dir, 'test_data.npy')
    else:
        train_data_file = os.path.join(
            working_dir, 'train_data_%s.npy' % dataset_part_name)
        test_data_file = os.path.join(
            working_dir, 'test_data_%s.npy' % dataset_part_name)
    if not (os.path.isfile(train_data_file) and os.path.isfile(test_data_file)):
        filename = DATASETS[dataset]
        local_file = utilities.maybe_download(
            filename=filename, working_dir=working_dir,
            source_url=SOURCE_URL + filename)
        dataset = _extract_data(
            filename=local_file, dataset_part_name=dataset_part_name)
        if isinstance(dataset, dict):
            raise ValueError('Only one dataset can be requested at a time.')
        train_data, test_data = dataset
        serialize_data(data=train_data, path=train_data_file)
        serialize_data(data=test_data, path=test_data_file)
    else:
        train_data = deserialize_data(path=train_data_file)
        test_data = deserialize_data(path=test_data_file)
    return train_data, test_data
