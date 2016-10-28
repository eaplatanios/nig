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

"""Utility functions for running experiments efficiently."""
from __future__ import absolute_import, division, print_function

import nig
import tensorflow as tf

__author__ = ['alshedivat']


def get_iterator(data,
                 batch_size=128,
                 shuffle=False,
                 cycle=False,
                 cycle_shuffle=False,
                 keep_last=True):
    return nig.NPArrayIterator(data, batch_size, shuffle=shuffle, cycle=cycle,
                               cycle_shuffle=cycle_shuffle, keep_last=keep_last)


def train(models, data, learner=nig.SimpleLearner,
          callbacks=None,
          pipelines=None,
          eval_metric=None,
          working_dir='run',
          validation_data=None,
          logging_frequency=100,
          summary_frequency=100,
          evaluation_frequency=500,
          metadata_frequency=1000,
          checkpoint_frequency=1000,
          checkpoint_file_prefix='checkpoint',
          restore_sequentially=False,
          variable_stats_frequency=200,
          predict_postprocess=None,
          save_trained=True):
    """Takes models, data, a learner and does the magic.
    Simply a wrapper that hides lots of code and specifies default parameters.

    Arguments:
    ----------
        TODO

    Returns:
    --------
        learner : Learner
            A trained instance of the provided learner.
    """

    # TODO: Run various sanity checks to ensure compatibility between
    #       the models, the data and the learner.

    if callbacks is None:
        callbacks = []

    # Set up training
    callbacks += [
        nig.LoggerCallback(frequency=logging_frequency),
        nig.SummaryWriterCallback(frequency=summary_frequency),
        nig.RunMetaDataSummaryWriterCallback(
            frequency=metadata_frequency,
            trace_level=tf.RunOptions.FULL_TRACE),
        nig.VariableStatisticsSummaryWriterCallback(
            frequency=variable_stats_frequency, variables='trainable'),
        nig.CheckpointWriterCallback(
            frequency=checkpoint_frequency, file_prefix=checkpoint_file_prefix),
        nig.EvaluationCallback(
            frequency=evaluation_frequency, data=get_iterator(data),
            metrics=eval_metric, name='eval/train'),
        nig.EvaluationCallback(
            frequency=evaluation_frequency, data=get_iterator(validation_data),
            metrics=eval_metric, name='eval/val')]

    learner = learner(models=models, predict_postprocess=predict_postprocess)

    # Train the learner
    learner.train(
        data=data, pipelines=pipelines, callbacks=callbacks,
        working_dir=working_dir, ckpt_file_prefix=checkpoint_file_prefix,
        restore_sequentially=restore_sequentially, save_trained=save_trained)

    return learner


def eval(learner, data, metric,
         working_dir='run',
         checkpoint_file_prefix='checkpoint'):
    preds = learner.predict(
        get_iterator(data, False), ckpt=-1, working_dir=working_dir,
        ckpt_file_prefix=checkpoint_file_prefix)
    return metric(preds, data[1])


def save_to_database():
    """Saves results into the database.
    """
    # TODO
