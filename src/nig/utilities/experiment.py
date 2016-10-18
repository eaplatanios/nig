"""
Utility functions for running experiments efficiently.
"""
import logging
import tensorflow as tf

from nig.data.iterators import NPArrayIterator
from nig.data.processors import *
from nig.learning.callbacks import *
from nig.learning.metrics import *
from nig.learning.learners import *
from nig.learning.optimizers import *
from nig.utilities.generic import get_from_module

__author__ = ['alshedivat']


def train(models, data, learner='SimpleLearner',
          callbacks=[],
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

    # Set up training
    callbacks += [
        LoggerCallback(frequency=logging_frequency),
        SummaryWriterCallback(frequency=summary_frequency),
        RunMetaDataSummaryWriter(
            frequency=metadata_frequency,
            trace_level=tf.RunOptions.FULL_TRACE),
        VariableStatisticsSummaryWriterCallback(
            frequency=variable_stats_frequency, variables='trainable'),
        CheckpointWriterCallback(
            frequency=checkpoint_frequency, file_prefix=checkpoint_file_prefix),
        EvaluationCallback(
            frequency=evaluation_frequency,
            data=get_iterator(data, pipelines=pipelines),
            metrics=eval_metric, name='eval/train'),
        EvaluationCallback(
            frequency=evaluation_frequency,
            data=get_iterator(validation_data, pipelines=pipelines),
            metrics=eval_metric, name='eval/valid')
    ]

    Learner = get_from_module(learner, globals(), 'experiment')
    learner = Learner(models=models, predict_postprocess=predict_postprocess)

    # Train the learner
    learner.train(
        data=data, pipelines=pipelines,
        per_model_callbacks=callbacks,
        combined_model_callbacks=callbacks,
        working_dir=working_dir,
        ckpt_file_prefix=checkpoint_file_prefix,
        restore_sequentially=restore_sequentially,
        save_trained=save_trained)

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
