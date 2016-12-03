from __future__ import absolute_import, division, print_function

import nig
import numpy as np
import os
import tensorflow as tf

from nig.data.loaders import mnist

from . import base

__author__ = 'eaplatanios'

__all__ = ['MNISTExperiment']


class MNISTExperiment(base.Experiment):
    def __init__(self, architectures, use_one_hot_encoding=True,
                 activation=tf.nn.relu, batch_size=128, labeled_batch_size=100,
                 unlabeled_batch_size=100, max_iter=1000, abs_loss_chg_tol=1e-6,
                 rel_loss_chg_tol=1e-6, loss_chg_iter_below_tol=5,
                 logging_frequency=10, summary_frequency=100,
                 checkpoint_frequency=1000, evaluation_frequency=10,
                 variable_statistics_frequency=-1, run_meta_data_frequency=-1,
                 working_dir=os.path.join(os.getcwd(), 'working'),
                 checkpoint_file_prefix='ckpt', restore_sequentially=False,
                 save_trained=True, optimizer=lambda: tf.train.AdamOptimizer(),
                 gradients_processor=None):
        self.architectures = architectures
        self.use_one_hot_encoding = use_one_hot_encoding
        if self.use_one_hot_encoding:
            loss = nig.CrossEntropyOneHotEncodingMetric()
        else:
            loss = nig.CrossEntropyIntegerEncodingMetric()
        optimizer_opts = {
            'batch_size': batch_size,
            'max_iter': max_iter,
            'abs_loss_chg_tol': abs_loss_chg_tol,
            'rel_loss_chg_tol': rel_loss_chg_tol,
            'loss_chg_iter_below_tol': loss_chg_iter_below_tol,
            'grads_processor': gradients_processor}
        models = [nig.MultiLayerPerceptron(
            784, 10, architecture, activation=activation,
            softmax_output=use_one_hot_encoding, use_log=use_one_hot_encoding,
            train_outputs_one_hot=use_one_hot_encoding, loss=loss,
            loss_summary=False, optimizer=optimizer,
            optimizer_opts=optimizer_opts)
                  for architecture in self.architectures]
        if self.use_one_hot_encoding:
            eval_metric = nig.AccuracyOneHotEncodingMetric()
        else:
            eval_metric = nig.AccuracyIntegerEncodingMetric()
        predict_postprocess = lambda l: tf.argmax(l, 1)
        inputs_pipeline = nig.ColumnsExtractor(list(range(784)))
        outputs_pipeline = nig.ColumnsExtractor(784)
        if self.use_one_hot_encoding:
            outputs_pipeline = outputs_pipeline | \
                              nig.DataTypeEncoder(np.int8) | \
                              nig.OneHotEncoder(10)
        super(MNISTExperiment, self).__init__(
            models=models, eval_metric=eval_metric,
            predict_postprocess=predict_postprocess,
            inputs_pipeline=inputs_pipeline, outputs_pipeline=outputs_pipeline,
            batch_size=batch_size, labeled_batch_size=labeled_batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
            logging_frequency=logging_frequency,
            summary_frequency=summary_frequency,
            checkpoint_frequency=checkpoint_frequency,
            evaluation_frequency=evaluation_frequency,
            variable_statistics_frequency=variable_statistics_frequency,
            run_meta_data_frequency=run_meta_data_frequency,
            working_dir=working_dir,
            checkpoint_file_prefix=checkpoint_file_prefix,
            restore_sequentially=restore_sequentially,
            save_trained=save_trained)

    def load_data(self):
        return mnist.load(
            os.path.join(self.working_dir, 'data'), float_images=True)
