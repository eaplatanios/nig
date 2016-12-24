from __future__ import absolute_import, division, print_function

import nig
import os
import tensorflow as tf

from nig.data.loaders import mediamill

from . import base

__author__ = 'eaplatanios'

__all__ = ['MediaMillExperiment']


class MediaMillExperiment(base.Experiment):
    def __init__(self, architectures, activation=tf.nn.relu,
                 labeled_batch_size=100, unlabeled_batch_size=100,
                 test_data_proportion=0.1, max_iter=1000, abs_loss_chg_tol=1e-6,
                 rel_loss_chg_tol=1e-6, loss_chg_iter_below_tol=5,
                 logging_frequency=10, summary_frequency=100,
                 checkpoint_frequency=1000, evaluation_frequency=10,
                 variable_statistics_frequency=-1, run_meta_data_frequency=-1,
                 working_dir=os.path.join(os.getcwd(), 'working'),
                 checkpoint_file_prefix='ckpt', restore_sequentially=False,
                 save_trained=True, optimizer=lambda: tf.train.AdamOptimizer(),
                 gradients_processor=None):
        self.architectures = architectures
        loss = nig.CrossEntropy(log_predictions=True, one_hot_truth=True)
        optimizer_opts = {
            'batch_size': labeled_batch_size,
            'max_iter': max_iter,
            'abs_loss_chg_tol': abs_loss_chg_tol,
            'rel_loss_chg_tol': rel_loss_chg_tol,
            'loss_chg_iter_below_tol': loss_chg_iter_below_tol,
            'grads_processor': gradients_processor}
        models = [nig.MultiLayerPerceptron(
            120, 101, architecture, activation=activation, softmax_output=False,
            sigmoid_output=True, log_output=True, train_outputs_one_hot=True,
            loss=loss, loss_summary=False, optimizer=optimizer,
            optimizer_opts=optimizer_opts)
                  for architecture in self.architectures]
        eval_metric = nig.HammingLoss()
        super(MediaMillExperiment, self).__init__(
            models=models, eval_metric=eval_metric,
            labeled_batch_size=labeled_batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
            test_data_proportion=test_data_proportion,
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
        train_data, val_data, test_data, _ = mediamill.load(
            os.path.join(self.working_dir, 'data'))
        return train_data, val_data, test_data
