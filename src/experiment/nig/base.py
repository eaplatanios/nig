from __future__ import absolute_import, division, print_function

import abc
import nig
import numpy as np
import os
import tensorflow as tf

from collections import OrderedDict
from six import with_metaclass

__author__ = 'eaplatanios'

__all__ = ['Experiment']


class Experiment(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, models, eval_metric, predict_postprocess=None,
                 inputs_pipeline=None, outputs_pipeline=None,
                 labeled_batch_size=100, unlabeled_batch_size=100,
                 test_data_proportion=0.1, logging_frequency=10,
                 summary_frequency=100, checkpoint_frequency=1000,
                 evaluation_frequency=10, variable_statistics_frequency=-1,
                 run_meta_data_frequency=-1,
                 working_dir=os.path.join(os.getcwd(), 'working'),
                 checkpoint_file_prefix='ckpt', restore_sequentially=False,
                 save_trained=True):
        if predict_postprocess is None:
            predict_postprocess = lambda x: x
        if inputs_pipeline is None and outputs_pipeline is not None:
            inputs_pipeline = lambda x: x
        if outputs_pipeline is None and inputs_pipeline is not None:
            outputs_pipeline = lambda x: x
        self.models = models
        self.eval_metric = eval_metric
        self.predict_postprocess = predict_postprocess
        self.inputs_pipeline = inputs_pipeline
        self.outputs_pipeline = outputs_pipeline
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.test_data_proportion = test_data_proportion
        self.logging_frequency = logging_frequency
        self.summary_frequency = summary_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.evaluation_frequency = evaluation_frequency
        self.variable_statistics_frequency = variable_statistics_frequency
        self.run_meta_data_frequency = run_meta_data_frequency
        self.working_dir = working_dir
        self.checkpoint_file_prefix = checkpoint_file_prefix
        self.restore_sequentially = restore_sequentially
        self.save_trained = save_trained

    @abc.abstractmethod
    def load_data(self):
        pass

    @staticmethod
    def _merge_data_sets(*data_sets):
        data_set_type = type(data_sets[0])
        for data_set in data_sets:
            if not isinstance(data_set, data_set_type):
                raise TypeError('All data sets being merged must have the same '
                                'data type.')
        if data_set_type == np.ndarray:
            return np.concatenate(data_sets)
        if data_set_type == tuple:
            return tuple(np.concatenate(d) for d in zip(*data_sets))
        if data_set_type == list:
            return [np.concatenate(d) for d in zip(*data_sets)]
        if data_set_type == dict:
            keys = set(data_sets[0].keys)
            for data_set in data_sets:
                if set(data_set.keys()) != keys:
                    raise ValueError('All data sets must contain the same '
                                     'dictionary keys.')
            return {k: np.concatenate(tuple(d[k] for d in data_sets))
                    for k in keys}
        raise TypeError('Unsupported data sets type %s.' % data_set_type)

    @staticmethod
    def _split_data_set(data_set, test_proportion=0.1):
        # TODO: Guarantee even label split using our cross-validation module.
        num_train = int(np.floor((1 - test_proportion) * data_set.shape[0]))
        return data_set[:num_train], data_set[num_train:]

    def _get_iterator(self, data, include_outputs=True):
        if self.inputs_pipeline is not None:
            pipelines = [self.inputs_pipeline]
        elif self.outputs_pipeline is None:
            pipelines = None
        else:
            pipelines = [lambda x: x]
        if include_outputs and self.outputs_pipeline is not None:
            pipelines.append(self.outputs_pipeline)
        return nig.get_iterator(
            data, shuffle=True, cycle=False,
            cycle_shuffle=False, keep_last=True, pipelines=pipelines)

    def _callbacks(self, train_data=None, test_data=None, loss_values=None,
                   eval_train_values=None, eval_test_values=None):
        callbacks = []
        if self.logging_frequency > 0:
            callbacks.append(nig.LoggerCallback(
                frequency=self.logging_frequency, stored_values=loss_values))
        if self.summary_frequency > 0:
            callbacks.append(nig.SummaryWriterCallback(
                frequency=self.summary_frequency))
        if self.checkpoint_frequency > 0:
            callbacks.append(nig.CheckpointWriterCallback(
                frequency=self.checkpoint_frequency,
                file_prefix=self.checkpoint_file_prefix))
        if self.evaluation_frequency > 0 and train_data is not None:
            callbacks.append(nig.EvaluationCallback(
                frequency=self.evaluation_frequency,
                data=self._get_iterator(train_data), metrics=self.eval_metric,
                name='eval/train', stored_values=eval_train_values))
        if self.evaluation_frequency > 0 and test_data is not None:
            callbacks.append(nig.EvaluationCallback(
                frequency=self.evaluation_frequency,
                data=self._get_iterator(test_data), metrics=self.eval_metric,
                name='eval/test', stored_values=eval_test_values))
        if self.variable_statistics_frequency > 0:
            callbacks.append(nig.VariableStatisticsSummaryWriterCallback(
                frequency=self.variable_statistics_frequency,
                variables='trainable'))
        if self.run_meta_data_frequency > 0:
            callbacks.append(nig.RunMetaDataSummaryWriterCallback(
                frequency=self.run_meta_data_frequency,
                trace_level=tf.RunOptions.FULL_TRACE))
        return callbacks

    def run(self, learners, show_plots=True, plots_folder=None):
        data = self._merge_data_sets(*self.load_data())
        train_data, test_data = self._split_data_set(
            data_set=data, test_proportion=self.test_data_proportion)

        def _run_learner(learner):
            losses = []
            train_evals = []
            test_evals = []
            callbacks = self._callbacks(
                train_data=train_data, test_data=test_data, loss_values=losses,
                eval_train_values=train_evals, eval_test_values=test_evals)
            if self.inputs_pipeline is not None:
                labeled_pipelines = [self.inputs_pipeline]
                unlabeled_pipelines = [self.inputs_pipeline]
            else:
                labeled_pipelines = [lambda x: x]
                unlabeled_pipelines = None
            if self.outputs_pipeline is None:
                labeled_pipelines.append(lambda x: x)
            else:
                labeled_pipelines.append(self.outputs_pipeline)
            learner = learner(
                models=self.models, new_graph=True,
                predict_postprocess=self.predict_postprocess)
            labeled_data = nig.get_iterator(
                data=train_data, batch_size=self.labeled_batch_size,
                shuffle=True, cycle=True, cycle_shuffle=True,
                pipelines=labeled_pipelines)
            unlabeled_data = nig.get_iterator(
                data=test_data, batch_size=self.unlabeled_batch_size,
                shuffle=True, cycle=True, cycle_shuffle=True,
                pipelines=unlabeled_pipelines)
            learner.train(
                labeled_data=labeled_data, pipelines=labeled_pipelines,
                init_option=True, per_model_callbacks=None,
                combined_model_callbacks=callbacks,
                working_dir=self.working_dir,
                ckpt_file_prefix=self.checkpoint_file_prefix,
                restore_sequentially=self.restore_sequentially,
                save_trained=self.save_trained, unlabeled_data=unlabeled_data)
            return losses, train_evals, test_evals

        if isinstance(learners, list):
            learners = OrderedDict([(str(learner), learner)
                                    for learner in learners])
        losses = dict()
        train_evals = dict()
        test_evals = dict()
        for name, learner in learners.items():
            results = _run_learner(learner)
            losses[name] = results[0]
            train_evals[name] = results[1]
            test_evals[name] = results[2]
        if show_plots or plots_folder is not None:
            if plots_folder is not None:
                plots_folder = os.path.join(self.working_dir, plots_folder)
                loss_filename = os.path.join(plots_folder, 'loss.pdf')
                train_filename = os.path.join(plots_folder, 'train_eval.pdf')
                test_filename = os.path.join(plots_folder, 'test_eval.pdf')
            else:
                loss_filename = None
                train_filename = None
                test_filename = None
            nig.plot_lines(
                lines=losses, style='ggplot', xlabel='Iteration',
                ylabel='Loss Value', title='Loss Function Value',
                include_legend=True, show_plot=show_plots,
                save_filename=loss_filename, dpi=300)
            nig.plot_lines(
                lines=train_evals, style='ggplot', xlabel='Iteration',
                ylabel=str(self.eval_metric),
                title=str(self.eval_metric) + ' Value', include_legend=True,
                show_plot=show_plots, save_filename=train_filename, dpi=300)
            nig.plot_lines(
                lines=test_evals, style='ggplot', xlabel='Iteration',
                ylabel=str(self.eval_metric),
                title=str(self.eval_metric) + ' Value', include_legend=True,
                show_plot=show_plots, save_filename=test_filename, dpi=300)
        return losses, train_evals, test_evals
