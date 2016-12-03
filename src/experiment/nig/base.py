from __future__ import absolute_import, division, print_function

import abc
import nig
import numpy as np
import os
import tensorflow as tf

from six import with_metaclass

__author__ = 'eaplatanios'

__all__ = ['Experiment']


class Experiment(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, models, eval_metric, predict_postprocess=None,
                 inputs_pipeline=None, outputs_pipeline=None, batch_size=100,
                 labeled_batch_size=100, unlabeled_batch_size=100,
                 logging_frequency=10, summary_frequency=100,
                 checkpoint_frequency=1000, evaluation_frequency=10,
                 variable_statistics_frequency=-1, run_meta_data_frequency=-1,
                 working_dir=os.path.join(os.getcwd(), 'working'),
                 checkpoint_file_prefix='ckpt', restore_sequentially=False,
                 save_trained=True):
        if predict_postprocess is None:
            predict_postprocess = lambda x: x
        if inputs_pipeline is None:
            inputs_pipeline = lambda x: x
        if outputs_pipeline is None:
            outputs_pipeline = lambda x: x
        self.models = models
        self.eval_metric = eval_metric
        self.predict_postprocess = predict_postprocess
        self.inputs_pipeline = inputs_pipeline
        self.outputs_pipeline = outputs_pipeline
        self.batch_size = batch_size
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
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

    def _get_iterator(self, data, include_outputs=True):
        pipelines = [self.inputs_pipeline]
        if include_outputs:
            pipelines.append(self.outputs_pipeline)
        return nig.NPArrayIterator(
            data, self.batch_size, shuffle=True, cycle=False,
            cycle_shuffle=False, keep_last=True, pipelines=pipelines)

    def _callbacks(self, train_data=None, val_data=None, test_data=None,
                   loss_values=None, eval_train_values=None,
                   eval_val_values=None, eval_test_values=None):
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
        if self.evaluation_frequency > 0 and val_data is not None:
            callbacks.append(nig.EvaluationCallback(
                frequency=self.evaluation_frequency,
                data=self._get_iterator(val_data), metrics=self.eval_metric,
                name='eval/val', stored_values=eval_val_values))
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
        train_data, val_data, test_data = self.load_data()

        def _run_learner(learner):
            losses = []
            train_evals = []
            val_evals = []
            test_evals = []
            callbacks = self._callbacks(
                train_data=train_data, val_data=val_data, test_data=test_data,
                loss_values=losses, eval_train_values=train_evals,
                eval_val_values=val_evals, eval_test_values=test_evals)
            learner = learner(
                models=self.models,
                predict_postprocess=self.predict_postprocess)
            labeled_data = nig.get_iterator(
                data=np.concatenate([train_data, val_data], axis=0),
                batch_size=self.labeled_batch_size, shuffle=True,
                cycle=True, cycle_shuffle=True,
                pipelines=[self.inputs_pipeline, self.outputs_pipeline])
            unlabeled_data = nig.get_iterator(
                data=test_data, batch_size=self.unlabeled_batch_size,
                shuffle=True, cycle=True, cycle_shuffle=True,
                pipelines=[self.inputs_pipeline])
            learner.train(
                data=train_data,
                pipelines=[self.inputs_pipeline, self.outputs_pipeline],
                init_option=True, per_model_callbacks=None,
                combined_model_callbacks=callbacks,
                working_dir=self.working_dir,
                ckpt_file_prefix=self.checkpoint_file_prefix,
                restore_sequentially=self.restore_sequentially,
                save_trained=self.save_trained, labeled_data=labeled_data,
                unlabeled_data=unlabeled_data)
            return losses, train_evals, val_evals, test_evals

        if isinstance(learners, list):
            learners = {str(learner): learner for learner in learners}
        losses = dict()
        train_evals = dict()
        val_evals = dict()
        test_evals = dict()
        for name, learner in learners.items():
            results = _run_learner(learner)
            losses[name] = results[0]
            train_evals[name] = results[1]
            val_evals[name] = results[2]
            test_evals[name] = results[3]
        if show_plots or plots_folder is not None:
            if plots_folder is not None:
                plots_folder = os.path.join(self.working_dir, plots_folder)
                loss_filename = os.path.join(plots_folder, 'loss.pdf')
                train_filename = os.path.join(plots_folder, 'train_eval.pdf')
                val_filename = os.path.join(plots_folder, 'val_eval.pdf')
                test_filename = os.path.join(plots_folder, 'test_eval.pdf')
            else:
                loss_filename = None
                train_filename = None
                val_filename = None
                test_filename = None
            nig.plot_lines(
                lines=losses, xlabel='Iteration', ylabel='Loss Value',
                title='Loss Function Value', include_legend=True,
                show_plot=show_plots, save_filename=loss_filename)
            nig.plot_lines(
                lines=train_evals, xlabel='Iteration',
                ylabel=str(self.eval_metric),
                title=str(self.eval_metric) + ' Value', include_legend=True,
                show_plot=show_plots, save_filename=train_filename)
            nig.plot_lines(
                lines=val_evals, xlabel='Iteration',
                ylabel=str(self.eval_metric),
                title=str(self.eval_metric) + ' Value', include_legend=True,
                show_plot=show_plots, save_filename=val_filename)
            nig.plot_lines(
                lines=test_evals, xlabel='Iteration',
                ylabel=str(self.eval_metric),
                title=str(self.eval_metric) + ' Value', include_legend=True,
                show_plot=show_plots, save_filename=test_filename)
        return losses, train_evals, val_evals, test_evals
