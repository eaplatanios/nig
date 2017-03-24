from __future__ import absolute_import, division, print_function

import abc
import logging
import nig
import numpy as np
import os
import shutil
import tensorflow as tf

from collections import OrderedDict
from matplotlib import pyplot as plt
from six import with_metaclass

__author__ = 'eaplatanios'

__all__ = ['get_consensus_configurations', 'ExperimentBase']


logger = logging.getLogger(__name__)


def get_consensus_configurations(consensus_loss_weights, multiplier=1.0):
    configurations = []
    for weight in consensus_loss_weights:
        configurations.extend([
            # ('Majority ' + str(weight), {
            #     'consensus_method': nig.Vote(
            #         trainable=False, hard_vote=False, argmax_vote=False),
            #     'consensus_loss_weight': weight * multiplier,
            #     'consensus_loss_metric': None,
            #     'first_consensus': 10}),
            ('Trainable Majority ' + str(weight), {
                'consensus_method': nig.Vote(
                    trainable=True, hard_vote=False, argmax_vote=False),
                'consensus_loss_weight': weight * multiplier,
                'consensus_loss_metric': None,
                'first_consensus': 10,
                'first_consensus_max_iter': 1000,
                'consensus_update_frequency': 10,
                'consensus_update_max_iter': 500}),
            ('RBM ' + str(weight), {
                'consensus_method': nig.RBMConsensus(),
                'consensus_loss_weight': weight * multiplier,
                'consensus_loss_metric': None,
                'first_consensus': 10,
                'first_consensus_max_iter': 10000,
                'consensus_update_frequency': 100,
                'consensus_update_max_iter': 1000})
        ])
    return configurations


def stratified_split(labels, test_proportion, seed=None):
    # TODO: Move this into a "split" module next to the cross_validation module.
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    num_samples = len(labels)
    num_test = int(num_samples * test_proportion)
    num_train = num_samples - num_test
    unique_labels, label_indices = np.unique(labels, return_inverse=True)
    num_labels = len(unique_labels)
    label_counts = np.bincount(label_indices)
    if np.min(label_counts) < 2:
        raise ValueError('The least populated label only has 1 sample, which '
                         'is not enough. The minimum allowed number of samples '
                         'for any label is 2.')
    if num_train < num_labels:
        raise ValueError('The number of train samples (%d) should be greater '
                         'than or equal to the number of classes (%d).'
                         % (num_train, num_labels))
    if num_test < num_labels:
        raise ValueError('The number of test samples (%d) should be greater '
                         'than or equal to the number of classes (%d).'
                         % (num_test, num_labels))
    # TODO: Convert this into a yielding loop.
    # If there are ties in the class-counts, we want to make sure to break them.
    train_counts = _approximate_hypergeometric_mode(
        label_counts=label_counts, num_samples=num_train, seed=rng)
    label_counts_remaining = label_counts - train_counts
    test_counts = _approximate_hypergeometric_mode(
        label_counts=label_counts_remaining, num_samples=num_test, seed=rng)
    train_indices = []
    test_indices = []
    for i, label in enumerate(unique_labels):
        permutation = rng.permutation(label_counts[i])
        label_indices = np.where((labels == label))[0][permutation]
        train = train_counts[i]
        test = test_counts[i]
        train_indices.extend(label_indices[:train])
        test_indices.extend(label_indices[train:train+test])
    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)
    return train_indices, test_indices


def _approximate_hypergeometric_mode(label_counts, num_samples, seed=None):
    """Computes the approximate mode of a multivariate hypergeometric
    distribution.
    This is an approximation to the mode of the multivariate
    hypergeometric given by class_counts and n_draws.
    It shouldn't be off by more than one.
    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts."""
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    # This computes a bad approximation to the mode of the multivariate
    # hypergeometric given by label_counts and num_samples.
    continuous = num_samples * label_counts / label_counts.sum()
    # Floored means we don't overshoot num_samples, but probably undershoot.
    floored = np.floor(continuous)
    # We add samples according to how much "left over" probability they had,
    # until we arrive at num_samples.
    need_to_add = int(num_samples - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        # Add according to the remainder, but break ties randomly to avoid bias.
        for value in values:
            indices, = np.where(remainder == value)
            # If we need to add less than what is in indices we draw randomly
            # from them. If we need to add more, we add them all and go to the
            # next value.
            add_now = min(len(indices), need_to_add)
            indices = rng.choice(indices, size=add_now, replace=False)
            floored[indices] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(np.int)


def log_results_summary(results, metrics=None):
    for experiment_name, experiments_results in results.items():
        for experiment_info, experiment_results in experiments_results.items():
            names = list(experiment_results.keys())
            values = list(experiment_results.values())
            data_metrics = list(values[0]['evaluations'].keys())
            if metrics is not None:
                metrics = [m for m in metrics if m in data_metrics]
            else:
                metrics = data_metrics
            logger.info('Results for "%s":.' % experiment_name.lower())
            for metric in metrics:
                train_evaluations = [v['evaluations'][metric][0][-1]
                                     for v in values]
                test_evaluations = [v['evaluations'][metric][1][-1]
                                    for v in values]
                logger.info('\tMetric "%s":' % metric.lower())
                for name, train_eval, test_eval \
                        in zip(names, train_evaluations, test_evaluations):
                    name += ":" + " " * (30 - len(name))
                    logger.info("\t\t%s Train = %.4f , Test = %.4f"
                                % (name, train_eval[1], test_eval[1]))


def plot_results(results, metrics=None):
    for experiment_name, experiments_results in results.items():
        for experiment_info, experiment_results in experiments_results.items():
            names = list(experiment_results.keys())
            values = list(experiment_results.values())
            data_metrics = list(values[0]['evaluations'].keys())
            if metrics is not None:
                metrics = [m for m in metrics if m in data_metrics]
            else:
                metrics = data_metrics
            num_rows = len(metrics) + 1
            with plt.style.context('ggplot'):
                figure = plt.figure(figsize=[100, 12])
                figure.suptitle(experiment_name.upper(), fontsize=20)
                losses = [v['losses'] for v in values]
                loss_axes = figure.add_subplot(num_rows, 1, 1)
                nig.plot_lines(
                    lines=losses, names=names, style='ggplot',
                    xlabel='Iteration', ylabel='Loss Value',
                    title='Loss Function Value', font_size=12,
                    include_legend=True, show_plot=False, save_filename=None,
                    dpi=300, figure=figure, axes=loss_axes)
                subplot_index = 3
                for metric in metrics:
                    train_evaluations = [v['evaluations'][metric][0]
                                         for v in values]
                    train_eval_axes = figure.add_subplot(
                        num_rows, 2, subplot_index)
                    subplot_index += 1
                    nig.plot_lines(
                        lines=train_evaluations, names=names, style='ggplot',
                        xlabel='Iteration', ylabel=metric.title(),
                        title='Train ' + metric.title() + ' Value',
                        font_size=12, include_legend=False, show_plot=False,
                        save_filename=None, dpi=300, figure=figure,
                        axes=train_eval_axes)
                    test_evaluations = [v['evaluations'][metric][1]
                                        for v in values]
                    test_eval_axes = figure.add_subplot(
                        num_rows, 2, subplot_index, sharey=train_eval_axes)
                    subplot_index += 1
                    nig.plot_lines(
                        lines=test_evaluations, names=names, style='ggplot',
                        xlabel='Iteration',  # ylabel=metric.title(),
                        title='Test ' + metric.title() + ' Value',
                        font_size=12, include_legend=False, show_plot=False,
                        save_filename=None, dpi=300, figure=figure,
                        axes=test_eval_axes)
                    plt.setp(test_eval_axes.get_yticklabels(), visible=False)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


def save_results(results, filename, update=True, use_backup=True,
                 delete_backup=False, yaml_format=True):
    if update and os.path.isfile(filename):
        if use_backup:
            shutil.copy2(filename, filename + '.bak')
        if yaml_format:
            old_results = nig.load_yaml(filename)
        else:
            old_results = nig.deserialize_data(filename)
        results = merge_results(old_results, results)
    if yaml_format:
        nig.save_yaml(results, filename)
    else:
        nig.serialize_data(results, filename)
    if use_backup and delete_backup:
        os.remove(filename + '.bak')


def load_results(filename, yaml_format=True):
    if yaml_format:
        return nig.load_yaml(filename)
    return nig.deserialize_data(filename)


def merge_results(results_1, results_2):
    merged_results = results_1.copy()
    for experiment_name, experiment_results in results_2.items():
        results = merged_results.get(experiment_name, dict())
        results.update(experiment_results)
        merged_results[experiment_name] = results
    return merged_results


class ExperimentBase(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, models, eval_metrics, predict_postprocess=None,
                 inputs_pipeline=None, outputs_pipeline=None,
                 labeled_batch_size=100, unlabeled_batch_size=100,
                 test_data_proportion=0.1, logging_frequency=10,
                 summary_frequency=100, checkpoint_frequency=1000,
                 evaluation_frequency=10, variable_statistics_frequency=-1,
                 run_meta_data_frequency=-1,
                 working_dir=os.path.join(os.getcwd(), 'working'),
                 checkpoint_file_prefix='ckpt', restore_sequentially=False,
                 save_trained=True, seed=None):
        if predict_postprocess is None:
            predict_postprocess = lambda x: x
        if inputs_pipeline is None and outputs_pipeline is not None:
            inputs_pipeline = lambda x: x
        if outputs_pipeline is None and inputs_pipeline is not None:
            outputs_pipeline = lambda x: x
        self.models = models
        if not isinstance(eval_metrics, list):
            eval_metrics = [eval_metrics]
        self.eval_metrics = eval_metrics
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
        self.seed = seed

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def experiment_information(self):
        pass

    @abc.abstractmethod
    def load_data(self, test_proportion=None):
        pass

    @staticmethod
    def _merge_datasets(*datasets):
        dataset_type = type(datasets[0])
        for dataset in datasets:
            if not isinstance(dataset, dataset_type):
                raise TypeError('All data sets being merged must have the same '
                                'data type.')
        if dataset_type == np.ndarray:
            return np.concatenate(datasets)
        if dataset_type == tuple:
            return tuple(np.concatenate(d) for d in zip(*datasets))
        if dataset_type == list:
            return [np.concatenate(d) for d in zip(*datasets)]
        if dataset_type == dict:
            keys = set(datasets[0].keys)
            for dataset in datasets:
                if set(dataset.keys()) != keys:
                    raise ValueError('All data sets must contain the same '
                                     'dictionary keys.')
            return {k: np.concatenate(tuple(d[k] for d in datasets))
                    for k in keys}
        raise TypeError('Unsupported data sets type %s.' % dataset_type)

    # @staticmethod
    # def _split_dataset(dataset, test_proportion=0.1):
    #     # TODO: Guarantee even label split using our cross-validation module.
    #     num_train = int(np.floor((1 - test_proportion) * dataset.shape[0]))
    #     return dataset[:num_train], dataset[num_train:]

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
                frequency=self.logging_frequency, stored_values=loss_values,
                name='loss'))
        if self.summary_frequency > 0:
            callbacks.append(nig.SummaryWriterCallback(
                frequency=self.summary_frequency))
        if self.checkpoint_frequency > 0:
            callbacks.append(nig.CheckpointWriterCallback(
                frequency=self.checkpoint_frequency,
                file_prefix=self.checkpoint_file_prefix))
        for i, metric in enumerate(self.eval_metrics):
            if self.evaluation_frequency > 0 and train_data is not None:
                callbacks.append(nig.EvaluationCallback(
                    frequency=self.evaluation_frequency,
                    data=self._get_iterator(train_data), metrics=metric,
                    name='eval/train/' + metric.name_scope,
                    stored_values=eval_train_values[i]))
            if self.evaluation_frequency > 0 and test_data is not None:
                callbacks.append(nig.EvaluationCallback(
                    frequency=self.evaluation_frequency,
                    data=self._get_iterator(test_data), metrics=metric,
                    name='eval/test/' + metric.name_scope,
                    stored_values=eval_test_values[i]))
        if self.variable_statistics_frequency > 0:
            callbacks.append(nig.VariableStatisticsSummaryWriterCallback(
                frequency=self.variable_statistics_frequency,
                variables='trainable'))
        if self.run_meta_data_frequency > 0:
            callbacks.append(nig.RunMetaDataSummaryWriterCallback(
                frequency=self.run_meta_data_frequency,
                trace_level=tf.RunOptions.FULL_TRACE))
        return callbacks

    def run(self, learners):
        if isinstance(learners, list):
            names = [str(learner) for learner in learners]
        elif isinstance(learners, dict) or isinstance(learners, OrderedDict):
            names = list(learners.keys())
            learners = list(learners.values())
        else:
            raise TypeError('Unsupported learners type %s.' % type(learners))
        train_data, test_data = self.load_data(
            test_proportion=self.test_data_proportion)

        def _run_learner(learner):
            losses = []
            train_evals = [[] for _ in self.eval_metrics]
            test_evals = [[] for _ in self.eval_metrics]
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
            consensus_learner = False
            try:
                learner = learner(
                    models=self.models, new_graph=True,
                    predict_postprocess=self.predict_postprocess)
                consensus_learner = True
            except TypeError:
                learner = learner(
                    models=self.models,
                    predict_postprocess=self.predict_postprocess)
            labeled_data = nig.get_iterator(
                data=train_data, batch_size=self.labeled_batch_size,
                shuffle=True, cycle=True, cycle_shuffle=True,
                pipelines=labeled_pipelines)
            unlabeled_data = nig.get_iterator(
                data=test_data, batch_size=self.unlabeled_batch_size,
                shuffle=True, cycle=True, cycle_shuffle=True,
                pipelines=unlabeled_pipelines)
            if consensus_learner:
                learner.train(
                    labeled_data=labeled_data, pipelines=labeled_pipelines,
                    init_option=True, per_model_callbacks=None,
                    combined_model_callbacks=callbacks,
                    working_dir=self.working_dir,
                    ckpt_file_prefix=self.checkpoint_file_prefix,
                    restore_sequentially=self.restore_sequentially,
                    save_trained=self.save_trained,
                    unlabeled_data=unlabeled_data)
            else:
                learner.train(
                    data=labeled_data, pipelines=labeled_pipelines,
                    init_option=True, callbacks=callbacks,
                    working_dir=self.working_dir,
                    ckpt_file_prefix=self.checkpoint_file_prefix,
                    restore_sequentially=self.restore_sequentially,
                    save_trained=self.save_trained)
            return losses, train_evals, test_evals

        results = []
        for name, learner in zip(names, learners):
            learner_results = _run_learner(learner)
            results.append((name, {
                'losses': learner_results[0],
                'evaluations': OrderedDict(
                    [(str(metric), (learner_results[1][i],
                                    learner_results[2][i]))
                     for i, metric in enumerate(self.eval_metrics)])}))
        information = frozenset(self.experiment_information().items())
        return {str(self): {information: OrderedDict(results)}}
