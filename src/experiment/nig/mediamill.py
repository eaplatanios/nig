import logging
import nig
import numpy as np
import os
import tensorflow as tf

from collections import OrderedDict
from functools import partial
from nig.data import loaders

from experiment.nig import experiments

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)


class MediaMillExperiment(experiments.ExperimentBase):
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
        # self.loss = nig.L2Loss()
        self.loss = nig.BinaryCrossEntropy(
            logit_outputs=False, one_hot_train_outputs=True)
        optimizer_opts = {
            'batch_size': labeled_batch_size,
            'max_iter': max_iter,
            'abs_loss_chg_tol': abs_loss_chg_tol,
            'rel_loss_chg_tol': rel_loss_chg_tol,
            'loss_chg_iter_below_tol': loss_chg_iter_below_tol,
            'grads_processor': gradients_processor}
        models = [nig.MultiLayerPerceptron(
            120, 101, architecture, activation=activation,
            softmax_output=False, sigmoid_output=True, log_output=False,
            train_outputs_one_hot=True, loss=self.loss, loss_summary=False,
            optimizer=optimizer, optimizer_opts=optimizer_opts)
                  for architecture in self.architectures]
        # eval_metric = nig.HammingLoss(log_predictions=False)
        eval_metrics = [
            nig.Accuracy(
                log_outputs=False, scaled_outputs=True,
                one_hot_train_outputs=True, thresholds=0.5, macro_average=True),
            nig.AreaUnderCurve(
                log_outputs=False, scaled_outputs=True,
                one_hot_train_outputs=True, curve='pr', num_thresholds=100,
                macro_average=True, name='auc'),
            nig.Precision(
                log_outputs=False, scaled_outputs=True,
                one_hot_train_outputs=True, thresholds=0.5, macro_average=True),
            nig.Recall(
                log_outputs=False, scaled_outputs=True,
                one_hot_train_outputs=True, thresholds=0.5, macro_average=True),
            nig.F1Score(
                log_outputs=False, scaled_outputs=True,
                one_hot_train_outputs=True, thresholds=0.5, macro_average=True)]
        super(MediaMillExperiment, self).__init__(
            models=models, eval_metrics=eval_metrics,
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

    def __str__(self):
        return 'mediamill'

    def experiment_information(self):
        return {'architectures': str(self.architectures),
                'loss': str(self.loss)}

    def load_data(self, test_proportion=None):
        train_data, test_data = loaders.mulan.load(
            os.path.join(self.working_dir, 'data'), 'mediamill')
        if test_proportion is None:
            return train_data, test_data
        data = (np.concatenate([train_data[0], test_data[0]], axis=0),
                np.concatenate([train_data[1], test_data[1]], axis=0))
        if isinstance(self.seed, np.random.RandomState):
            rng = self.seed
        else:
            rng = np.random.RandomState(self.seed)
        indices = rng.permutation(np.arange(data[0].shape[0]))
        num_samples = len(indices)
        num_test = int(num_samples * test_proportion)
        train_data = tuple(d[indices[:-num_test]] for d in data)
        test_data = tuple(d[indices[-num_test:]] for d in data)
        return train_data, test_data


if __name__ == '__main__':
    seed = 9999
    architectures = [[1], [8],
                     [16, 8], [32, 16],
                     [128, 64, 32, 16], [128, 32, 8], [256, 128],
                     [1024, 1024], [2048, 2048]]
    use_one_hot_encoding = True
    activation = nig.leaky_relu(0.01)
    labeled_batch_size = 128
    unlabeled_batch_size = 128
    test_data_proportion = 0.95
    max_iter = 2000
    abs_loss_chg_tol = 1e-6
    rel_loss_chg_tol = 1e-6
    loss_chg_iter_below_tol = 5
    logging_frequency = 100
    summary_frequency = -1
    checkpoint_frequency = -1
    evaluation_frequency = 50
    variable_statistics_frequency = -1
    run_meta_data_frequency = -1
    working_dir = os.path.join(os.getcwd(), 'working', 'mediamill')
    checkpoint_file_prefix = 'ckpt'
    restore_sequentially = False
    save_trained = False
    optimizer = lambda: tf.train.AdamOptimizer()  # nig.gradient_descent(1e-1, decay_rate=0.99)
    gradients_processor = None  # lambda g: tf.clip_by_norm(g, 1e-1)

    # optimizer = tf.contrib.opt.ScipyOptimizerInterface
    # optimizer_opts = {'options': {'maxiter': 10000}}

    # def consensus_loss_metric(outputs, consensus):
    #     with tf.name_scope('consensus_loss_metric'):
    #         outputs = tf.exp(outputs)
    #         metric = tf.square(tf.sub(outputs, consensus))
    #         metric = tf.reduce_sum(metric)
    #     return metric
    consensus_loss_metric = None
    consensus_configurations = experiments.get_consensus_configurations(
        consensus_loss_weights=[0.0, 1.0],
        multiplier=labeled_batch_size / unlabeled_batch_size)

    with nig.dummy():  # tf.device('/cpu:0'):
        experiment = MediaMillExperiment(
            architectures=architectures, activation=activation,
            labeled_batch_size=labeled_batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
            test_data_proportion=test_data_proportion, max_iter=max_iter,
            abs_loss_chg_tol=abs_loss_chg_tol,
            rel_loss_chg_tol=rel_loss_chg_tol,
            loss_chg_iter_below_tol=loss_chg_iter_below_tol,
            logging_frequency=logging_frequency,
            summary_frequency=summary_frequency,
            checkpoint_frequency=checkpoint_frequency,
            evaluation_frequency=evaluation_frequency,
            variable_statistics_frequency=variable_statistics_frequency,
            run_meta_data_frequency=run_meta_data_frequency,
            working_dir=working_dir,
            checkpoint_file_prefix=checkpoint_file_prefix,
            restore_sequentially=restore_sequentially,
            save_trained=save_trained, optimizer=optimizer,
            gradients_processor=gradients_processor)
        learners = []
        for name, configuration in consensus_configurations:
            learner = partial(nig.ConsensusLearner, **configuration)
            learners.append((name, learner))
        learners = OrderedDict(learners)
        results = experiment.run(learners)
    experiments.save_results(
        results, filename=os.path.join(working_dir, 'results.pk'), update=True,
        use_backup=True, delete_backup=False)
    # results = experiments.load_results(
    #     filename=os.path.join(working_dir, 'results.pk'))
    experiments.plot_results(results)
