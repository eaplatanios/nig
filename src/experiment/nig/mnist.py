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


class MNISTExperiment(experiments.ExperimentBase):
    def __init__(self, architectures, use_one_hot_encoding=True,
                 activation=tf.nn.relu, labeled_batch_size=100,
                 unlabeled_batch_size=100, test_data_proportion=0.1,
                 max_iter=1000, abs_loss_chg_tol=1e-6, rel_loss_chg_tol=1e-6,
                 loss_chg_iter_below_tol=5, logging_frequency=10,
                 summary_frequency=100, checkpoint_frequency=1000,
                 evaluation_frequency=10, variable_statistics_frequency=-1,
                 run_meta_data_frequency=-1,
                 working_dir=os.path.join(os.getcwd(), 'working'),
                 checkpoint_file_prefix='ckpt', restore_sequentially=False,
                 save_trained=True, optimizer=lambda: tf.train.AdamOptimizer(),
                 gradients_processor=None, seed=None):
        self.architectures = architectures
        self.use_one_hot_encoding = use_one_hot_encoding
        # self.loss = nig.L2Loss()
        self.loss = nig.CrossEntropy(
            log_outputs=False, scaled_outputs=True,
            one_hot_train_outputs=self.use_one_hot_encoding)
        optimizer_opts = {
            'batch_size': labeled_batch_size,
            'max_iter': max_iter,
            'abs_loss_chg_tol': abs_loss_chg_tol,
            'rel_loss_chg_tol': rel_loss_chg_tol,
            'loss_chg_iter_below_tol': loss_chg_iter_below_tol,
            'grads_processor': gradients_processor}
        dataset_info = loaders.mnist.dataset_info
        num_features = dataset_info['num_features']
        num_labels = dataset_info['num_labels']
        models = [nig.MultiLayerPerceptron(
            input_size=num_features, output_size=num_labels,
            hidden_layer_sizes=architecture, activation=activation,
            softmax_output=True, log_output=False,
            train_outputs_one_hot=use_one_hot_encoding, loss=self.loss,
            loss_summary=False, optimizer=optimizer,
            optimizer_opts=optimizer_opts)
                  for architecture in self.architectures]
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
        predict_postprocess = lambda l: tf.argmax(l, 1)
        inputs_pipeline = nig.ColumnsExtractor(list(range(num_features)))
        outputs_pipeline = nig.ColumnsExtractor(num_features)
        if self.use_one_hot_encoding:
            outputs_pipeline = outputs_pipeline | \
                               nig.DataTypeEncoder(np.int8) | \
                               nig.OneHotEncoder(10)
        super(MNISTExperiment, self).__init__(
            models=models, eval_metrics=eval_metrics,
            predict_postprocess=predict_postprocess,
            inputs_pipeline=inputs_pipeline, outputs_pipeline=outputs_pipeline,
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
            save_trained=save_trained, seed=seed)

    def __str__(self):
        return 'mnist'

    def experiment_information(self):
        return {'architectures': str(self.architectures),
                'loss': str(self.loss)}

    def load_data(self, test_proportion=None):
        train_data, test_data = loaders.mnist.load(
            os.path.join(self.working_dir, 'data'), float_images=True)
        if test_proportion is None:
            return train_data, test_data
        data = self._merge_datasets(train_data, test_data)
        train_indices, test_indices = experiments.stratified_split(
            labels=data[:, -1], test_proportion=test_proportion, seed=self.seed)
        return data[train_indices], data[test_indices]


if __name__ == '__main__':
    seed = 9999
    architectures = [[16], [128], [128, 64, 32], [512, 256, 128], [1024]]
    use_one_hot_encoding = True
    activation = nig.leaky_relu(0.01)
    labeled_batch_size = 128
    unlabeled_batch_size = 128
    test_data_proportion = 0.95
    max_iter = 1000
    abs_loss_chg_tol = 1e-6
    rel_loss_chg_tol = 1e-6
    loss_chg_iter_below_tol = 5
    logging_frequency = 10
    summary_frequency = -1
    checkpoint_frequency = -1
    evaluation_frequency = 50
    variable_statistics_frequency = -1
    run_meta_data_frequency = -1
    working_dir = os.path.join(os.getcwd(), 'working', 'mnist')
    checkpoint_file_prefix = 'ckpt'
    restore_sequentially = False
    save_trained = False
    optimizer = lambda: tf.train.AdamOptimizer()  # nig.gradient_descent(1e0, decay_rate=0.99)
    gradients_processor = None  # lambda g: tf.clip_by_norm(g, 0.1)

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
        experiment = MNISTExperiment(
            architectures=architectures,
            use_one_hot_encoding=use_one_hot_encoding,
            activation=activation, labeled_batch_size=labeled_batch_size,
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
            gradients_processor=gradients_processor, seed=seed)
        learners = []
        for name, configuration in consensus_configurations:
            learner = partial(nig.ConsensusLearner, **configuration)
            learners.append((name, learner))
        learners = OrderedDict(learners)
        results = experiment.run(learners)

        # maj_trust_based_learner = partial(nig.TrustBasedLearner, first_trust_update=max_iter + 1)
        # trust_based_learner = partial(
        #     nig.TrustBasedLearner, first_trust_update=10, trust_update_frequency=10)

        # test_predictions = learner.predict(
        #     _get_iterator(test_data, False), ckpt=False, working_dir=working_dir,
        #     ckpt_file_prefix=checkpoint_file_prefix)
        # test_truth = test_data[:, -1]
        # logger.info(np.mean(test_predictions == test_truth))

    experiments.save_results(
        results, filename=os.path.join(working_dir, 'results.pk'),
        update=True, use_backup=True, delete_backup=False, yaml_format=False)
    # results = experiments.load_results(
    #     filename=os.path.join(working_dir, 'results.pk'), yaml_format=False)
    experiments.plot_results(results)
