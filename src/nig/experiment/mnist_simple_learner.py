from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import numpy as np
import os
import tensorflow as tf

from nig.data.aggregators import NPArrayColumnsAggregator
from nig.data.encoders import OneHotEncoder, DataTypeEncoder, ReshapeEncoder
from nig.data.extractors import NPArrayColumnsExtractor
from nig.data.iterators import NPArrayIterator
from nig.learn.callback import LoggerCallback, SummaryWriterCallback, \
    CheckpointWriterCallback, EvaluationCallback
from nig.learn.evaluation import CrossEntropyOneHotEncodingMetric, \
    AccuracyOneHotEncodingMetric, CrossEntropyIntegerEncodingMetric, \
    AccuracyIntegerEncodingMetric
from nig.learn.learner import TensorFlowLearner
from nig.learn.symbol import MultiLayerPerceptron

learning_rate = 0.01
batch_size = 100
number_of_iterations = 2000
working_dir = os.getcwd()
checkpoint_file_prefix = 'checkpoint'
restore_sequentially = False
save_trained = False


data = read_data_sets('data', False)
aggregate = NPArrayColumnsAggregator().aggregator
inputs_pipeline = NPArrayColumnsExtractor(list(range(784))).extractor
labels_pipeline = NPArrayColumnsExtractor(784).extractor | DataTypeEncoder(np.int8).encoder | OneHotEncoder(10).encoder
train_data = NPArrayIterator(
    data.train.next_batch(data.train.num_examples, False) >> aggregate,
    batch_size, shuffle=False, cycle=False, cycle_shuffle=False,
    keep_last_batch=True, pipelines=[inputs_pipeline, labels_pipeline])
eval_data = NPArrayIterator(
    data.validation.next_batch(data.validation.num_examples, False) >> aggregate,
    batch_size, shuffle=False, cycle=False, cycle_shuffle=False,
    keep_last_batch=True, pipelines=[inputs_pipeline, labels_pipeline])
test_data = NPArrayIterator(
    data.test.next_batch(data.test.num_examples, False) >> aggregate,
    batch_size, shuffle=False, cycle=False, cycle_shuffle=False,
    keep_last_batch=True, pipelines=[inputs_pipeline, labels_pipeline])

symbol = MultiLayerPerceptron(784, 10, [128, 32], softmax_output=True)

learner = TensorFlowLearner(symbol, inputs_dtype=tf.float32,
                            outputs_dtype=tf.float32, loss_summary=True,
                            gradient_norm_summary=True)
loss = CrossEntropyOneHotEncodingMetric()
callbacks = []
callbacks.append(LoggerCallback(frequency=100, header_frequency=1000))
callbacks.append(SummaryWriterCallback(frequency=100, working_dir=working_dir))
callbacks.append(CheckpointWriterCallback(frequency=1000,
                                          working_dir=working_dir, file_prefix=checkpoint_file_prefix))
callbacks.append(EvaluationCallback(frequency=1000, iterator=train_data, metrics=AccuracyOneHotEncodingMetric(), name='Train Accuracy'))
callbacks.append(EvaluationCallback(frequency=1000, iterator=eval_data, metrics=AccuracyOneHotEncodingMetric(), name='Eval Accuracy'))
callbacks.append(EvaluationCallback(frequency=1000, iterator=test_data, metrics=AccuracyOneHotEncodingMetric(), name='Test Accuracy'))
learner.train(loss, train_data, learning_rate=learning_rate,
              batch_size=batch_size, number_of_iterations=number_of_iterations,
              initialization_option=True, callbacks=callbacks,
              working_dir=working_dir,
              checkpoint_file_prefix=checkpoint_file_prefix,
              restore_sequentially=restore_sequentially,
              save_trained=save_trained)
