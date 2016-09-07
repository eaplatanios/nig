from nig.data.iterators import NPArrayIterator
from nig.data.loaders import mnist
from nig.data.processors import *
from nig.learning.callbacks import *
from nig.learning.metrics import *
from nig.learning.learners import SimpleLearner, MultiModelValidationSetLearner
from nig.learning.optimizers import gradient_descent
from nig.learning.processors import norm_summary, norm_clipping
from nig.learning.symbols import MultiLayerPerceptron

__author__ = 'eaplatanios'

use_one_hot_encoding = False
architectures = [[5], [64, 32, 16]]
activation = tf.nn.relu
optimizer = gradient_descent(1e-1, decay_rate=0.99, learning_rate_summary=True)
gradients_processor = None #norm_clipping(clip_norm=0.1) \
                      #| norm_summary(name='gradients/norm')
batch_size = 100
max_iter = 2000
loss_chg_tol = 1e-6
loss_chg_iter_below_tol = 5
logging_frequency = 100
summary_frequency = 100
checkpoint_frequency = 1000
evaluation_frequency = 1000
working_dir = os.path.join(os.getcwd(), 'run', str(architectures))
checkpoint_file_prefix = 'checkpoint'
restore_sequentially = False
save_trained = False

train_data, val_data, test_data = mnist.load('data', float_images=True)

inputs_pipeline = ColumnsExtractor(list(range(784)))
labels_pipeline = ColumnsExtractor(784)
if use_one_hot_encoding:
    labels_pipeline = labels_pipeline | \
                      DataTypeEncoder(np.int8) | \
                      OneHotEncoder(10)


def get_iterator(mnist_data, include_labels=True):
    pipelines = [inputs_pipeline]
    if include_labels:
        pipelines.append(labels_pipeline)
    return NPArrayIterator(mnist_data, batch_size, shuffle=False, cycle=False,
                           cycle_shuffle=False, keep_last=True,
                           pipelines=pipelines)

symbols = [MultiLayerPerceptron(784, 10, architecture, activation=activation,
                              softmax_output=use_one_hot_encoding,
                              use_log=use_one_hot_encoding)
           for architecture in architectures]

outputs_dtype = tf.float32 if use_one_hot_encoding else tf.int32
output_shape = 10 if use_one_hot_encoding else 1

learner = MultiModelValidationSetLearner(
    symbols, inputs_dtype=tf.float32, outputs_dtype=outputs_dtype,
    output_shape=output_shape, predict_postprocess=lambda l: tf.argmax(l, 1))

loss = CrossEntropyOneHotEncodingMetric() if use_one_hot_encoding \
    else CrossEntropyIntegerEncodingMetric()
eval_metric = AccuracyOneHotEncodingMetric() if use_one_hot_encoding \
    else AccuracyIntegerEncodingMetric()

callbacks = [
    LoggerCallback(frequency=logging_frequency),
    SummaryWriterCallback(frequency=summary_frequency,
                          working_dir=working_dir),
    VariableStatisticsSummaryWriterCallback(frequency=200,
                                            variables='trainable'),
    CheckpointWriterCallback(frequency=checkpoint_frequency,
                             working_dir=working_dir,
                             file_prefix=checkpoint_file_prefix),
    EvaluationCallback(frequency=evaluation_frequency,
                       iterator=get_iterator(train_data),
                       metrics=eval_metric, name='eval/train'),
    EvaluationCallback(frequency=evaluation_frequency,
                       iterator=get_iterator(val_data),
                       metrics=eval_metric, name='eval/val'),
    EvaluationCallback(frequency=evaluation_frequency,
                       iterator=get_iterator(test_data),
                       metrics=eval_metric, name='eval/test'),
]

learner.train(
    loss, (get_iterator(train_data), get_iterator(val_data)),
    optimizers=optimizer, max_iter=max_iter, loss_chg_tol=loss_chg_tol,
    loss_chg_iter_below_tol=loss_chg_iter_below_tol, init_option=True,
    callbacks=callbacks, loss_summary=True,
    gradients_processor=gradients_processor,
    run_metadata_collection_frequency=1000,
    trace_level=tf.RunOptions.FULL_TRACE, working_dir=working_dir,
    checkpoint_file_prefix=checkpoint_file_prefix,
    restore_sequentially=restore_sequentially, save_trained=save_trained)
test_predictions = learner.predict(
    get_iterator(test_data, False), checkpoint=-1, working_dir=working_dir,
    checkpoint_file_prefix=checkpoint_file_prefix)
test_truth = test_data[:, -1]
print(np.mean(test_predictions == test_truth))

# Test loading the best performing trained model using a simple learner
simple_learner = SimpleLearner(
    symbols[learner.best_symbol], inputs_dtype=tf.float32,
    outputs_dtype=outputs_dtype, output_shape=output_shape,
    predict_postprocess=lambda l: tf.argmax(l, 1))
simple_learner_test_predictions = simple_learner.predict(
    get_iterator(test_data, False), checkpoint=-1, working_dir=working_dir,
    checkpoint_file_prefix=checkpoint_file_prefix)
print(np.mean(simple_learner_test_predictions == test_truth))
