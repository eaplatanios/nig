import numpy as np
import tensorflow as tf

from nig.learning.metrics import CrossEntropyIntegerEncodingMetric
from nig.learning.models import MultiLayerPerceptron as MLP
from nig.learning.optimizers import gradient_descent
from nig.learning.learners import SimpleLearner
from nig.data.processors import ColumnsExtractor
from nig.data.iterators import NPArrayIterator


def test_simple_learner():
    N = 1000
    input_sz = 64
    output_sz = 10
    architecture = [64, 32, 16]
    rng = np.random.RandomState(42)

    # Generate some dummy data
    X = rng.normal(size=(N, input_sz))
    Y = rng.randint(0, 10, size=(N, 1))

    # Loss and optimizer
    loss = CrossEntropyIntegerEncodingMetric()
    optimizer = gradient_descent(1e-1, decay_rate=0.99)

    # Construct an MLP
    model = MLP(input_sz, output_sz, architecture, tf.nn.relu,
                loss=loss, optimizer=optimizer)

    # Construct a SimpleLearner
    learner = SimpleLearner(model,
                            predict_postprocess=lambda l: tf.argmax(l, 1))

    # Train the learner for a few steps
    max_iter = 10
    learner.train((X, Y),  max_iter=max_iter, init_option=True)

    # TODO: add any necessary assertions here
