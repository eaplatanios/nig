import numpy as np
import tensorflow as tf

from nig.learning.models import Input
from nig.learning.models import MultiLayerPerceptron as MLP
from nig.learning.models import ADIOS


def test_MultiLayerPerceptron():
    N = 100
    input_sz = 64
    output_sz = 10
    architecture = [64, 32, 16]
    rng = np.random.RandomState(42)

    # Generate some dummy data
    data = rng.normal(size=(N, input_sz))

    # Construct an MLP
    model = MLP(input_sz, output_sz, architecture, tf.nn.relu)

    # Evaluate
    graph = tf.Graph()
    with graph.as_default():
        # Construct symbols
        inputs_op = tf.placeholder(tf.float32, [None, input_sz])
        outputs_op = model(inputs_op)

        # Evaluate symbols
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        outputs = session.run([outputs_op], {inputs_op: data})

    assert outputs[0].shape == (N, output_sz)


def test_ADIOS():
    N = 100
    input_sz = 64
    output_sz = [8, 16]
    architecture = [64, 32, 16]
    rng = np.random.RandomState(42)

    # Generate some dummy data
    data = rng.normal(size=(N, input_sz))

    # Construct an ADIOS model
    model = ADIOS([input_sz], output_sz, architecture, tf.nn.relu)

    # Evaluate
    graph = tf.Graph()
    with graph.as_default():
        # Construct symbols
        inputs_op = tf.placeholder(tf.float32, [None, input_sz])
        outputs_op = model(inputs_op)

        # Evaluate symbols
        session = tf.Session()
        session.run(tf.initialize_all_variables())
        outputs = session.run(outputs_op, {inputs_op: data})

    assert outputs[0].shape == (N, output_sz[0])
    assert outputs[1].shape == (N, output_sz[1])
