import tensorflow as tf

__author__ = 'eaplatanios'


def leaky_relu(alpha=0.01):
    return lambda input: tf.maximum(alpha * input, input)
