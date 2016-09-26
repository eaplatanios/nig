import tensorflow as tf


def leaky_relu(alpha=0.01):
    return lambda input: tf.maximum(alpha * input, input)
