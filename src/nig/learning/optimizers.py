import tensorflow as tf

__author__ = 'eaplatanios'


def gradient_descent(
        learning_rate, decay_steps=100, decay_rate=1.0, staircase=False,
        learning_rate_summary=False, use_locking=False, name='GradientDescent'):
    def builder():
        if decay_rate != 1.0:
            learning_rate_variable = tf.train.exponential_decay(
                learning_rate=float(learning_rate),
                global_step=tf.contrib.framework.get_or_create_global_step(),
                decay_steps=int(decay_steps), decay_rate=decay_rate,
                staircase=staircase)
        else:
            learning_rate_variable = learning_rate
        if learning_rate_summary:
            tf.scalar_summary('train/learning_rate', learning_rate_variable)
        return tf.train.GradientDescentOptimizer(
            learning_rate_variable, use_locking, name)
    return builder
