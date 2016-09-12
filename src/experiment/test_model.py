import tensorflow as tf

from nig.learning.models import Model

a = tf.Variable(tf.zeros([200]), name='biases')
b = tf.Variable(tf.zeros([200]), name='biases')
c = a + b
d = tf.reduce_mean(c)
m = Model([a, b], [c, d])
variables = m._variables()
print(variables)
