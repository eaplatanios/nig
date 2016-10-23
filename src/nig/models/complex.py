# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from math import pi

__author__ = 'eaplatanios'

__all__ = ['complex_tensor', 'zero_complex', 'zero_complex_like',
           'unit_complex', 'complex_magnitude', 'complex_mul_real',
           'complex_normalize', 'complex_reflect', 'mod_relu',
           'matmul_parameterized_unitary']


def complex_tensor(name, shape, initializer=None):
    """Returns a new complex variable."""
    real = tf.get_variable(name + '_real', shape=shape, initializer=initializer)
    imag = tf.get_variable(name + '_imag', shape=shape, initializer=initializer)
    return tf.complex(real=real, imag=imag, name=name)


def zero_complex(shape, dtype=tf.float32, name=None):
    if name is not None:
        return tf.complex(
            real=tf.zeros(shape=shape, dtype=dtype, name=name + '_real'),
            imag=tf.zeros(shape=shape, dtype=dtype, name=name + '_imag'),
            name=name)
    return tf.complex(
        real=tf.zeros(shape=shape, dtype=dtype),
        imag=tf.zeros(shape=shape, dtype=dtype), name=name)


def zero_complex_like(tensor, dtype=None, name=None):
    if name is not None:
        return tf.complex(
            real=tf.zeros_like(tensor=tensor, dtype=dtype, name=name + '_real'),
            imag=tf.zeros_like(tensor=tensor, dtype=dtype, name=name + '_imag'),
            name=name)
    tf.complex(
        real=tf.zeros_like(tensor=tensor, dtype=dtype),
        imag=tf.zeros_like(tensor=tensor, dtype=dtype), name=name)


def unit_complex(name, shape):
    """Returns a unit complex number."""
    theta = tf.get_variable(
        name=name, shape=shape,
        initializer=tf.random_uniform_initializer(-pi, pi))
    return tf.complex(tf.cos(theta), tf.sin(theta))


def complex_magnitude(z):
    """Returns the magnitude of a complex number."""
    return tf.real(z) * tf.real(z) + tf.imag(z) * tf.imag(z)


def complex_mul_real(complex, real):
    return tf.complex(tf.real(complex) * real, tf.imag(complex) * real)


def complex_normalize(tensor):
    norm = tf.reduce_sum(
        input_tensor=complex_magnitude(tensor), reduction_indices=-1)
    return tf.transpose(complex_mul_real(
        tf.transpose(tensor), 1.0 / (1e-5 + tf.transpose(norm))))


def complex_reflect(input_tensor, reflection_tensor):
    reflection_tensor = tf.expand_dims(reflection_tensor, 1)
    scale = 2 * tf.matmul(input_tensor, tf.conj(reflection_tensor))
    return input_tensor - tf.matmul(scale, tf.transpose(reflection_tensor))


def mod_relu(inputs, bias, eps=1e-5):
    if not inputs.dtype.is_complex:
        raise ValueError('inputs must be a complex tensor.')
    if bias.dtype.is_complex:
        raise ValueError('bias must be a complex tensor.')
    mag = tf.complex_abs(inputs)
    return complex_mul_real(inputs, (tf.nn.relu(mag + bias) / (mag + eps)))


def matmul_parameterized_unitary(inputs, scope=None):
    # TODO: Add support for dtype.
    """Multiplies a complex tensor with a parameterized unitary matrix. The
    parameterized form is: W = D2 R1 IT D1 Perm R0 FT D0."""
    if not inputs.dtype.is_complex:
        raise ValueError('inputs must be a complex tensor.')
    shape = inputs.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError('inputs must be a 2D tensor (i.e., batch of vectors).')
    inputs_depth = shape[1]
    with tf.variable_scope(scope or 'matmul_parameterized_unitary'):
        d = [unit_complex(name='D_' + i, shape=[inputs_depth]) for i in '012']
        r = [complex_normalize(complex_tensor(
            name='R_' + i, shape=[inputs_depth],
            initializer=tf.random_uniform_initializer(-1., 1.))) for i in '01']
        perm = tf.constant(
            value=np.random.permutation(inputs_depth), dtype=tf.int32,
            name='Perm')
        # TODO: Fix this so that it can use the TensorFlow FFT ops when they support CPUs.
        fft = lambda x: tf.py_func(
            lambda a: np.fft.fft(a).astype(np.complex64), [x], [tf.complex64]
        )[0]
        ifft = lambda x: tf.py_func(
            lambda a: np.fft.ifft(a).astype(np.complex64), [x], [tf.complex64]
        )[0]
        outputs = inputs * d[0]
        outputs = complex_reflect(fft(outputs), r[0])
        outputs = d[1] * tf.transpose(tf.gather(tf.transpose(outputs), perm))
        outputs = d[2] * complex_reflect(ifft(outputs), r[1])
        return outputs / inputs_depth
