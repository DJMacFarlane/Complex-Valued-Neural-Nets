import tensorflow as tf
import math


def abs_softmax(x):
    return tf.math.softmax(tf.math.abs(x))

def real_softmax(x):
    return tf.math.softmax(tf.math.real(x))

def cmplx_rrelu(x):
    # Take relu of just the real part keeping the imaginary part the same
    return tf.complex(tf.nn.relu(tf.math.real(x)), tf.math.imag(x))

def cmplx_crelu(x):
    # Take relu of both the real and imaginary parts
    return tf.complex(tf.nn.relu(tf.math.real(x)), tf.nn.relu(tf.math.imag(x)))

def polar_relu(x):
    # Take relu of the magnitude and keep the phase the same
    return tf.complex(tf.nn.relu(tf.math.abs(x)), tf.math.angle(x))

