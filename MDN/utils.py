""" Utils for creating neural networks."""
import tensorflow as tf


def linear(x, n_output, activation=None):
    """Creates a dense layer.
    Parameters
    ----------
    x : Tensor.
        Input tensor.
    n_output : int.
        number of output neurons.
    activation: tf.nn.activation.
        activation to use.

    Returns
    -------
    h: Tensor.
        result tensor.
    """
    with tf.variable_scope("Linear"):
        w = tf.get_variable("W",
                            [x.get_shape().as_list()[1], n_output],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable("b",
                            [n_output],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

        h = tf.add(tf.matmul(x, w), b)

        if activation is not None:
            h = activation(h)
            return h, w

        else:
            return h, w
