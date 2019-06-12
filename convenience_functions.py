import tensorflow as tf


def ivar(shape, bshape, name, bname):
    return tf.get_variable(name=name, shape=shape, dtype=tf.float64, initializer=tf.initializers.truncated_normal(), trainable=True)\
        ,tf.get_variable(name=bname, shape=bshape, dtype=tf.float64, initializer=tf.initializers.random_normal(), trainable=True)


def instance_norm(x):
    with tf.variable_scope("instance_norm", reuse=tf.AUTO_REUSE):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), dtype=tf.float64)
        offset = tf.get_variable('offset', [x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float64
        )
        out = scale * tf.divide(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def leaky_relu(x, leaky=True):
    if leaky:
        return tf.nn.leaky_relu(x)
    else:
        return tf.nn.relu(x)
