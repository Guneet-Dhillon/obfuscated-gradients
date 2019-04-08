import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from cifar_model import Model

class SAPModel(Model):

    def init_mask(self, x_input, sess):
        var = [x for x in tf.global_variables() if x.name.startswith('model') and 'SAP' in x.name]
        tf.variables_initializer(var).run(session=sess, feed_dict={'model/input/x_input_model:0': x_input})

    def _relu(self, x, leakiness=0.0):
        x = super(SAPModel, self)._relu(x, leakiness)

        x = tf.cast(x, tf.float64)
        n, w, h, c = x.get_shape().as_list()
        N = w * h * c
        x = tf.reshape(x, [-1, N])
        S = N

        zero = tf.zeros_like(x)
        p = tf.abs(x) / tf.reduce_sum(tf.abs(x), axis=1, keepdims=True)
        scale = 1.0 - tf.pow(1.0 - p, S)

        ind = tf.transpose(tfp.distributions.Categorical(probs=p, allow_nan_stats=False).sample(S))
        keep = tf.Variable(tf.zeros_like(x, dtype=tf.int32), validate_shape=False, name='SAP')
        keep = tf.batch_scatter_update(keep, ind, tf.ones_like(ind, dtype=tf.int32), use_locking=False)
        keep = keep > 0

        x = tf.where(keep, x / scale, zero)
        x = tf.reshape(x, [-1, w, h, c])
        x = tf.cast(x, tf.float32)

        return x

