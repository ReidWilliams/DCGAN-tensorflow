import tensorflow as tf

# he initialization for dense layers
he_init = tf.contrib.layers.variance_scaling_initializer

def conv2d(inputs, filters):
    return tf.layers.conv2d(inputs, filters, 5, strides=2, padding='same')

# conv2d transpose
def conv2dtr(inputs, filters):
    return tf.layers.conv2d_transpose(inputs, filters, 5, strides=2, padding='same')

def dense(inputs, units):
    return tf.layers.dense(inputs, units, kernel_initializer=he_init())

# batch normalization
# unclear how important, but use scale=True and epsilon 1e-5
# from https://github.com/carpedm20/DCGAN-tensorflow/blob/b138300623b933e2076872e7f812ba553e862355/ops.py
class BN:
    def __init__(self, is_training=True):
        self.is_training = is_training

    def __call__(self, inputs):
        return tf.contrib.layers.batch_norm(
            inputs, updates_collections=None, is_training=self.is_training,
            scale=True, epsilon=1e-5)

def flatten(inputs):
    return tf.contrib.layers.flatten(inputs)

def reshape(inputs, shape):
    return tf.reshape(inputs, shape)

# def elu(inputs):
#     return tf.nn.elu(inputs)

def lrelu(inputs, leak=0.2):
    return tf.maximum(inputs, leak*inputs)

def sigmoid(inputs):
    return tf.sigmoid(inputs)

def tanh(inputs):
    return tf.tanh(inputs)