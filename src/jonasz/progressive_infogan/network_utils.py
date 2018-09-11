import tensorflow as tf
from jonasz.lib import tensor_util
import numpy as np
from tensorflow import keras
import math
from tensorflow.python.framework import tensor_util as tf_tensor_util


# Taken from Nvidia's progressive_growing_of_GANs.
def upscale2d(x, factor=2, data_format='NCHW'):
  assert isinstance(factor, int) and factor >= 1, factor
  if factor == 1: return x
  with tf.variable_scope('Upscale2D'):
    if data_format == 'NHWC':
      x = tensor_util.nhwc_to_nchw(x)
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    if data_format == 'NHWC':
      x = tensor_util.nchw_to_nhwc(x)
    return x


# Taken from Nvidia's progressive_growing_of_GANs.
def downscale2d(x, factor=2, data_format='NCHW'):
  assert isinstance(factor, int) and factor >= 1
  if factor == 1: return x
  with tf.variable_scope('Downscale2D'):
    if data_format == 'NCHW':
      ksize = [1, 1, factor, factor]
    else:
      ksize = [1, factor, factor, 1]
    res = tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID',
                         data_format=data_format)
  return res


def downgrade2d(x, factor=2, data_format='NCHW'):
  x = downscale2d(x, factor, data_format)
  x = upscale2d(x, factor, data_format)
  return x


def concat_features_stddev(layer):
  tf.logging.info('Adding minibatch features stddev')

  _, stddev = tf.nn.moments(layer, [0])
  tf.logging.info('MBS layer shape: %s', layer.shape.as_list())
  tf.logging.info('MBS stddev shape: %s', stddev.shape.as_list())
  stddev_mean = tf.reduce_mean(stddev)
  batch_size = layer.shape.as_list()[0]
  stddev_feature = tf.zeros([batch_size, 1, 4, 4]) + stddev_mean
  layer = tf.concat([layer, stddev_feature], axis=1)
  tf.logging.info('MBS new layer shape %s:', layer.shape.as_list())
  tf.summary.scalar('minibatch_stddev_mean', stddev_mean)
  return layer


# Taken from Nvidia's progressive_growing_of_GANs.
def pixel_norm(x, axis, epsilon=1e-8, scope='pixel_norm'):
  with tf.variable_scope('scope'):
      return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis,
                                         keepdims=True) + epsilon)


def append_one_hot_to_tensor(tensor, one_hot):
  batch_size = tensor.shape.as_list()[0]

  #  tensor: [N][C][H][W] or [N][C]
  #  one_hot: [N][D]
  shape_len = len(tensor.shape)

  if shape_len == 2:
    return tf.concat([tensor, one_hot], axis=1)

  elif shape_len == 4:
    # [N][D][1][1]
    one_hot = tf.expand_dims(tf.expand_dims(one_hot, 2), 3)

    # [N][D][H][W]
    zeros = tf.zeros([batch_size,
                      one_hot.shape[1], tensor.shape[2], tensor.shape[3]])

    one_hot_cube = zeros + one_hot

    # [N][C+D][H][W]
    return tf.concat([tensor, one_hot_cube], axis=1)

  else:
    assert False, 'Unsupported shape_len: %s' % shape_len


def batch_norm(net, axis, scope='batch_norm'):
  with tf.variable_scope(scope):
    net = tf.layers.batch_normalization(
      inputs=net,
      axis=axis,
      training=True,  # See comment below.
      #  fused=False,
      fused=True,
    )
    # Super hacky. Otherwise training fails due to usage of tf.cond.
    # The model will not work when training=False.
    del tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)[-2:]
    return net


def batch_norm_in_place(net, axis, scope='batch_norm_in_place',
                        is_training=None):
  # Here we use updates_collections=None. This ensures updates to shadow vars
  # are performed in place. Otherwise, the dynamic training (with tf.conds)
  # fails to update the vars, and the model needs to be evaluated in training
  # mode after exporting.
  assert is_training is not None
  assert axis in [1, 3]
  data_format = 'NCHW' if axis == 1 else 'NHWC'
  with tf.variable_scope(scope):
    net = tf.contrib.layers.batch_norm(
      inputs=net,
      is_training=is_training,
      data_format=data_format,
      fused=True,
      updates_collections=None,
    )
    return net


def layer_norm(net, axis, scope='layer_norm'):
  assert False, ('layer norm only works for NHWC,'
                 'the implementation below is broken')
  with tf.variable_scope(scope):
    net = tf.contrib.layers.layer_norm(
      inputs=net,
      center=True,
      scale=True,
      activation_fn=None,
      reuse=False,
      begin_params_axis=axis,  # TODO: Not sure about this...
    )
    return net


#  def dense(net, scope='dense', **kwargs):
#    net.shape.assert_has_rank(2)
#    with tf.variable_scope(scope):
#      net = keras.layers.Dense(
#        **kwargs
#      )(net)
#      return net

def dense(inp, units=None, scope='linear', weight_norm=None):
  with tf.variable_scope(scope):
    filters_in = inp.shape.as_list()[1]
    W = get_weights(shape=[filters_in, units], weight_norm=weight_norm)
    b = tf.get_variable(name='biases', shape=[units],
                        initializer=tf.constant_initializer(0.00))
    return tf.matmul(inp, W) + b


#  def _conv(_keras_layer, net, scope='conv', kernel_size=3, strides=1, **kwargs):
#    with tf.variable_scope(scope):
#      net = _keras_layer(
#        padding='same',
#        data_format='channels_first',
#        kernel_size=kernel_size,
#        **kwargs
#      )(net)
#    return net


#  def conv(*args, **kwargs):
#    return _conv(keras.layers.Conv2D, *args, **kwargs)


#  def conv_trans(*args, **kwargs):
#    return _conv(keras.layers.Conv2DTranspose, *args, **kwargs)


def _prod(s):
  if s == []: return 1
  return s[0] * _prod(s[1:])


def get_weights(shape, weight_norm):
  if weight_norm == 'dynamic':
    assert False, 'something is broken here'
    W = tf.get_variable(name='weights', shape=shape,
                        initializer=tf.glorot_uniform_initializer())
    v = tf.get_variable(name='dynamic_weights_norm', shape=[],
                        initializer=tf.constant_initializer(1.))
    W = W / tf.norm(W) * v
  elif weight_norm == 'equalized':
    W = tf.get_variable(name='weights', shape=shape,
                        initializer=tf.random_normal_initializer())
    W = W * np.sqrt(2) / np.sqrt(_prod(shape[:-1]))
  else:
    assert weight_norm is None
    W = tf.get_variable(name='weights', shape=shape,
                        initializer=tf.glorot_uniform_initializer())
  return W


def conv(net, kernel_size=3, strides=1, filters=None, scope='conv',
         weight_norm=None):
  assert strides == 1
  with tf.variable_scope(scope):
    filters_in = net.shape.as_list()[1]
    W = get_weights(
      shape=[kernel_size, kernel_size, filters_in, filters],
      weight_norm=weight_norm)
    conv = tf.nn.conv2d(input=net,
                        filter=W,
                        strides=[1, 1, strides, strides],
                        padding='SAME',
                        data_format='NCHW')

    b = tf.get_variable(name='biases', shape=[filters],
                        initializer=tf.constant_initializer(0.00))
    #  b = b / 100.  # XXX
    conv = tf.nn.bias_add(conv, b, data_format='NCHW')
    return conv


def conv_trans(*args, **kwargs):
  # TODO: double check: is conv_trans the same as conv, when there's no stride?
  assert False


def _norm(net, axis, version, scope='norm', is_training=None):
  assert is_training is not None
  with tf.variable_scope(scope):
    if version is None:
      return net
    elif version == 'batch_norm':
      return batch_norm(net, axis=axis)
    elif version == 'layer_norm':
      return layer_norm(net, axis=axis)
    elif version == 'pixel_norm':
      return pixel_norm(net, axis=axis)
    elif version == 'batch_norm_in_place':
      return batch_norm_in_place(net, axis=axis, is_training=is_training)
    else:
      assert False, 'Unknown normalization: %s' % version


def norm(net, axis, version, scope='norm', is_training=None, gpu_id=None,
         per_gpu=None):
  assert gpu_id is not None
  assert per_gpu is not None
  if per_gpu:
    with tf.variable_scope('norm_gpu_%d' % gpu_id):
      with tf.device('/gpu:%d' % gpu_id):
        return _norm(net, axis, version, scope, is_training)
  else:
    return _norm(net, axis, version, scope, is_training)


def _get_shape(tensor):
  tensor_shape = tf.shape(tensor)
  static_tensor_shape = tf_tensor_util.constant_value(tensor_shape)
  return (static_tensor_shape if static_tensor_shape is not None else
          tensor_shape)


# Taken from tf.contrib.tfgan. Added the option to use activation on
# mapped_contitioning.
def condition_tensor(tensor, conditioning, act=None):
  tensor.shape[1:].assert_is_fully_defined()
  num_features = tensor.shape[1:].num_elements()

  mapped_conditioning = tf.contrib.layers.linear(
      tf.contrib.layers.flatten(conditioning), num_features)
  if act is not None:
    mapped_contitioning = act(mapped_conditioning)
  if not mapped_conditioning.shape.is_compatible_with(tensor.shape):
    mapped_conditioning = tf.reshape(
        mapped_conditioning, _get_shape(tensor))
  return tensor + mapped_conditioning
