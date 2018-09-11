import tensorflow as tf


def nchw_to_nhwc(images):
  return tf.transpose(images, [0, 2, 3, 1])


def nhwc_to_nchw(images):
  return tf.transpose(images, [0, 3, 1, 2])


def nchw_to_nhwc_single(images):
  return tf.transpose(images, [1, 2, 0])


def nhwc_to_nchw_single(images):
  return tf.transpose(images, [2, 0, 1])
