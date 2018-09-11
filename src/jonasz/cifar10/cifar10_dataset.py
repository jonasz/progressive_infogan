"""Utility class for loading cifar 10 data.

The "CIFAR-10 python version" data format described at
https://www.cs.toronto.edu/~kriz/cifar.html is assumed.
"""

import os.path
import pickle
import numpy as np
import tensorflow as tf
from google.cloud import storage as gcs
from jonasz import constants

from jonasz.lib import util

class DatasetParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return {
      'values_range': (0., 1.),
      'gcs_bucket': None,  # 'seer-of-visions-ml2',

      #  Older code uses data_dir. Newer code uses *tfrecord stuff.
      'data_dir': None,  # 'cifar-10-batches-py',
      'train_data_tfrecord': None,
      'test_data_tfrecord': None,

      'train_max_crop_shift': 0,
      'train_size_for_crop': None,
      'train_can_flip': False,
      'train_hue_max_delta': 0.0,
      'train_shuffle': False,
      'train_random_contrast': (1.-1e-5, 1.),
      'train_brightness_max_delta': 0.,
      'test_shuffle': True,
      'data_format': 'NCHW',
      'include_test_data_for_training': False,
    }

  def validate(self):
    if self.train_data_tfrecord:
      assert self.data_dir is None

    if self.data_dir:
      assert self.train_data_tfrecord is None
      assert self.test_data_tfrecord is None


def _load_from_gcs(gcs_bucket, data_dir, filename):
  if gcs_bucket is None:
    with open(os.path.join(data_dir, filename)) as f:
      data = f.read()
      return data
  client = gcs.Client()
  bucket = client.get_bucket(gcs_bucket)
  blob = bucket.get_blob(data_dir + '/' + filename)
  return blob.download_as_string()


def _load_dict(gcs_bucket, data_dir, filename):
  print 'loading', filename
  blob = _load_from_gcs(gcs_bucket, data_dir, filename)
  d_raw = pickle.loads(blob)
  d = {}
  d['labels'] = np.array(d_raw[b'labels'], dtype=np.int32)

  images = np.array(d_raw[b'data'])
  new_images = []
  for img in images:
    new_img = img.astype(np.float32) / 256.
    new_img = new_img.reshape(3, 32, 32)
    new_images.append(new_img)
  d['data'] = np.array(new_images)
  return d


def _load_train_data(gcs_bucket, data_dir):
  train_data_dict = None
  for i in range(5):
    cur_dict = _load_dict(gcs_bucket, data_dir, 'data_batch_%d' % (i+1))
    if train_data_dict is None:
      train_data_dict = cur_dict
    else:
      for key in train_data_dict.keys():
        train_data_dict[key] = np.concatenate((
          train_data_dict[key], cur_dict[key]))
  return train_data_dict


def _write_data_to_tfrecord(data_dict, path):
  data, labels = data_dict['data'], data_dict['labels']
  with tf.python_io.TFRecordWriter(path) as writer:
    for img, label in zip(data, labels):
      example = tf.train.Example()
      example.features.feature['images'].float_list.value.extend(img.reshape(-1))
      example.features.feature['labels'].int64_list.value.append(label)
      writer.write(example.SerializeToString())


def write_train_data_to_tfrecord(target_path, data_dir):
  train_data = _load_train_data(None, data_dir)
  _write_data_to_tfrecord(train_data, target_path)


def write_test_data_to_tfrecord(target_path, data_dir):
  test_data = _load_test_data(None, data_dir)
  _write_data_to_tfrecord(test_data, target_path)


def _load_test_data(gcs_bucket, data_dir):
  return _load_dict(gcs_bucket, data_dir, 'test_batch')


def _random_crop(img, params):
  size = params.train_size_for_crop
  if size is None:
    return img
  img = tf.image.resize_images(img, [size, size])
  max_offset = size - 32
  offset_height = tf.random_uniform([], minval=0, maxval=max_offset,
                                    dtype=tf.int32)
  offset_width = tf.random_uniform([], minval=0, maxval=max_offset,
                                   dtype=tf.int32)
  img = tf.image.crop_to_bounding_box(img, offset_height, offset_width,
                                      32, 32)
  return img



def _random_shift(img, params):
  max_shift = params.train_max_crop_shift
  if not max_shift:
    return img
  size = 32 + max_shift * 2
  img = tf.image.pad_to_bounding_box(img, max_shift, max_shift, size, size)
  offset_height = tf.random_uniform([], minval=0, maxval=max_shift*2+1,
                                    dtype=tf.int32)
  offset_width = tf.random_uniform([], minval=0, maxval=max_shift*2+1,
                                   dtype=tf.int32)
  img = tf.image.crop_to_bounding_box(img, offset_height, offset_width,
                                      32, 32)
  return img


def _image_noise(params):
  def f(img, label):
    img = tf.transpose(img, [1, 2, 0])  # CHW to HWC
    if params.train_can_flip:
      img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(
        img, max_delta=params.train_brightness_max_delta)
    img = tf.clip_by_value(img, 0., 1.)  # Why doesn't random_brightness do that?
    img = tf.image.random_hue(img, max_delta=params.train_hue_max_delta)
    img = tf.image.random_contrast(img, *params.train_random_contrast)
    img = _random_crop(img, params)
    img = _random_shift(img, params)
    if params.data_format == 'NCHW':
      img = tf.transpose(img, [2, 0, 1])  # HWC TO CHW
    return img, label
  return f


def _shift_pixel_values(img, params):
  left, right = params.values_range
  img = img * (right - left) + left
  return img


def _example_to_img_and_label(serialized_example):
  features = {
      'images': tf.FixedLenFeature([3, 32, 32], tf.float32),
      'labels': tf.FixedLenFeature([], tf.int64)
  }
  d = tf.parse_single_example(serialized_example, features)
  return d['images'], d['labels']



def get_train_input_fn(params, batch_size=128):
  # NOTE: When working with Dataset.from_tensor_slices, TensorFlow produces
  # ridiculously large log files. It seems that the input itself, represented as
  # a tf.constant, is serialized into the graph file, and dumped multiple times
  # into the events file as well. This hurts performance considerably.
  # For this reason the dataset is serialized into tfrecords, and we're working
  # with TFRecordDataset.

  if params.data_dir:
    train_path = os.path.join(params.data_dir, 'train_data.tfrecord')
    test_path = os.path.join(params.data_dir, 'test_data.tfrecord')
  else:
    train_path = params.train_data_tfrecord
    test_path = params.test_data_tfrecord
  if params.gcs_bucket:
    train_path = os.path.join('gs://', params.gcs_bucket, train_path)
    test_path = os.path.join('gs://', params.gcs_bucket, test_path)
  paths = [train_path]
  if params.include_test_data_for_training:
    paths += [test_path]
  def train_input_fn():
    d = tf.data.TFRecordDataset(paths)
    d = d.map(_example_to_img_and_label)
    d = d.repeat()
    if params.train_shuffle:
      d = d.shuffle(batch_size * 100)
    d = d.map(
        _image_noise(params),
        num_parallel_calls=8)
    d = d.map(
        lambda img, label: (_shift_pixel_values(img, params), label),
        num_parallel_calls=8)
    # TODO: this can likely be optimized better.
    d = d.batch(batch_size)
    d = d.prefetch(buffer_size=8)  # TODO: this means 8 batches, right?
    iterator = d.make_one_shot_iterator()
    imgs, labels = iterator.get_next()
    return {'images': imgs}, labels
  return train_input_fn


def get_test_input_fn(params, batch_size=128):
  assert params.data_format == 'NCHW'
  if params.test_data_tfrecord:
    path = params.test_data_tfrecord
  else:
    path = os.path.join(params.data_dir, 'test_data.tfrecord')

  if params.gcs_bucket:
    path = os.path.join('gs://', params.gcs_bucket, path)
  def test_input_fn():
    d = tf.data.TFRecordDataset(path)
    d = d.map(_example_to_img_and_label)
    d = d.repeat()
    if params.test_shuffle:
      d = d.shuffle(batch_size * 10)
    d = d.map(
        lambda img, label: (_shift_pixel_values(img, params), label),
        num_parallel_calls=8)
    d = d.batch(batch_size)
    iterator = d.make_one_shot_iterator()
    imgs, labels = iterator.get_next()
    return {'images': imgs}, labels
  return test_input_fn


def _get_class_name(label):
  return {
    0: 'plane',
    1: 'car',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'boat',
    9: 'truck',
  }[label]


def _test_params1(shuffle=True):
  return DatasetParams(
      gcs_bucket=None,
      data_dir=constants.CIFAR10_DATA_DIR,
      train_can_flip=True,
      train_hue_max_delta=0.01,
      train_shuffle=shuffle,
      train_random_contrast=(.85, 1.),
      train_brightness_max_delta=.05,
      train_size_for_crop=40,
      #  train_max_crop_shift=3,
  )


def _test_params2():
  return DatasetParams(
      gcs_bucket=None,
      data_dir=constants.CIFAR10_DATA_DIR)


def test1():
  print "Should show an identical frog seven times."
  params = _test_params2()
  test_imgs = []
  for i in range(7):
    with tf.Session(graph=tf.Graph()) as sess:
      features, labels = get_train_input_fn(params=params, batch_size=1)()
      imgs = features['images']
      cur_imgs, cur_labels = sess.run([imgs, labels])
      print ' '.join(map(_get_class_name, cur_labels))
      cur_imgs = list(cur_imgs)
      test_imgs.extend(cur_imgs)
  util.show_imgs(test_imgs, data_type='CHW')


def test2():
  print "Should show a frog seven times, each time reasonably modified."
  params = _test_params1(shuffle=False)
  test_imgs = []
  for i in range(7):
    with tf.Session(graph=tf.Graph()) as sess:
      features, labels = get_train_input_fn(params=params, batch_size=1)()
      imgs = features['images']
      cur_imgs, cur_labels = sess.run([imgs, labels])
      print ' '.join(map(_get_class_name, cur_labels))
      cur_imgs = list(cur_imgs)
      test_imgs.extend(cur_imgs)
  util.show_imgs(test_imgs, data_type='CHW')


def test3():
  print "Should always show first 7 images of the dataset, unchanged."
  params = _test_params2()
  with tf.Session(graph=tf.Graph()) as sess:
    features, labels = get_train_input_fn(params=params, batch_size=7)()
    imgs = features['images']
    cur_imgs, cur_labels = sess.run([imgs, labels])
    print ' '.join(map(_get_class_name, cur_labels))
    cur_imgs = list(cur_imgs)
    util.show_imgs(cur_imgs, data_type='CHW')


def test4():
  print "Should show 7 randomly chosen, reasonably modified images."
  params = _test_params1()
  with tf.Session(graph=tf.Graph()) as sess:
    features, labels = get_train_input_fn(params=params, batch_size=7)()
    imgs = features['images']
    cur_imgs, cur_labels = sess.run([imgs, labels])
    print ' '.join(map(_get_class_name, cur_labels))
    cur_imgs = list(cur_imgs)
    util.show_imgs(cur_imgs, data_type='CHW')


def test5():
  print "Should show 7 randomly chosen images from the test set."
  params = _test_params2()
  with tf.Session(graph=tf.Graph()) as sess:
    features, labels = get_test_input_fn(params=params, batch_size=7)()
    imgs = features['images']
    cur_imgs, cur_labels = sess.run([imgs, labels])
    print ' '.join(map(_get_class_name, cur_labels))
    cur_imgs = list(cur_imgs)
    util.show_imgs(cur_imgs, data_type='CHW')
