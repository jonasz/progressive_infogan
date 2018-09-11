from jonasz.lib import util
import tensorflow as tf
import numpy as np
from jonasz.nvidia_celeb import celeba_align_dataset
from jonasz.nvidia_celeb import celeba_hq_dataset
from jonasz.cifar10 import cifar10_dataset

class DummyDatasetParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return dict(
      values_range  = (-1., 1.),
      img_side      = 128,
      img_channels  = 3,
    )


def get_dummy_dataset_input_fn(params, batch_size):
  def train_input_fn():
    low, high = params.values_range
    shape = [params.img_channels, params.img_side, params.img_side]
    def gen():
      return (np.random.uniform(high=high, low=low, size=shape)
              for i in xrange(1000))
    d = tf.data.Dataset.from_generator(gen, tf.float32)
    d = d.repeat()
    d = d.batch(batch_size)
    d = d.prefetch(buffer_size=8)
    iterator = d.make_one_shot_iterator()
    imgs = iterator.get_next()
    return {'images': imgs}, tf.constant(1, shape=[batch_size, 1])
  return train_input_fn


def get_input_fn(dataset_params, batch_size):
  if type(dataset_params) is cifar10_dataset.DatasetParams:
    return cifar10_dataset.get_train_input_fn(params=dataset_params,
                                              batch_size=batch_size)
  elif (type(dataset_params) is
        celeba_align_dataset.CelebADatasetParams):
    return celeba_align_dataset.get_train_input_fn(params=dataset_params,
                                                   batch_size=batch_size)
  elif (type(dataset_params) is
        celeba_hq_dataset.CelebAHQDatasetParams):
    return celeba_hq_dataset.get_train_input_fn(params=dataset_params,
                                                batch_size=batch_size)
  elif (type(dataset_params) is DummyDatasetParams):
    return get_dummy_dataset_input_fn(params=dataset_params,
                                      batch_size=batch_size)
  else:
    raise RuntimeError, 'Unknown dataset_params: %s ' % type(dataset_params)
