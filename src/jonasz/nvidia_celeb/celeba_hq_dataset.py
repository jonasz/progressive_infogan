import tensorflow as tf
import os
import math
from jonasz import constants
from jonasz.lib import util
from jonasz.lib import tensor_util
import lmdb
import cv2
import numpy as np


class CelebAHQDatasetParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return dict(
      values_range   = (-1., 1.),
      img_side       = 128,
      data_dir       = None,
      train_shuffle  = True,
      gcs_bucket     = None,
      tfrecord_dir   = constants.NVIDIA_CELEBA_HQ_DATASET_PATH,
      random_flip    = False,
      crop_at_center = False,
      restrict_to_num_imgs = None,
      max_tfrecord_res_available = 10,
    )

  def validate(self):
    assert self.img_side in [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def get_dataset_params(
    is_gcloud=False,
    tfrecord_dir=constants.NVIDIA_CELEBA_HQ_DATASET_PATH_GCLOUD,
    **kwargs):
  if is_gcloud:
    return CelebAHQDatasetParams(
      gcs_bucket=constants.GCLOUD_BUCKET,
      tfrecord_dir=tfrecord_dir,
      **kwargs)
  else:
    return CelebAHQDatasetParams(**kwargs)


# Note: input_img_size may be different than params.img_size, when
# crop_at_center == True. (We want to downscale from a higher res image,
# rather than upscale the cropped part from same-size image.)
def process(img, params, input_img_size):
  img = tf.parse_single_example(
    img,
    {'data': tf.FixedLenFeature([], tf.string)}
  )['data']
  img = tf.decode_raw(img, tf.uint8)
  img = tf.cast(img, tf.float32)
  img = tf.reshape(img, [3, input_img_size, input_img_size])
  img = tensor_util.nchw_to_nhwc_single(img)
  img = img / 256.  # [0., 1.]
  left, right = params.values_range
  img = img * (right - left) + left
  if params.random_flip:
    img = tf.image.random_flip_left_right(img)
  if params.crop_at_center:
    img_side = params.img_side
    img = tf.image.resize_images([img], [input_img_size, input_img_size])[0]
    img = tf.image.crop_to_bounding_box(img,
                                        input_img_size/12,
                                        input_img_size/12,
                                        input_img_size*10/12,
                                        input_img_size*10/12)
    img = tf.image.resize_images([img], [img_side, img_side])[0]

  img.shape.assert_is_compatible_with([params.img_side, params.img_side, 3])
  return img


def get_train_input_fn(params, batch_size=128):
  def train_input_fn():
    log_img_side = int(math.log(params.img_side, 2))
    assert 2**log_img_side == params.img_side, str((log_img_side, params.img_side))

    tfrecord_id = log_img_side
    if params.crop_at_center:
      tfrecord_id += 1
      tfrecord_id = min(tfrecord_id, params.max_tfrecord_res_available)

    full_path = os.path.join(
      params.tfrecord_dir,
      'celeba_hq_tfrecord-r%02d.tfrecords' % tfrecord_id)
    tf.logging.info('Using dataset at %s', full_path)

    if params.gcs_bucket:
      full_path = os.path.join('gs://', params.gcs_bucket, full_path)

    d = tf.data.TFRecordDataset([full_path])
    if params.restrict_to_num_imgs is not None:
      d = d.take(params.restrict_to_num_imgs)
    d = d.map(lambda img: process(img, params, input_img_size=2**tfrecord_id),
              num_parallel_calls=16)
    d = d.repeat()
    if params.train_shuffle:
      d = d.shuffle(batch_size * 10)
    d = d.batch(batch_size)
    d = d.prefetch(buffer_size=64)
    iterator = d.make_one_shot_iterator()
    imgs = iterator.get_next()
    # TODO: Should do something else than return constant labels?
    return {'images': imgs}, tf.constant(1, shape=[batch_size, 1])
  return train_input_fn
