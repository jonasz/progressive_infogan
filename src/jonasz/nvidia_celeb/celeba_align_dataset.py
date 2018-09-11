import tensorflow as tf
import os
from jonasz import constants
from jonasz.lib import util
import lmdb
import cv2
import numpy as np


class CelebADatasetParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return dict(
      values_range  = (-1., 1.),
      img_side      = 128,
      data_dir      = None,
      train_shuffle = True,
      gcs_bucket    = None,
      imgs_path     = constants.NVIDIA_CELEBA_ALIGN_DATASET_PATH,
      random_flip   = False,
    )


def get_dataset(is_gcloud=False, **kwargs):
  if is_gcloud:
    return CelebADatasetParams(
      gcs_bucket=constants.GCLOUD_BUCKET,
      imgs_path=constants.NVIDIA_CELEBA_ALIGN_DATASET_PATH_GCLOUD,
      **kwargs)
  else:
    return CelebADatasetParams(**kwargs)


def open_img(path):
  with tf.gfile.Open(path, 'rb') as f:
    val = f.read()
    img_hwc = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
    return img_hwc


def img_generator(imgs_path, gcs_bucket=None):
  if gcs_bucket:
    imgs_path = os.path.join('gs://', gcs_bucket, imgs_path)
  tf.logging.info('img_generator tf.gfile.Glob start')
  paths = tf.gfile.Glob(os.path.join(imgs_path, '*'))
  tf.logging.info('img_generator tf.gfile.Glob done')
  for path in paths: yield open_img(path)


def process(img, params):
  img = tf.cast(img, tf.float32)
  img = img / 256.  # [0., 1.]
  left, right = params.values_range
  img = img * (right - left) + left
  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  c = tf.shape(img)[2]
  m = tf.minimum(h, w)
  # Crop to square.
  img = tf.image.crop_to_bounding_box(
    img, (h-m) / 2, (w - m) / 2, m, m)
  # Stretch.
  img = tf.image.resize_images(img, [params.img_side, params.img_side])
  # Not sure why - need to reverse the channels for imgs to show right colors.
  img = tf.concat([
    img[:,:,2:3],
    img[:,:,1:2],
    img[:,:,0:1],
  ], axis=2)
  if params.random_flip:
    img = tf.image.random_flip_left_right(img)
  return img


def get_train_input_fn(params, batch_size=128):
  def train_input_fn():
    d = tf.data.Dataset.from_generator(
      lambda: img_generator(params.imgs_path, gcs_bucket=params.gcs_bucket),
      tf.uint8)
    d = d.map(lambda img: process(img, params))
    d = d.repeat()
    if params.train_shuffle:
      d = d.shuffle(batch_size * 10)
    d = d.batch(batch_size)
    d = d.prefetch(buffer_size=8)
    iterator = d.make_one_shot_iterator()
    imgs = iterator.get_next()
    # TODO: Should do something else than return constant labels?
    return {'images': imgs}, tf.constant(1, shape=[batch_size, 1])
  return train_input_fn


if __name__ == '__main__':
  params = CelebADatasetParams(
    #  train_shuffle=False,
  )
  with tf.Session() as sess:
    features, labels = get_train_input_fn(params, 100)()
    imgs = features['images']
    cur_imgs = sess.run(imgs)
    util.show_imgs(list(cur_imgs)[:25])
