import tensorflow as tf
import numpy as np
from tensorflow.contrib import gan as tfgan
import copy


# Not very pretty but does the work...
class InfoGanSummary(object):
  def __init__(self, training_params, generator_inputs, sample_t, reps=3):
    self.training_params = training_params
    self.generator_inputs = generator_inputs
    self.sample_t = sample_t
    self.cont_placeholder_map = {}
    self.cat_placeholder_map = {}
    self.cont_grid_side = 5
    self.cat_grid_side = 1
    while self.cat_grid_side**2 < training_params.infogan_cat_dim:
      self.cat_grid_side += 1
    self.reps = reps

    if training_params.infogan_cont_num_vars:
      self._infogan_images_summary(
        num_coords=training_params.infogan_cont_num_vars,
        prefix='infogan_cont_',
        grid_side=self.cont_grid_side,
        placeholder_map=self.cont_placeholder_map)

    if training_params.infogan_cat_num_vars:
      self._infogan_images_summary(
        num_coords=training_params.infogan_cat_num_vars,
        prefix='infogan_cat_',
        grid_side=self.cat_grid_side,
        placeholder_map=self.cat_placeholder_map)


  def _feed_dict_from_gen_inputs(self, cur_generator_inputs):
    feed_dict = {}
    for key, val in cur_generator_inputs.items():
      if val is not None:
        feed_dict[self.generator_inputs[key]] = val
    return feed_dict


  def _construct_cont_infogan_images(self, sess, coord, cur_generator_inputs):
    all_imgs = []
    cur_generator_inputs = copy.deepcopy(cur_generator_inputs.copy())
    for i, val in enumerate(np.linspace(-1.25, 1.25, self.cont_grid_side**2)):
      # We substitute a single value in the first example of the batch.
      # We need to keep the rest of the batch unchanged, so interpolation works
      # well in presence of batch norm.
      cur_generator_inputs['structured_continuous_input'][0][coord] = val
      cur_imgs = sess.run(self.sample_t,
                          feed_dict=self._feed_dict_from_gen_inputs(
                            cur_generator_inputs))
      cur_img = cur_imgs[0]
      all_imgs.append(cur_img)
    return all_imgs


  def _construct_cat_infogan_images(self, sess, coord, cur_generator_inputs):
    all_imgs = []
    cur_generator_inputs = copy.deepcopy(cur_generator_inputs.copy())
    for val in range(self.training_params.infogan_cat_dim):
      # We substitute a single value in the first example of the batch.
      # We need to keep the rest of the batch unchanged, so interpolation works
      # well in presence of batch norm.
      cur_generator_inputs['structured_categorical_input'][0][coord] = val
      cur_imgs = sess.run(self.sample_t,
                          feed_dict=self._feed_dict_from_gen_inputs(
                            cur_generator_inputs))
      cur_img = cur_imgs[0]
      all_imgs.append(cur_img)

    black_img = np.full(
      [self.training_params.image_side, self.training_params.image_side, 3], 0.)
    all_imgs += [black_img] * ((self.cat_grid_side**2) - len(all_imgs))
    return all_imgs


  def _fetch_generator_inputs(self, sess, generator_inputs):
    result = {}
    to_fetch = {}
    for key, val in generator_inputs.items():
      if val is None:
        result[key] = val
      else:
        to_fetch[key] = val

    keys, vals = zip(*to_fetch.items())
    fetched = sess.run(vals)
    for key, val in zip(keys, fetched):
      result[key] = val
    return result


  def construct_feed_dict(self, sess):
    feed_dict = {}
    generator_inputs_per_rep = [
      self._fetch_generator_inputs(sess, self.generator_inputs)
      for rep in range(self.reps)
    ]
    for (coord, rep), placeholder in self.cont_placeholder_map.items():
      generator_inputs = generator_inputs_per_rep[rep].copy()
      imgs = self._construct_cont_infogan_images(sess, coord, generator_inputs)
      feed_dict[placeholder] = imgs
    for (coord, rep), placeholder in self.cat_placeholder_map.items():
      generator_inputs = generator_inputs_per_rep[rep].copy()
      imgs = self._construct_cat_infogan_images(sess, coord, generator_inputs)
      feed_dict[placeholder] = imgs
    return feed_dict

  def _infogan_images_summary(self, num_coords, prefix, grid_side,
                              placeholder_map):
    img_side = self.training_params.image_side
    for coord in range(num_coords):
      for rep in range(self.reps):
        imgs = tf.placeholder(dtype=tf.float32,
                              shape=[grid_side**2, img_side, img_side, 3])
        placeholder_map[(coord, rep)] = imgs
        tf.summary.image(
          prefix + 'coord_%d_rep_%d' % (coord, rep),
          tfgan.eval.eval_utils.image_grid(imgs[:grid_side*grid_side],
                                           grid_shape=(grid_side, grid_side),
                                           image_shape=(img_side, img_side)),
          family=prefix + 'interpolation')
        tf.summary.image(
          prefix + 'coord_%d_rep_%d_diff' % (coord, rep),
          tfgan.eval.eval_utils.image_grid(imgs[:grid_side*grid_side]-imgs[0:1],
                                           grid_shape=(grid_side, grid_side),
                                           image_shape=(img_side, img_side)),
          family=prefix + 'diffs')
