import os
import math
import time
import functools
import tensorflow as tf
import numpy as np
from jonasz.progressive_infogan import network_utils
from jonasz import constants

from jonasz.lib import util
from tensorflow.contrib import gan as tfgan


def _flatten(net):
  shape = net.shape.as_list()
  def prod(x): return x[0]*prod(x[1:]) if x else 1
  return tf.reshape(net, shape=[shape[0] or -1, prod(shape[1:])])


def _vanilla_consistency_loss(cur_rgb, prev_rgb, block_id, tp):
  if not tp.generator_params.consistency_loss:
    return 0.
  cur_rgb_down = network_utils.downscale2d(
    cur_rgb, 2**(tp.target_side_log - block_id + 1), data_format='NHWC')
  prev_rgb_down = network_utils.downscale2d(
    prev_rgb, 2**(tp.target_side_log - block_id + 1), data_format='NHWC')
  cur_loss = tf.losses.mean_squared_error(
    _flatten(cur_rgb_down),
    _flatten(prev_rgb_down))
  cur_loss *= tp.generator_params.consistency_loss
  return cur_loss


def _msssim256_consistency_loss(cur_rgb, prev_rgb, block_id, tp):
  if not tp.generator_params.consistency_loss_msssim:
    return 0.
  # First: scale down to the same resolution.
  cur_rgb = network_utils.downscale2d(
    cur_rgb, 2**(tp.target_side_log - block_id + 1), data_format='NHWC')
  prev_rgb = network_utils.downscale2d(
    prev_rgb, 2**(tp.target_side_log - block_id + 1), data_format='NHWC')

  # Second: scale up to 256x256 so that msssim works well.
  cur_rgb = tf.image.resize_images(
    cur_rgb, size=[256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  prev_rgb = tf.image.resize_images(
    prev_rgb, size=[256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  sim = tf.image.ssim_multiscale(cur_rgb + 1., prev_rgb + 1., 2.)
  cur_loss = 1.-tf.reduce_mean(sim)
  cur_loss *= tp.generator_params.consistency_loss_msssim
  return cur_loss



def _add_consistency_loss(gan_loss, gan_model_dict, training_params):
  consistency_loss = tf.constant(0., shape=[])
  target_side_log = training_params.target_side_log
  for block_id in range(3, target_side_log+1):
    cur_rgb = gan_model_dict['gen_all_imgs'][block_id-2]
    prev_rgb = gan_model_dict['gen_all_imgs'][block_id-2-1]
    if not (training_params.generator_params.consistency_loss_at_blocks is None or
        block_id in training_params.generator_params.consistency_loss_at_blocks):
      continue
    tf.logging.info('Constructing consistency loss at block %d', block_id)
    if training_params.generator_params.consistency_loss:
      cur_loss = _vanilla_consistency_loss(cur_rgb, prev_rgb,
                                           block_id, training_params)
    elif training_params.generator_params.consistency_loss_msssim:
      cur_loss = _msssim256_consistency_loss(cur_rgb, prev_rgb,
                                             block_id, training_params)
    else:
      assert False

    if training_params.generator_params.consistency_temporal:
      # Only in relevant phase
      cur_loss = tf.cond(
        tf.equal(gan_model_dict['phase'], block_id),
        lambda: cur_loss,
        lambda: tf.constant(0.))
      # Only during transition (phase_progress <= 50).
      cur_loss = tf.cond(
        tf.less_equal(gan_model_dict['phase_progress'], 50.),
        lambda: cur_loss,
        lambda: tf.constant(0.))
    else:
      # Only in and after relevant phase.
      cur_loss = tf.cond(
        tf.greater_equal(gan_model_dict['phase'], block_id),
        lambda: cur_loss,
        lambda: tf.constant(0.))
    tf.summary.scalar('consistency_loss_%d' % block_id, cur_loss,
                      family='consistency_loss')
    consistency_loss += cur_loss

  tf.summary.scalar('consistency_loss', consistency_loss,
                    family='consistency_loss')

  return tfgan.GANLoss(
    gan_loss.generator_loss + consistency_loss,
    gan_loss.discriminator_loss,
  )


# TODO: Refactor cont and cat versions into unified logic.
def _add_categorical_mutual_information_penalty(gan_loss,
                                                gan_model_dict,
                                                training_params):
  gpu_batch_size = (training_params.batch_size /
                    training_params.num_gpus)
  gan_model = gan_model_dict['gan_model']
  mutual_information_penalties = []

  for i in range(training_params.infogan_cat_num_vars):
    dim = training_params.infogan_cat_dim
    variable = gan_model.generator_inputs['structured_categorical_input'][:,i]
    variable.shape.assert_is_compatible_with([gpu_batch_size])
    probs=gan_model_dict['predicted_categorical_softmax_list'][i]
    # TODO: why does validate_args lead to failure? It seems the probs sum
    # to 1 + eps - is this normal?
    pre = tf.distributions.Categorical(probs=probs, validate_args=False)
    cur_penalty = tfgan.losses.wargs.mutual_information_penalty(
      structured_generator_inputs=[variable],
      predicted_distributions=[pre],
      add_summaries=False,
    )
    cur_penalty *= training_params.infogan_cat_weight

    tf.summary.scalar('infogan_cat_penalty_coord_%d' % i, cur_penalty,
                      family='infogan_cat_penalty')
    mutual_information_penalties.append(cur_penalty)

  mutual_information_penalty = tf.reduce_mean(mutual_information_penalties)
  tf.summary.scalar('infogan_cat_penalty',
                    mutual_information_penalty,
                    family='infogan_cat_penalty')
  return tfgan.GANLoss(
     gan_loss.generator_loss + mutual_information_penalty,
     gan_loss.discriminator_loss + mutual_information_penalty,
   )


# TODO: Use this in _add_continuous_mutual_information_penalty. Right now, it's
# only used by eval code.
def unweighted_mutual_information_penalty_per_coord(gan_model_dict,
                                                    training_params,
                                                    coord):
  tp = training_params
  gan_model = gan_model_dict['gan_model']
  var = tf.slice(gan_model.generator_inputs['structured_continuous_input'],
                 [0, coord], [tp.batch_size_per_gpu, 1])
  loc = tf.slice(gan_model_dict['predicted_distributions_loc'],
                 [0, coord], [tp.batch_size_per_gpu, 1])
  pre = tf.distributions.Normal(loc=loc, scale=tf.ones_like(loc))
  var.shape.assert_is_compatible_with(loc.shape)
  abs_error = tf.abs(var-loc)
  abs_error.shape.assert_is_compatible_with([None, 1])
  abs_error_scalar = tf.reduce_mean(abs_error)
  cur_penalty = tfgan.losses.wargs.mutual_information_penalty(
    structured_generator_inputs=[var],
    predicted_distributions=[pre],
    add_summaries=False,
  )
  cur_penalty.shape.assert_is_compatible_with([])
  abs_error_scalar.shape.assert_is_compatible_with([])
  return cur_penalty, abs_error_scalar


def _add_continuous_mutual_information_penalty(gan_loss,
                                               gan_model_dict,
                                               training_params):
  gpu_batch_size = (training_params.batch_size /
                    training_params.num_gpus)
  gan_model = gan_model_dict['gan_model']
  assert training_params.infogan_cont_weight is not None
  mutual_information_penalties = []

  abs_errors = []
  for i in range(training_params.infogan_cont_num_vars):
    if training_params.infogan_cont_loss_phase_to_active_coords:
      include = False
      for (phase, coords) in training_params.infogan_cont_loss_phase_to_active_coords.items():
        if i in coords and phase <= training_params.phase:
          include = True
      if not include:
        tf.logging.info('Excluding coord %s from mutual information penalty '
                        'loss, phase %s', i, training_params.phase)
        continue
      else:
        tf.logging.info('Including coord %s in mutual information penalty '
                        'loss, phase %s', i, training_params.phase)

    var = tf.slice(gan_model.generator_inputs['structured_continuous_input'],
                   [0, i], [gpu_batch_size, 1])
    loc = tf.slice(gan_model_dict['predicted_distributions_loc'],
                   [0, i], [gpu_batch_size, 1])
    pre = tf.distributions.Normal(loc=loc, scale=tf.ones_like(loc))
    var.shape.assert_is_compatible_with(loc.shape)
    abs_error = tf.abs(var-loc)
    abs_error.shape.assert_is_compatible_with([None, 1])
    abs_error_scalar = tf.reduce_mean(abs_error)
    abs_errors.append(abs_error_scalar)
    cur_penalty = tfgan.losses.wargs.mutual_information_penalty(
      structured_generator_inputs=[var],
      predicted_distributions=[pre],
      add_summaries=False,
    )
    cur_penalty.shape.assert_is_compatible_with([])
    cur_weight = training_params.infogan_cont_weight
    if isinstance(cur_weight, dict):
      cur_weight = cur_weight[training_params.phase]
    cur_penalty *= cur_weight

    tf.summary.scalar('infogan_cont_penalty_coord_%d' % i, cur_penalty,
                      family='infogan_cont_penalty')
    tf.summary.histogram('infogan_cont_abs_error_coord_%d' % i, abs_error,
                         family='infogan_cont_abs_error')
    tf.summary.scalar('infogan_cont_mean_abs_error_coord_%d' % i,
                      tf.reduce_mean(abs_error_scalar),
                      family='infogan_cont_abs_error')

    mutual_information_penalties.append(cur_penalty)

  tf.summary.scalar('infogan_cont_mean_abs_error_mean',
                    tf.add_n(abs_errors) / float(len(abs_errors)),
                    family='infogan_cont_abs_error')

  mutual_information_penalty = tf.reduce_mean(mutual_information_penalties)
  tf.summary.scalar('infogan_cont_penalty',
                    mutual_information_penalty,
                    family='infogan_cont_penalty')
  return tfgan.GANLoss(
     gan_loss.generator_loss + mutual_information_penalty,
     gan_loss.discriminator_loss + mutual_information_penalty,
   )


def _add_drift_loss(gan_loss, gan_model_dict, training_params):
  gan_model = gan_model_dict['gan_model']
  assert training_params.discriminator_params.eps_drift
  drift_loss = tf.reduce_mean(
    training_params.discriminator_params.eps_drift
    * tf.square(gan_model.discriminator_real_outputs))
  tf.summary.scalar('drift_loss', drift_loss)
  return tfgan.GANLoss(
    gan_loss.generator_loss,
    gan_loss.discriminator_loss + drift_loss
  )


def construct_gan_loss(training_params, gan_model_dict):
  gan_model = gan_model_dict['gan_model']
  gan_loss = tfgan.gan_loss(
      gan_model,
      generator_loss_fn=eval(training_params.generator_params.loss_fn),
      discriminator_loss_fn=eval(
        training_params.discriminator_params.loss_fn),
      gradient_penalty_weight=(
        training_params.discriminator_params.gradient_penalty_weight),
      add_summaries=True,
  )

  if training_params.discriminator_params.eps_drift:
    gan_loss = _add_drift_loss(gan_loss, gan_model_dict, training_params)

  if (training_params.generator_params.consistency_loss
      or training_params.generator_params.consistency_loss_msssim):
    gan_loss = _add_consistency_loss(gan_loss, gan_model_dict, training_params)

  if training_params.infogan_cont_weight is not None:
    gan_loss = _add_continuous_mutual_information_penalty(
      gan_loss, gan_model_dict, training_params)

  if training_params.infogan_cat_weight is not None:
    gan_loss = _add_categorical_mutual_information_penalty(
      gan_loss, gan_model_dict, training_params)

  return gan_loss
