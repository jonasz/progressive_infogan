import os
import collections
import copy
import multiprocessing
import random
import subprocess
import math
import cPickle
import time
import functools
import tensorflow as tf
import numpy as np
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.contrib.gan.python.eval.python import summaries as tfgan_summaries
from tensorflow.python.training import saver
from jonasz.progressive_infogan import networks
from jonasz.progressive_infogan import network_utils
from jonasz.progressive_infogan import progressive_infogan_losses
from jonasz.progressive_infogan import create_animation
from jonasz.progressive_infogan import export_utils
from jonasz.progressive_infogan import info_utils
from jonasz import constants

from jonasz.cifar10 import cifar10_dataset
from jonasz.gan import evaluation
from jonasz.lib import util
from jonasz.lib import datasets
from tensorflow.contrib import gan as tfgan


# By default runs on cifar10.
class TrainingParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return dict(
      _PHASE                      = None,

      is_gcloud                    = False,
      description                  = '',
      output_dir                   = None,
      continue_from                = None,  # Only works locally.
      continue_from_ckpt           = None,
      allow_initial_partial_restore = False,
      num_gpus                     = 1,
      vars_device                  = None,

      dynamic_steps_per_phase      = None,
      dynamic_batch_size           = None,
      stop_after                   = None,

      noise_size                   = 64,  # Exclusive of both cat and cont
                                          # structured variables. See infogan_*
                                          # params for these.
      noise_stddev                 = .4,
      old_generator                = False,
      generator_params             = None,
      old_discriminator            = False,
      discriminator_params         = None,
      dataset_params               = _default_dataset_params(),
      image_channels               = 3,
      eval_every_n_steps           = None,
      eval_every_n_secs            = 60*6,
      eval_at_start                = False,  # Eval at each phase's start.
      checkpoint_every_n_steps     = None,
      checkpoint_every_n_secs      = 60*10,
      write_summaries_every_n_steps = 100,
      infogan_summary_reps         = 3,
      eval_params                  = evaluation.EvalParams(
        inception_num_images=None,  # Disabled
        isolation_num_images=128,
        frechet_num_images = 5000,
        msssim_num_pairs = 2000,
      ),
      create_final_animation       = True,
      summary_grid_size            = 5,
      allow_simultaneous_steps     = True,  # Simultaneous steps are faster
                                            # (I checked) but require more
                                            # memory (I think).
      compute_gradients_gate       = None,  # None means default.

      use_gpu_tower_scope          = False,  # Whether to use the new scope.
                                             # Needs to be False for legacy
                                             # experiments to work.


      infogan_cont_weight   = None,
      infogan_cont_num_vars = None,
      infogan_cont_loss_phase_to_active_coords = None,

      infogan_cat_weight    = None,
      infogan_cat_num_vars  = None,
      infogan_cat_dim       = None,


      # If variable i has depth d, it means:
      # - Generator recieves var_i as input to block_d
      # - Discriminator outputs prediction for var_i based on output of block_d.
      # If None, all vars are assumed to be depth 2.
      infogan_cont_depth_to_vars = None,
      infogan_cont_depth_to_num_vars = None,  # Just a convenience utility.
      infogan_cont_unmask_depth_to_vars = None,
      infogan_cat_depth_to_vars  = None,
    )

  def validate(self):
    # Only validate after specialization.
    assert self.dynamic_steps_per_phase.keys() == self.dynamic_batch_size.keys()
    if isinstance(self.infogan_cont_weight, dict):
      assert self.dynamic_steps_per_phase.keys() == self.infogan_cont_weight.keys()
    target_side_log = max(self.dynamic_batch_size.keys())
    block_ids = range(2, target_side_log+1)

    min_batch_size = min(self.dynamic_batch_size.values())

    #  assert self.summary_grid_size**2 <= min_batch_size / self.num_gpus

    if self.infogan_cont_depth_to_vars is not None:
      assert sorted(self.infogan_cont_depth_to_vars.keys()) == block_ids
      present_vars = sum(self.infogan_cont_depth_to_vars.values(), [])
      assert (sorted(present_vars) == range(self.infogan_cont_num_vars)), (
        'Need to provide None or all vars.')
    if self.infogan_cat_depth_to_vars is not None:
      assert sorted(self.infogan_cat_depth_to_vars.keys()) == block_ids
      present_vars = sum(self.infogan_cat_depth_to_vars.values(), [])
      assert (sorted(present_vars) ==range(self.infogan_cat_num_vars)), (
        'Need to provide None or all vars.')

    if self.discriminator_params.gradient_penalty_weight:
      assert not (self.discriminator_params.norm_per_gpu or
                  self.generator_params.norm_per_gpu), (
        'Need to double check how per-gpu normalization should work ' +
        'with gradient penalty.')

    #  if self.num_gpus > 1:
    #    assert (self.discriminator_params.norm_per_gpu and
    #            self.generator_params.norm_per_gpu), (
    #      'Really want to run multi gpu training without norm_per_gpu?')

  @property
  def max_steps(self):
    res = 0
    assert self._PHASE is not None
    for phase, steps in self.dynamic_steps_per_phase.items():
      if phase <= self._PHASE:
        res += steps
    return res

  @property
  def batch_size(self):
    assert self._PHASE is not None
    return self.dynamic_batch_size[self._PHASE]


  @property
  def image_side(self):
    assert self._PHASE is not None
    return 2**self._PHASE


  @property
  def phase(self):
    assert self._PHASE is not None
    return self._PHASE

  @property
  def batch_size_per_gpu(self):
    return self.batch_size / self.num_gpus


  @property
  def infogan_cont_depth_to_vars(self):
    res = self.params['infogan_cont_depth_to_vars']
    if res is None:
      num_vars = self.infogan_cont_depth_to_num_vars
      res = {}
      first_unused = 0
      assert self.dynamic_steps_per_phase is not None
      for key in sorted(self.dynamic_steps_per_phase.keys()):
        cur_num_vars = num_vars.get(key, 0)
        res[key] = range(first_unused, first_unused+cur_num_vars)
        first_unused += cur_num_vars
    return res


  @property
  def infogan_cont_num_vars(self):
    res = self.params['infogan_cont_num_vars']
    if res is None:
      res = 0
      for vars_ in self.infogan_cont_depth_to_vars.values():
        if vars_: res = max(res, max(vars_)+1)
    return res

  @property
  def target_side_log(self):
    res = int(math.log(self.image_side, 2))
    assert self.image_side == 2**res
    return res

  @property
  def block_ids(self):
    return range(2, self.target_side_log+1)


def _specialize_training_params_for_phase(tp, phase):
  tp = copy.deepcopy(tp)
  tp.overwrite(
      _PHASE = phase,
  )
  # Not very pretty... TODO: Think how to make it more robust and isolated.
  tp.dataset_params.overwrite(
      img_side = 2**phase,
  )
  tp.validate()
  return tp


def _gan_inputs(training_params):
  train_input_fn = datasets.get_input_fn(training_params.dataset_params,
                                         training_params.batch_size)

  features, labels = train_input_fn()
  one_hot_labels = tf.one_hot(labels, depth=10)
  images = features['images']

  # This reshaping is needed so that the batch dimension is specified.
  side = training_params.image_side
  images = tf.reshape(images, [training_params.batch_size, side, side, 3])
  #  images = tf.image.resize_images(images, [8, 8])


  if training_params.infogan_cat_num_vars:
    structured_categorical_input = tf.random_uniform(
      [training_params.batch_size,
       training_params.infogan_cat_num_vars],
      minval=0, maxval=training_params.infogan_cat_dim, dtype=tf.int32)
  else:
    structured_categorical_input = None
  if training_params.infogan_cont_num_vars:
    structured_continuous_input = tf.random_normal(
      [training_params.batch_size,
       training_params.infogan_cont_num_vars],
      mean=0., stddev=training_params.noise_stddev)
  else:
    structured_continuous_input = None
  noise = tf.random_normal([training_params.batch_size,
                            training_params.noise_size],
                            mean=0., stddev=training_params.noise_stddev)

  generator_inputs = {
      'noise': noise,
      'structured_categorical_input': structured_categorical_input,
      'structured_continuous_input': structured_continuous_input,
      'one_hot_labels': one_hot_labels,
  }
  return generator_inputs, images


def _slice_gan_inputs(generator_inputs, images, num_gpus=1):
  generator_inputs_slices = [{} for i in range(num_gpus)]
  for key, val in generator_inputs.items():
    if val is None:
      for i in range(num_gpus):
        generator_inputs_slices[i][key] = None
    else:
      for i, slice_ in enumerate(tf.split(val, num_gpus)):
        generator_inputs_slices[i][key] = slice_
  images_slices = tf.split(images, num_gpus)
  return zip(generator_inputs_slices, images_slices)


def _get_phase(training_params, global_step):
  phase = training_params.phase
  steps_cur_phase = training_params.dynamic_steps_per_phase[phase]
  steps_prev_phases = training_params.max_steps - steps_cur_phase

  steps_passed_cur_phase = global_step - steps_prev_phases
  phase_progress = (tf.cast(steps_passed_cur_phase, tf.float32)
                    / float(steps_cur_phase) * 100.)
  return tf.constant(phase, dtype=tf.int64), phase_progress


# Training params -> gan model
def _gan_fn(training_params, is_training=True, return_dict=False, gpu_id=None):
  assert training_params.num_gpus == 1, '_gan_fn only supports 1 gpu for now.'
  global_step = tf.train.get_or_create_global_step()
  phase, phase_progress = _get_phase(training_params, global_step)
  generator_inputs, real_imgs = _gan_inputs(training_params)
  return _gan_fn_from_inputs(
    training_params=training_params,
    global_step=global_step,
    phase=phase,
    phase_progress=phase_progress,
    generator_inputs=generator_inputs,
    real_imgs=real_imgs,
    is_training=is_training,
    return_dict=return_dict,
    gpu_id=gpu_id)


def _gan_fn_from_inputs(training_params,
                        global_step, phase, phase_progress,
                        generator_inputs, real_imgs,
                        is_training=True, return_dict=False,
                        gpu_id=None):
  gen_fn = functools.partial(
    networks.generator_fn,
    training_params=training_params,
    phase=phase,
    phase_progress=phase_progress,
    is_training=is_training,
    gpu_id=gpu_id,
  )
  dis_fn = functools.partial(
    networks.discriminator_fn,
    training_params=training_params,
    phase=phase,
    phase_progress=phase_progress,
    is_training=is_training,
    gpu_id=gpu_id,
  )

  # Extracted from tfgan.gan_model, and later modified.
  with tf.variable_scope('Generator') as gen_scope:
    gen_dict = gen_fn(generator_inputs, return_dict=True)
    gen_imgs = gen_dict['img']
    gen_all_imgs = gen_dict['all_imgs']
  with tf.variable_scope('Discriminator') as dis_scope:
    disc_gen_dict = dis_fn(gen_imgs, generator_inputs, return_dict=True)
  with tf.variable_scope(dis_scope, reuse=True):
    disc_real_out = dis_fn(real_imgs, generator_inputs)

  if not gen_imgs.shape.is_compatible_with(real_imgs.shape):
    raise ValueError(
        'Generator output shape (%s) must be the same shape as real data '
        '(%s).' % (gen_imgs.shape, real_imgs.shape))
  gen_vars = tf.trainable_variables(gen_scope.name)
  disc_vars = tf.trainable_variables(dis_scope.name)

  if gpu_id == 0 or gpu_id is None:
    for i, img in enumerate(gen_all_imgs):
      block_id = 2+i
      grid_size = min(2, int(training_params.batch_size_per_gpu**.5))
      tf.summary.image('generated_image_block_%d' % block_id,
                       _make_grid(img, grid_size),
                       family='all_imgs')
      target_side_log = int(math.log(training_params.image_side, 2))
      downgrade_factor = 2**(target_side_log-block_id)
      downgraded_real_img = network_utils.downgrade2d(real_imgs, downgrade_factor,
                                                      data_format='NHWC')
      tf.summary.image('real_image_block_%d' % block_id,
                       _make_grid(downgraded_real_img, grid_size),
                       family='all_imgs')


  gan_model = tfgan.GANModel(
      generator_inputs           = generator_inputs,
      generated_data             = gen_imgs,
      generator_variables        = gen_vars,
      generator_scope            = gen_scope,
      generator_fn               = gen_fn,
      real_data                  = real_imgs,
      discriminator_real_outputs = disc_real_out,
      discriminator_gen_outputs  = disc_gen_dict['logits'],
      discriminator_variables    = disc_vars,
      discriminator_scope        = dis_scope,
      discriminator_fn           = dis_fn)

  if return_dict:
    return dict(
      gan_model=gan_model,
      phase=phase,
      phase_progress=phase_progress,
      predicted_distributions_loc=disc_gen_dict['predicted_distributions_loc'],
      predicted_categorical_softmax_list=disc_gen_dict[
        'predicted_categorical_softmax_list'],
      gen_all_imgs=gen_all_imgs,
    )
  else:
    return gan_model


def _should_do_eval(params, time_passed, steps_passed):
  if (params.eval_every_n_secs is not None
      and time_passed >= params.eval_every_n_secs):
    return True
  if (params.eval_every_n_steps is not None
      and steps_passed >= params.eval_every_n_steps):
    return True
  return False


def generator_params_ema(training_params, gan_model, gen_train_step):
  params_ema = tf.train.ExponentialMovingAverage(
    decay=training_params.generator_params.ema_decay_for_visualization)
  with tf.control_dependencies([gen_train_step]):
    variables = gan_model.generator_variables
    new_train_op = params_ema.apply(variables)
    for var in variables:
      tf.summary.histogram(var.op.name + '/ema',
                           params_ema.average(var),
                           family='generator_params_ema')
  return new_train_op


class GradsAccumulator(object):
  def __init__(self):
    self.vars_to_grads_list = collections.defaultdict(list)
    self.all_grads_healths = []

  def add(self, grads_and_vars):
    for grad, var in grads_and_vars:
      self.vars_to_grads_list[var].append(grad)


  def _summarize_grad_health(self, t, name):
    var = tf.Variable(0., name=name + '_cumulative_var')
    self.all_grads_healths.append(var)
    update_op = tf.assign_add(var, tf.cast(t, dtype=tf.float32))
    tf.summary.scalar(name + '_cumulative', var,
                      family='gradients_health')
    with tf.control_dependencies([update_op]):
      return tf.identity(t)

  def _check_gradient_health(self, grad, name):
    # is_finite returns False in case of NaNs and infs.
    grad_finite = tf.reduce_all(tf.is_finite(grad))
    grad_corrupted = tf.logical_not(grad_finite)
    grad_corrupted = self._summarize_grad_health(
      grad_corrupted, name=name + '_grad_corrupted')
    return tf.cond(
      grad_corrupted,
      lambda: tf.zeros_like(grad),
      lambda: grad,
    )

  def avg_grads_and_vars(self):
    res = []
    for var, raw_grads_list in self.vars_to_grads_list.items():
      grads_list = raw_grads_list[:]
      if None in grads_list:
        tf.logging.warning(
          'Some GPUs did not report grads for %s' % var.op.name)
        grads_list = filter(lambda x: x is not None, grads_list)
      if grads_list:
        deno = float(len(grads_list))
        avg_grad = tf.add_n(grads_list) / deno
        avg_grad = self._check_gradient_health(avg_grad, name=var.op.name)
        res.append((avg_grad, var))
      else:
        tf.logging.warning(
          'No gradients for %s' % var.op.name)
    tf.summary.scalar('overall_cumulative_gradients_health',
                      tf.reduce_sum(self.all_grads_healths),
                      family='overall_gradients_health')
    return res


# From tfgan.train
def _get_update_ops(gen_scope, dis_scope, check_for_unused_ops=True):
  update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
  all_gen_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS, gen_scope.name))
  all_dis_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS, dis_scope.name))

  if check_for_unused_ops:
    unused_ops = update_ops - all_gen_ops - all_dis_ops
    if unused_ops:
      raise ValueError('There are unused update ops: %s' % unused_ops)

  gen_update_ops = list(all_gen_ops & update_ops)
  dis_update_ops = list(all_dis_ops & update_ops)

  return gen_update_ops, dis_update_ops


# See: https://github.com/tensorflow/tensorflow/issues/9517
_PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
           'MutableHashTableOfTensors', 'MutableDenseHashTable']
def _assign_to_device(device, ps_device=None):
  def _assign(op):
    node_def = op if isinstance(op, tf.NodeDef) else op.node_def
    if node_def.op in _PS_OPS:
      return ps_device
    else:
      return device
  return _assign


def _get_cur_global_step(training_params):
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    with tf.Session(config=tf.ConfigProto(device_count={})) as sess:
      sess.run(tf.global_variables_initializer())
      export_utils.maybe_restore(sess, training_params,
                                 allow_partial_restore=False)
      cur_global_step, = sess.run([global_step])
  return cur_global_step


def get_vars_device(training_params):
  if training_params.vars_device:
    vars_device = training_params.vars_device
  elif training_params.num_gpus == 1:
    vars_device = '/gpu:0'
  else:
    vars_device = '/cpu:0'
  tf.logging.info('vars_device %s', vars_device)
  return vars_device


def _train_gan_multi_gpu(training_params):
  tf.logging.info("Starting / resuming training.")
  # All new training should use this.
  assert training_params.use_gpu_tower_scope
  assert not training_params.old_generator
  assert not training_params.old_discriminator

  vars_device = get_vars_device(training_params)

  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    phase, phase_progress = _get_phase(training_params, global_step)
    tf.summary.scalar('phase', phase)
    tf.summary.scalar('phase_progress', phase_progress)
    with tf.device(vars_device):
      generator_inputs, real_imgs = _gan_inputs(training_params)

    gen_optimizer = util.get_optimizer(training_params.generator_params,
                                       'generator')
    dis_optimizer = util.get_optimizer(training_params.discriminator_params,
                                       'discriminator')

    gen_grad_acu = GradsAccumulator()
    dis_grad_acu = GradsAccumulator()
    input_slices = _slice_gan_inputs(generator_inputs, real_imgs,
                                     num_gpus=training_params.num_gpus)

    # Construct towers, calculate per-tower grads.
    for i, (generator_inputs_slice, real_imgs_slice) in enumerate(input_slices):
      with tf.variable_scope('gpu_tower',
                             reuse=False if i==0 else tf.AUTO_REUSE):
        with tf.device(_assign_to_device('/gpu:%d' % i, ps_device=vars_device)):
          gpu_model_dict =  _gan_fn_from_inputs(
            training_params=training_params,
            global_step=global_step,
            phase=phase,
            phase_progress=phase_progress,
            generator_inputs=generator_inputs_slice,
            real_imgs=real_imgs_slice,
            is_training=True,
            return_dict=True,
            gpu_id=i)
          gpu_model = gpu_model_dict['gan_model']

          gpu_loss = progressive_infogan_losses.construct_gan_loss(
            training_params, gpu_model_dict)

          # gate_gradients=tf.train.Optimizer.GATE_NONE doesn't make training on
          # single gpu / 4 gpus any faster.
          compute_gradients_kwargs = {}
          if training_params.compute_gradients_gate == 'GATE_NONE':
            compute_gradients_kwargs['gate_gradients'] = (
              tf.train.Optimizer.GATE_NONE)
          else:
            assert training_params.compute_gradients_gate is None
          tf.logging.info('Graph: constructing G gradients')
          gen_gradients = gen_optimizer.compute_gradients(
            gpu_loss.generator_loss, var_list=gpu_model.generator_variables,
            **compute_gradients_kwargs)
          gen_grad_acu.add(gen_gradients)

          tf.logging.info('Graph: constructing D gradients')
          dis_gradients = dis_optimizer.compute_gradients(
            gpu_loss.discriminator_loss,
            var_list=gpu_model.discriminator_variables,
            **compute_gradients_kwargs)
          dis_grad_acu.add(dis_gradients)

          # Logging
          tf.logging.info('Graph: constructing summaries for gpu 0')
          if i == 0:
            grid_size = min(
              training_params.summary_grid_size,
              int(training_params.batch_size_per_gpu**.5))
            tfgan_summaries.add_gan_model_image_summaries(
              gpu_model, grid_size=grid_size, model_summaries=True)

            for var in gpu_model.generator_variables:
              tf.summary.histogram(var.op.name, var,
                                   family='generator_variables')
            for var in gpu_model.discriminator_variables:
              tf.summary.histogram(var.op.name, var,
                                   family='discriminator_variables')
            for grad, var in gen_gradients:
              tf.summary.histogram(var.op.name + '_grad', grad,
                                   family='generator_gradients')
            for grad, var in dis_gradients:
              tf.summary.histogram(var.op.name + '_grad', grad,
                                   family='discriminator_gradients')

    # NOTE: `gpu_model` from the last iteration will be used for various
    # purposes below.

    tf.logging.info('Graph: combining and applying gradients.')
    with tf.device(vars_device):
      gen_update_ops, dis_update_ops = _get_update_ops(
        gpu_model.generator_scope, gpu_model.discriminator_scope)
      with tf.control_dependencies(gen_update_ops):
        gen_train = gen_optimizer.apply_gradients(
          gen_grad_acu.avg_grads_and_vars())
      with tf.control_dependencies(dis_update_ops):
        dis_train = dis_optimizer.apply_gradients(
          dis_grad_acu.avg_grads_and_vars())

      gan_train_ops = tfgan.GANTrainOps(
        gen_train, dis_train, tf.assign(global_step, global_step+1))
    tf.logging.info('Graph: combining and applying gradients finished.')


    if training_params.generator_params.ema_decay_for_visualization:
      tf.logging.info('Enabling params exponential moving average for'
                      ' generator')

      gan_train_ops = tfgan.GANTrainOps(
        generator_params_ema(training_params, gpu_model,
                             gan_train_ops.generator_train_op),
        gan_train_ops.discriminator_train_op,
        gan_train_ops.global_step_inc_op,
      )

    session_config = tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement=True,
      #  intra_op_parallelism_threads = 16,
      #  inter_op_parallelism_threads = 16,
    )
    #  session_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    gen_steps = training_params.generator_params.train_steps
    disc_steps = training_params.discriminator_params.train_steps
    if training_params.allow_simultaneous_steps:
      both_steps = min(gen_steps, disc_steps)
    else:
      tf.logging.warning('Simultaneous training steps not allowed.')
      both_steps = 0
    disc_steps -= both_steps
    gen_steps -= both_steps
    tf.logging.info("both_steps %d, disc_steps %d, gen_steps %d",
                    both_steps, disc_steps, gen_steps)

    time_start = time.time()
    last_checkpoint_time = time.time()
    last_time_reported = time.time()

    # Note: reusing slice and model from last gpu.
    infogan_summary = info_utils.InfoGanSummary(
      training_params,
      generator_inputs_slice,
      gpu_model.generated_data,
      reps=training_params.infogan_summary_reps,
    )

    with tf.Session(config=session_config) as sess:

      tf.keras.backend.set_learning_phase(1)
      sess.run(tf.global_variables_initializer())

      export_utils.maybe_restore(
        sess, training_params,
        allow_partial_restore=training_params.allow_initial_partial_restore)

      tf.logging.info('Entering the training loop.')
      for i in xrange(int(1e9)):
        for j in range(both_steps):
          sess.run([gan_train_ops.discriminator_train_op,
                    gan_train_ops.generator_train_op])
        for j in range(disc_steps):
          sess.run(gan_train_ops.discriminator_train_op)
        for j in range(gen_steps):
          sess.run(gan_train_ops.generator_train_op)

        sess.run(gan_train_ops.global_step_inc_op)
        cur_global_step, cur_phase, cur_phase_progress = sess.run(
          [global_step, phase, phase_progress])

        if i % 10 == 0:
          time_passed = time.time() - last_time_reported
          last_time_reported = time.time()
          tf.logging.info("Step %5d (%.3fs), phase %d, phase_progress %2d, " +
                          "batch_size %d",
                          cur_global_step, time_passed, cur_phase,
                          cur_phase_progress, training_params.batch_size)

        if i % training_params.write_summaries_every_n_steps == 0:
          export_utils.write_summaries(
            training_params.output_dir, sess, cur_global_step,
            feed_dict=infogan_summary.construct_feed_dict(sess))

        if cur_global_step >= training_params.max_steps:
          tf.logging.info('Training finished')
          break

        secs_since_last_checkpoint = time.time() - last_checkpoint_time
        if export_utils.should_checkpoint(
            training_params, steps_passed=i+1,
            secs_passed=secs_since_last_checkpoint):
          export_utils.checkpoint(sess, cur_global_step, training_params)
          last_checkpoint_time = time.time()

        if _should_do_eval(training_params,
                           steps_passed=i+1,
                           time_passed=time.time()-time_start):
          tf.logging.info('Time for evaluation')
          break

      export_utils.checkpoint(sess, cur_global_step, training_params)


def _download_checkpoint(src_path, dst_path, num_checkpoint=None):
  if src_path.startswith('gs://'):
    cmd = 'gsutil ls %s' % os.path.join(src_path)
  else:
    cmd = 'find %s -maxdepth 1' % os.path.join(src_path)
  lines = subprocess.check_output(cmd.split(' ')).split('\n')
  lines = map(lambda x: x.strip(), lines)
  lines = filter(None, lines)
  lines = filter(lambda x: x.find('model.ckpt') != -1, lines)
  lines = sorted(lines)

  if num_checkpoint:
    lines = [line for line in lines
             if line.find('model.ckpt-%06d' % num_checkpoint) != -1]
    assert len(lines) == 3, lines
    tf.logging.info('Overwriting the checkpoint file to start from %d',
                    num_checkpoint)
    with tf.gfile.Open(os.path.join(dst_path, 'checkpoint'), 'w') as f:
      f.write(
        """
          model_checkpoint_path: "model.ckpt-%06d"
          all_model_checkpoint_paths: "model.ckpt-%06d"
        """ % (num_checkpoint, num_checkpoint))
  else:
    lines = lines[-3:]

  if src_path.startswith('gs://'):
    cmd = 'gsutil cp -n %s %s' % (' '.join(lines), dst_path)
  else:
    cmd = 'cp -n %s %s' % (' '.join(lines), dst_path)
  res = os.system(cmd)
  assert res == 0, res


def _single_gpu_training_params(training_params):
  new_tp = copy.deepcopy(training_params)
  new_tp.overwrite(
    batch_size=training_params.batch_size / training_params.num_gpus,
    num_gpus=1,
  )
  return new_tp


def _create_animation(training_params, saved_model_subdir, seeds):
  tf.logging.info('Creating animation for %s', saved_model_subdir)
  if (training_params.generator_params.norm == 'batch_norm' or
      training_params.discriminator_params.norm == 'batch_norm'):
    tf.logging.warning(
      'WARNING: animation doesnt work well with batch_norm')

  saved_model_path = os.path.join(training_params.output_dir,
                                  saved_model_subdir)
  export_path = os.path.join(training_params.output_dir,
                             saved_model_subdir + '_animation.mp4')
  rev_range = lambda x: list(reversed(range(x)))
  with util.TFModelServer(port=9997,
                          models={saved_model_path: []},
                          initial_sleep=10,  # Safe choice for local jobs
                          interruptible=True) as (_, (host, port)):
    create_animation.create_animation(
      saved_model_path=saved_model_path,
      noise_stddev=training_params.noise_stddev,
      cat_coords=rev_range(training_params.infogan_cat_num_vars or 0),
      cont_coords=rev_range(training_params.infogan_cont_num_vars or 0),
      grid_size=2,
      host_port=(host, port),
      batch_size=training_params.batch_size,
      export_path=export_path,
      embed_in_random_request=False,
      coord_resolution=32,
      frame_duration=50,
      numpy_random_seeds=seeds,
    )


def _evaluation(training_params):
  single_gpu_tp = _single_gpu_training_params(training_params)
  def gan_fn(is_training, return_dict=False):
    return _gan_fn(single_gpu_tp, is_training=is_training, gpu_id=0,
                   return_dict=return_dict)
  evaluation.evaluate(gan_fn, single_gpu_tp)


def _run_single_phase_training(training_params, phase):
  tf.logging.info('_run_single_phase_training, phase %d', phase)
  phase_start = time.time()
  training_params = _specialize_training_params_for_phase(
      training_params, phase)
  training_started = False
  while True:
    cur_global_step = _get_cur_global_step(training_params)
    tf.logging.info('cur_global_step %d, max steps %d. ',
                    cur_global_step, training_params.max_steps)
    if cur_global_step >= training_params.max_steps:
      break
    if phase >= 5 and training_params.eval_at_start and not training_started:
      _evaluation(training_params)

    _train_gan_multi_gpu(training_params)
    training_started = True

    if phase >= 5:
      _evaluation(training_params)
  phase_duration = int(time.time() - phase_start)
  tf.logging.info('Phase %d training took %d hours %d minutes in total.',
                  phase, phase_duration/60/60, (phase_duration/60)%60)


def _train_and_evaluate_gan(training_params):
  if training_params.continue_from:
    src = os.path.join(training_params.continue_from, 'checkpoint')
    dst = os.path.join(training_params.output_dir, 'checkpoint')
    if tf.gfile.Exists(dst):
      tf.logging.info('%s already exists, so not copying from %s', dst, src)
    else:
      tf.logging.info('Continuing training from: %s', src)
      print 'Continuing training from: %s' % src
      tf.gfile.Copy(src, dst, overwrite=False)
      _download_checkpoint(training_params.continue_from,
                           training_params.output_dir,
                           num_checkpoint=training_params.continue_from_ckpt)

  for phase in sorted(training_params.dynamic_steps_per_phase.keys()):
    if training_params.stop_after and phase > training_params.stop_after:
      tf.logging.info('Phase %d reached, stopping due to stop_after = %d',
                      phase, training_params.stop_after)
      break
    _run_single_phase_training(training_params, phase)


def _create_animations(training_params):
  if (training_params.create_final_animation
      and not training_params.is_gcloud):
    seeds = [random.randint(0, 1000000) for i in range(4)]
    _create_animation(training_params, 'saved_model', seeds=seeds)
    if training_params.generator_params.ema_decay_for_visualization:
      _create_animation(training_params, 'saved_model_ema', seeds=seeds)


@util.tf_logging_decorator
def run_training(training_params):
  tf.gfile.MakeDirs(training_params.output_dir)
  _train_and_evaluate_gan(training_params)
  #  _create_animations(training_params)


def _make_grid(imgs, grid_side):
  img_side = imgs.shape.as_list()[1]
  return tfgan.eval.eval_utils.image_grid(imgs[:grid_side*grid_side],
                                          grid_shape=(grid_side, grid_side),
                                          image_shape=(img_side, img_side))

def _flatten(net):
  shape = net.shape.as_list()
  def prod(x): return x[0]*prod(x[1:]) if x else 1
  return tf.reshape(net, shape=[shape[0] or -1, prod(shape[1:])])


def _default_dataset_params():
  return cifar10_dataset.DatasetParams(
        values_range                   = (-1., 1.),
        include_test_data_for_training = True,
        data_dir=constants.CIFAR10_DATA_DIR,
        data_format                    = 'NHWC',
        gcs_bucket                     = None,
        test_shuffle                   = True,
        train_brightness_max_delta     = 0.16200970439486795,
        train_can_flip                 = True,
        train_hue_max_delta            = 0.01611112590098971,
        train_max_crop_shift           = 0,
        train_random_contrast          = (0.8946658836783294, 1.0),
        train_shuffle                  = True,
        #  train_size_for_crop         = 40,
      )
