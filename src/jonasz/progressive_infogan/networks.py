# TODO: try out weight normalization as an alternative to batch norm.

import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.gan.python.eval.python import summaries as tfgan_summaries
import math
from tensorflow.contrib import gan as tfgan
from jonasz.progressive_infogan import network_utils
import collections
from jonasz.lib import util
from jonasz.lib import tensor_util

class GeneratorParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return dict(
      debug_mode                = False,
      channels_at_4x4           = 256,
      channels_max              = int(1e9),
      channels_min              = 1,
      optimizer                 = ('rmsprop', 0.00007),
      learning_rate_decay_steps = None,
      learning_rate_decay_rate  = None,
      double_conv               = False,
      norm                      = 'batch_norm',  # TODO: experiment
      norm_per_gpu              = False,
      pn_after_act              = False,
      residual                  = False,  # TODO: experiment
      loss_fn                   = 'tfgan.losses.modified_generator_loss',
      train_steps               = 1,
      conditioning              = True,
      condition_with            = 'one_hot_labels',
      restrict_conditioning_to_blocks = None,
      weight_norm               = None,
      torgb_tanh                = True,
      ema_decay_for_visualization = None,
      normalize_latents         = False,
      consistency_loss          = None,
      consistency_loss_msssim   = None,
      consistency_loss_at_blocks = None,
      # Useful to gracefully take over from previous phase, but without
      # maintaining consistency forever.
      consistency_temporal      = False,
      kernel_size               = 3,
      prev_rgb_stop_gradient    = False,
      infogan_input_method = 'append',
      append_channels_div       = None,

      # TODO
      #  per_phase_unmask_structured_variables = None,
      #  unmask_gradually = False,
      #  per_block_structured_input = None,
      incubate                  = False,
    )

  def validate(self):
    assert self.consistency_loss is None or self.consistency_loss_msssim is None



class DiscriminatorParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return dict(
      debug_mode                = False,
      channels_at_2x2           = 512,
      channels_max              = int(1e9),
      channels_min              = 1,
      second_conv_channels_x2   = False,
      optimizer                 = ('rmsprop', 0.0007),
      learning_rate_decay_steps = None,
      learning_rate_decay_rate  = None,
      double_conv               = False,
      norm                      = 'batch_norm',  # TODO: experiment
      norm_per_gpu              = False,
      pn_after_act              = False,
      residual                  = False,  # TODO: experiment
      loss_fn                   = 'tfgan.losses.modified_discriminator_loss',
      gradient_penalty_weight   = None,
      eps_drift                 = None,
      train_steps               = 1,
      conditioning              = True,
      weight_norm               = None,
      kernel_size               = 3,
      fromrgb_use_n_img_diffs   = None,
      elastic_block_input       = False,
      image_noise_stddev        = None,

      infogan_max_pool          = False,
      infogan_all_vars_in_block2 = False,
    )

  def validate(self):
    pass


def phase_progress_to_alpha(block, phase, progress):
  assert block > 2
  progress = tf.cast(progress, tf.float32)  # Not sure if actually needed.

  # First stays at zero,
  # when block == phase, rises to 1. through half of the progress
  # then stays at 1.
  return tf.cond(tf.less(phase, block),
          lambda: 0.,
          lambda: tf.cond(tf.greater(phase, block),
                          lambda: 1.,
                          lambda: tf.minimum(progress * 2., 100) / 100.
                         )
         )


def generator_fn(generator_inputs,
                 training_params,
                 is_training,
                 phase,
                 phase_progress,
                 return_dict=False,
                 gpu_id=0):
  """
    Args:
      generator_inputs: dict of tensors
      training_params: (TrainingParams)
      phase: (Tensor) 0-d integer, identifies which block is being trained
          at each moment. Iterates through [2, side_log].
      phase_progress: (Tensor) 0-d integer, identifies how far into the current
          phase we are. Iterates through [0, 100).

    Returns:
      A tensor with generated images, shape NCHW.
  """
  params = training_params.generator_params
  img_side = training_params.image_side
  tf.logging.info('generator_fn, reuse %s, gpu_id %s',
                  tf.get_variable_scope().reuse, gpu_id)
  noise = generator_inputs['noise']
  noise.shape.assert_has_rank(2)  # [batch_dim, latent_size]
  tf.summary.histogram('noise', noise)
  gpu_batch_size = get_gpu_batch_size(noise, training_params)

  block_id_to_structured_vars = _block_id_to_structured_vars(
    generator_inputs, training_params, phase, phase_progress)

  one_hot_labels = generator_inputs['one_hot_labels']
  target_side_log = int(math.log(img_side, 2))
  assert 2**target_side_log == img_side

  def channels_at_res(res):
    channels = params.channels_at_4x4 / 2**(res-2)
    channels = min(channels, params.channels_max)
    channels = max(channels, params.channels_min)
    assert channels >= 16, 'res %d, channels %d' % (res, channels)
    return channels

  def _append_channels(net, infogan_vars, infogan_num_vars, block_id):
    if infogan_vars is None:
      return net
    with tf.variable_scope('append_channels'):
      if net.shape.ndims == 2:
        net = tf.concat([net, infogan_vars], axis=1)
      else:
        infogan_channels = tf.reshape(
          infogan_vars, [gpu_batch_size, infogan_num_vars, 1, 1])
        infogan_num_channels = (net.shape.as_list()[1]
                                / params.append_channels_div)
        tf.logging.info('Block %d, infogan_input_method append_channels, '
                        'infogan_num_channels %d', block_id, infogan_num_channels)
        infogan_channels = network_utils.conv(
          infogan_channels,
          filters=infogan_num_channels,
          kernel_size=1,
          weight_norm=params.weight_norm)
        infogan_channels = tf.nn.leaky_relu(infogan_channels)
        side = net.shape.as_list()[-1]
        infogan_channels = tf.tile(infogan_channels, [1, 1, side, side])
        net = tf.concat([net, infogan_channels], axis=1)
        tf.logging.info('Block %d, infogan_input_method append_channels, '
                        'new net shape %s', block_id,
                        net.shape.as_list())
      return net

  def condition(tensor, block_id):
    if (params.restrict_conditioning_to_blocks is not None and
        block_id not in params.restrict_conditioning_to_blocks):
      return tensor
    if params.conditioning is False:
      return tensor
    if params.condition_with == 'one_hot_labels':
      conditioning_tensor = one_hot_labels
    elif params.condition_with == 'structured_input':
      conditioning_tensor = structured_input
    else:
      assert False, 'Unknown params.condition_with: %s' % params.condition_with
    assert conditioning_tensor is not None
    return network_utils.append_one_hot_to_tensor(tensor, conditioning_tensor)

  def torgb(net, scope='torgb', block_id=None):
    with tf.variable_scope(scope):
      net = condition(net, block_id)
      net = network_utils.conv(net, filters=3, kernel_size=1,
                               weight_norm=params.weight_norm)
      if params.torgb_tanh:
        net = tf.nn.tanh(net)
      return net

  # CCNA - conditino, conv, norm, activation
  def CCNA(net, block_id, scope):
    with tf.variable_scope(scope):
      net = condition(net, block_id)
      net = network_utils.conv(net, filters=channels_at_res(block_id),
                               weight_norm=params.weight_norm,
                               kernel_size=params.kernel_size)
      net = network_utils.norm(net, axis=1, version=params.norm,
                               is_training=is_training, gpu_id=gpu_id,
                               per_gpu=params.norm_per_gpu)
      net = tf.nn.leaky_relu(net)
      if params.pn_after_act: net = network_utils.pixel_norm(net, axis=1)
      return net

  def infogan_input_hook(net, infogan_vars, infogan_num_vars, block_id):
    if (params.infogan_input_method in ['custom03', 'custom03v2']
        and infogan_vars is not None):
      shape1 = net.shape.as_list()
      net = tfgan.features.condition_tensor(net, infogan_vars)
      if params.infogan_input_method == 'custom03v2':
        prev_net = _append_channels(net, infogan_vars, infogan_num_vars,
                                    block_id)
      shape2 = net.shape.as_list()
      tf.logging.info('Infogan_input_hook triggered (%s), net shape %s -> %s',
                      params.infogan_input_method, shape1, shape2)
    if params.infogan_input_method == 'custom04' and infogan_vars is not None:
      shape1 = net.shape.as_list()
      net = network_utils.condition_tensor(net, infogan_vars, act=tf.nn.relu)
      shape2 = net.shape.as_list()
      tf.logging.info('Infogan_input_hook triggered (%s), net shape %s -> %s',
                      params.infogan_input_method, shape1, shape2)
    return net

  def inject_infogan_vars(block_id, prev_net, infogan_vars, infogan_num_vars):
    if params.infogan_input_method == 'append':
      for var in block_id_to_structured_vars[block_id]:
        # TODO: what we're appending is not one hot, but shouldn't be a problem.
        # Rename `append_one_hot_to_tensor`.
        tf.logging.info('InfoGAN: appending var %s to input of G.block_%d',
                        var.op.name, block_id)
        prev_net = network_utils.append_one_hot_to_tensor(prev_net, var)
    elif params.infogan_input_method == 'tfgan_condition':
      tf.logging.info('InfoGAN: conditioning with vars, shape %s, the input of '
                      'G.block_%d', infogan_vars.shape.as_list(), block_id)
      prev_net = tfgan.features.condition_tensor(prev_net, infogan_vars)
    elif params.infogan_input_method == 'custom01':
      tf.logging.info('InfoGAN: custom01 input with vars %s, '
                      'G.block_%d', infogan_vars.shape.as_list(), block_id)
      with tf.variable_scope('infogan_input_custom01'):
        with tf.variable_scope('mul'):
          prev_net_weight = tfgan.features.condition_tensor(
            tf.zeros_like(prev_net), infogan_vars)
          prev_net_weight = tf.sigmoid(prev_net_weight)
          prev_net = prev_net * prev_net_weight
        with tf.variable_scope('add'):
          prev_net = tfgan.features.condition_tensor(prev_net, infogan_vars)
    elif params.infogan_input_method == 'custom02':
      tf.logging.info('InfoGAN: custom02 input with vars %s, '
                      'G.block_%d', infogan_vars.shape.as_list(), block_id)
      with tf.variable_scope('infogan_input_custom02'):
        units_base = len(block_id_to_structured_vars[block_id])
        infogan_vars_subnet = infogan_vars
        infogan_vars_subnet = network_utils.dense(infogan_vars_subnet,
                                              units=units_base*2,
                                              weight_norm=params.weight_norm,
                                              scope='custom02_subnet1')
        infogan_vars_subnet = tf.nn.leaky_relu(infogan_vars_subnet)
        infogan_vars_subnet = network_utils.dense(infogan_vars_subnet,
                                              units=units_base*4,
                                              weight_norm=params.weight_norm,
                                              scope='custom02_subnet2')
        infogan_vars_subnet = tf.nn.leaky_relu(infogan_vars_subnet)
        prev_net = tfgan.features.condition_tensor(prev_net,
                                                   infogan_vars_subnet)
    elif params.infogan_input_method in ['custom03', 'custom03v2']:
      prev_net = infogan_input_hook(prev_net, infogan_vars, infogan_num_vars,
                                    block_id)
    elif params.infogan_input_method == 'custom04':
      prev_net = infogan_input_hook(prev_net, infogan_vars, infogan_num_vars,
                                    block_id)
    elif params.infogan_input_method == 'append_channels':
      prev_net = _append_channels(prev_net, infogan_vars, infogan_num_vars,
                                  block_id)
    else:
      assert False, 'unknown infogan_input_method: %s' % (
          params.infogan_input_method)
    return prev_net

  # Takes in prev net, prev img at 2**(block_id-1).
  # Outputs net and image at resolution 2**block_id.
  def block(block_id, prev_net, prev_all_imgs):
    assert 2 <= block_id <= target_side_log, block_id

    if block_id > 2 and params.incubate:
      alpha = phase_progress_to_alpha(block_id, phase, phase_progress)
      prev_net = tf.cond(
        tf.less(alpha, 0.5),
        lambda: tf.stop_gradient(prev_net),
        lambda: prev_net,
      )
      prev_all_imgs = tf.cond(
        tf.less(alpha, 0.5),
        lambda: tf.stop_gradient(prev_all_imgs),
        lambda: prev_all_imgs,
      )


    with tf.variable_scope('block_%d' % block_id):

      infogan_vars = block_id_to_structured_vars[block_id]
      infogan_num_vars = len(infogan_vars)
      if infogan_vars:
        infogan_vars = tf.stack(infogan_vars, axis=1)
        infogan_vars = tf.layers.flatten(infogan_vars)
      else:
        infogan_vars = None

      prev_net = inject_infogan_vars(block_id, prev_net, infogan_vars,
                                     infogan_num_vars)

      side = 2**block_id
      if block_id == 2:
        assert side == 4
        net = prev_net
        if params.normalize_latents: net = network_utils.pixel_norm(net, axis=1)
        net = condition(net, block_id)
        net = network_utils.dense(net, units=4*4*channels_at_res(block_id),
                                  weight_norm=params.weight_norm)
        net = network_utils.norm(net, axis=1, version=params.norm,
                                 is_training=is_training, gpu_id=gpu_id,
                                 per_gpu=params.norm_per_gpu)
        net = tf.nn.leaky_relu(net)
        if params.pn_after_act: net = network_utils.pixel_norm(net, axis=1)
        net = tf.reshape(net, [-1, channels_at_res(block_id), 4, 4])
        net = infogan_input_hook(net, infogan_vars, infogan_num_vars, block_id)
        if params.double_conv: net = CCNA(net, block_id, scope='conv2')
        rgb_block = network_utils.upscale2d(
          torgb(net, block_id=block_id),
          2**(target_side_log-block_id))
        rgb_combined = rgb_block
      else:
        prev_net2 = network_utils.upscale2d(prev_net, 2)
        net = prev_net2
        net.shape.assert_is_compatible_with([None, None, side, side])
        net = infogan_input_hook(net, infogan_vars, infogan_num_vars,
                                 block_id)
        net = CCNA(net, block_id, scope='conv1')
        if params.double_conv:
          net = infogan_input_hook(net, infogan_vars, infogan_num_vars,
                                   block_id)
          net = CCNA(net, block_id, scope='conv2')
        if params.residual: net += prev_net_2
        rgb_block = network_utils.upscale2d(
          torgb(net, block_id=block_id),
          2**(target_side_log-block_id))
        # Alpha = how much img at current block contributes to output.
        alpha = phase_progress_to_alpha(block_id, phase, phase_progress)
        prev_rgb = tf.unstack(prev_all_imgs, axis=1)[block_id-2-1]
        if params.prev_rgb_stop_gradient:
          prev_rgb = tf.stop_gradient(prev_rgb)
        rgb_combined = (1.-alpha) * prev_rgb + alpha * rgb_block

    # Disabled temporarily. TODO
    #  _img_summary('rgb_block_%d' % block_id, rgb_block)
    #  _img_summary('rgb_combined_%d' % block_id, rgb_combined)
    tf.logging.info("Generator's block %d: net %s, rgb_block %s",
                    block_id, net.shape.as_list(), rgb_block.shape.as_list())

    tf.summary.histogram('block_%d_final_net' % block_id, net,
                         family='generator_activations')

    all_imgs = tf.unstack(prev_all_imgs, axis=1)
    all_imgs = (all_imgs[:block_id-2] +
                [rgb_combined] * (target_side_log - block_id + 1))
    all_imgs = tf.stack(all_imgs, axis=1)
    if block_id == target_side_log:
      out = all_imgs
    elif params.debug_mode or (not is_training):
      # We build entire network unconditionally, and so all tf.summary ops
      # are fetchable (they'll show up in tensorboard).
      out = block(block_id+1, prev_net=net, prev_all_imgs=all_imgs)
    else:
      out = tf.cond(
        tf.greater(phase, block_id),
        lambda: block(block_id+1, prev_net=net, prev_all_imgs=all_imgs),
        lambda: all_imgs
      )
    return out

  all_imgs_shape = [gpu_batch_size,
                    target_side_log-2+1,
                    3,
                    img_side,
                    img_side,]
  all_imgs = tf.constant(0., shape=all_imgs_shape)
  all_imgs = block(2, noise, all_imgs)
  all_imgs.set_shape(all_imgs_shape)
  all_imgs = tf.unstack(all_imgs, axis=1)
  all_imgs = map(tensor_util.nchw_to_nhwc, all_imgs)
  img = all_imgs[-1]
  tf.logging.info('generator_fn finished (gpu_id %s)', gpu_id)
  if return_dict:
    return dict(
      all_imgs=all_imgs,
      img=img,
    )
  else:
    return img


def discriminator_fn(image,
                     generator_inputs,
                     training_params,
                     is_training,
                     phase,
                     phase_progress,
                     return_dict=False,
                     gpu_id=0):
  """
    Args:
      params: (TrainingParams)
      iamge: (Tensor) NCHW input, real or generated images
      phase: (Tensor) 0-d integer, identifies which block is being trained
          at each moment. Iterates through [2, side_log].
      phase_progress: (Tensor) 0-d integer, identifies how far into the current
          phase we are. Iterates through [0, 100).

    Returns:
      A tensor with generated images, shape NCHW.
  """
  params = training_params.discriminator_params
  img_side = training_params.image_side
  tf.logging.info('discriminator_fn, reuse %s, gpu_id %s',
                  tf.get_variable_scope().reuse, gpu_id)
  one_hot_labels = generator_inputs['one_hot_labels']
  target_side_log = int(math.log(img_side, 2))
  assert 2**target_side_log == img_side
  image = tensor_util.nhwc_to_nchw(image)
  tf.summary.histogram('image', image)
  gpu_batch_size = get_gpu_batch_size(image, training_params)

  if params.image_noise_stddev:
    image_noise =  tf.random_normal(
      shape = image.shape, mean=0., stddev=params.image_noise_stddev)
    tf.summary.histogram('image_noise', image_noise)
    image = image + image_noise
    tf.summary.histogram('image_with_noise_added', image)

  def channels_at_res(res):
    channels = params.channels_at_2x2 / 2**(res-1)
    channels = min(channels, params.channels_max)
    channels = max(channels, params.channels_min)
    return channels

  def condition(tensor):
    if params.conditioning is False:
      return tensor
    return network_utils.append_one_hot_to_tensor(tensor, one_hot_labels)

  def fromrgb(img, block_id, scope='fromrgb'):
    with tf.variable_scope(scope):
      net = img
      if params.fromrgb_use_n_img_diffs:
        for d in range(1, min(params.fromrgb_use_n_img_diffs, block_id-2)+1):
          scale = 2**d
          tf.logging.info('D.block_%d fromrgb uses img diff with scale %d',
                          block_id, scale)
          img_down = img
          img_down = network_utils.downscale2d(img_down, scale)
          img_down = network_utils.upscale2d(img_down, scale)
          net = tf.concat([net, img-img_down], axis=1)
      tf.logging.info('D.block_%d fromrgb input net shape %s', block_id,
                      net.shape.as_list())

      if params.second_conv_channels_x2:
        channels = channels_at_res(block_id-1)
      else:
        channels = channels_at_res(block_id)
      net = condition(net)
      net = network_utils.conv(net, filters=channels,
                               kernel_size=1, weight_norm=params.weight_norm)
      net = tf.nn.leaky_relu(net)
      return net


  # CCNA - conditino, conv, norm, activation
  def CCNA(net, block_id, scope='CCNA', second=False):
    with tf.variable_scope(scope):
      net = condition(net)
      if second and params.second_conv_channels_x2:
        channels=channels_at_res(block_id-2)
      else:
        channels=channels_at_res(block_id-1)
      net = network_utils.conv(net, filters=channels,
                               weight_norm=params.weight_norm,
                               kernel_size=params.kernel_size)
      net = network_utils.norm(net, axis=1, version=params.norm,
                               is_training=is_training, gpu_id=gpu_id,
                               per_gpu=params.norm_per_gpu)
      net = tf.nn.leaky_relu(net)
      if params.pn_after_act: net = network_utils.pixel_norm(net, axis=1)
      return net

  # Consumes image/net with resolution 2**block_id, and returns a list:
  # net, predictions for cat variables,predictions for cont vars
  # This is not pretty but is required for the dynamic structure (tf.conds) to
  # work properly.
  def block(block_id):
    assert 2 <= block_id <= target_side_log
    _dummy_infogan_cat = tf.ones([gpu_batch_size,
                                  training_params.infogan_cat_num_vars or 0,
                                  training_params.infogan_cat_dim or 0])
    _dummy_infogan_cont = tf.zeros([gpu_batch_size,
                                    training_params.infogan_cont_num_vars or 0])
    _dummy_next_net = tf.zeros([gpu_batch_size,
                                channels_at_res(block_id-1
                                                if params.second_conv_channels_x2
                                                else block_id),
                                2**block_id,
                                2**block_id])
    dummy_next_block_output = [_dummy_next_net, _dummy_infogan_cat,
                               _dummy_infogan_cont]

    if block_id == target_side_log:
      next_block = lambda: dummy_next_block_output
    else:
      next_block = lambda: block(block_id+1)

    with tf.variable_scope('block_%d' % block_id):
      # Translate input image to desired channels and resolution.
      img = network_utils.downscale2d(image, 2**(target_side_log - block_id))
      img.shape.assert_is_compatible_with([None, 3, 2**block_id, 2**block_id])
      img_with_channels = fromrgb(img, block_id, 'img_with_channels')

      # input_net: either image, next block, or a mix of the two.
      # Note: here alpha means "to what extent we use the net from next block",
      # that's why we need to pass block_id+1.
      alpha = phase_progress_to_alpha(block_id+1, phase, phase_progress)
      tf.summary.scalar('alpha', alpha)
      if params.debug_mode or (not is_training):
        next_net, infogan_cat, infogan_cont = next_block()
      else:
        next_net, infogan_cat, infogan_cont = tf.cond(
          tf.less(alpha, 1e-8),
          lambda: dummy_next_block_output,
          lambda: next_block(),
        )
      next_net.set_shape(img_with_channels.shape)
      input_net = (1. - alpha) * img_with_channels + alpha * next_net
      if params.elastic_block_input:
        with tf.variable_scope('elastic_block_input'):
          # A weight for each channel.
          w = tf.get_variable(
            'elastic_block_input_w',
            shape=[1, img_with_channels.shape.as_list()[1], 1, 1],
            initializer=tf.initializers.random_normal)
          w = tf.sigmoid(w)
          tf.logging.info('Elastic block input for block_id %d. Weights shape '
                          '%s', block_id, w.shape.as_list())
          input_net = w * input_net + (1. - w) * img_with_channels

      # Actual work: conv and downscale.
      if block_id == 2:
        net = network_utils.concat_features_stddev(input_net)
        net = CCNA(net, block_id)
        net = _flatten(net)
        net = condition(net)
        net = network_utils.dense(net, units=channels_at_res(block_id-1),
                                  weight_norm=params.weight_norm)
        net = network_utils.norm(net, axis=1, version=params.norm,
                                 is_training=is_training, gpu_id=gpu_id,
                                 per_gpu=params.norm_per_gpu)
        net = tf.nn.leaky_relu(net)
        if params.pn_after_act: net = network_utils.pixel_norm(net, axis=1)
      else:
        net = CCNA(input_net, block_id, scope='conv1')
        if params.double_conv: net = CCNA(net, block_id, scope='conv2',
                                          second=True)
        if params.residual: net += input_net
        net = network_utils.downscale2d(net)
        net.shape.assert_is_compatible_with(
          [None, None, 2**(block_id-1), 2**(block_id-1)])

      # Logging.
      tf.logging.info("Discriminator's block %d, img_with_channels: %s, net %s",
                      block_id, img_with_channels.shape.as_list(),
                      net.shape.as_list())
      tf.summary.histogram('net', net)
      #  _img_summary('img_block_%d' % block_id, img)


      # InfoGAN predictions.
      if training_params.infogan_cont_num_vars:
        if params.infogan_all_vars_in_block2:
          depth_to_vars = {2: range(training_params.infogan_cont_num_vars)}
        else:
          depth_to_vars = (training_params.infogan_cont_depth_to_vars or
                           {2: range(training_params.infogan_cont_num_vars)})
        for coord in depth_to_vars.get(block_id, []):
          if params.infogan_max_pool and block_id>=4:
            predicted_loc_input = tf.nn.max_pool(
              value=net,
              ksize=[1,1,2,2],
              strides=[1,1,2,2],
              padding='SAME',
              data_format='NCHW',
            )
          else:
            predicted_loc_input = net
          predicted_loc = network_utils.dense(
            _flatten(predicted_loc_input), units=1, weight_norm=params.weight_norm,
            scope='predicted_cont_loc_%02d' % coord)
          infogan_cont = tf.concat([
            infogan_cont[:,:coord],
            predicted_loc,
            infogan_cont[:,coord+1:],
          ], axis=1)
          tf.logging.info('InfoGAN: predicting cont var %d from D.block_%d',
                          coord, block_id)
      if training_params.infogan_cat_num_vars:
        depth_to_vars = (training_params.infogan_cat_depth_to_vars or
                         {2: range(training_params.infogan_cat_num_vars)})
        for coord in depth_to_vars.get(block_id, []):
          logits = network_utils.dense(_flatten(net),
                                       units=training_params.infogan_cat_dim,
                                       weight_norm=params.weight_norm,
                                       scope='categorical_%d_logits' % coord)
          #  softmax = tf.nn.softmax(logits)
          logits.shape.assert_is_compatible_with(
            [gpu_batch_size, training_params.infogan_cat_dim])
          infogan_cat = tf.concat([
            infogan_cat[:,:coord,:],
            tf.expand_dims(logits, axis=1),  # XXX
            infogan_cat[:,coord+1:,:],
          ], axis=1)
          tf.logging.info('InfoGAN: predicting cat var %d from D.block_%d',
                          coord, block_id)
      tf.summary.histogram('block_%d_final_net' % block_id, net,
                           family='discriminator_activations')
      return [net, infogan_cat, infogan_cont]


  net, predicted_categorical_logits, predicted_distributions_loc = block(2)
  logits = network_utils.dense(net, units=1, weight_norm=params.weight_norm,
                               scope='logits')
  if training_params.infogan_cont_num_vars:
    predicted_distributions_loc.shape.assert_is_compatible_with(
      [gpu_batch_size, training_params.infogan_cont_num_vars])
  else:
    predicted_distributions_loc = None

  if training_params.infogan_cat_num_vars:
    predicted_categorical_softmaxes = tf.nn.softmax(
      predicted_categorical_logits, axis=-1)
    tf.summary.histogram('predicted_categorical_logits',
                         predicted_categorical_logits)
    tf.summary.histogram('predicted_categorical_softmaxes',
                         predicted_categorical_softmaxes)
    predicted_categorical_softmaxes.shape.assert_is_compatible_with(
      [gpu_batch_size, training_params.infogan_cat_num_vars,
       training_params.infogan_cat_dim])
    predicted_categorical_softmax_list = [
      predicted_categorical_softmaxes[:,i,:]
      for i in range(training_params.infogan_cat_num_vars)]
  else:
    predicted_categorical_softmax_list = None

  tf.summary.histogram('logits', logits)
  tf.logging.info('discriminator_fn finished (gpu_id %d)', gpu_id)
  if return_dict:
    return dict(
      logits=logits,
      predicted_distributions_loc=predicted_distributions_loc,
      predicted_categorical_softmax_list=predicted_categorical_softmax_list,
    )
  else:
    return logits


def _img_summary(name, img):
  tf.summary.histogram(name, img)
  #  img = tf.clip_by_value(img, -1., 1.)  # Breaks tensorflow model server.
  img = tf.maximum(img, -1.)
  img = tf.minimum(img, 1.)
  img = tensor_util.nchw_to_nhwc(img)
  tf.summary.image(name, img)


def _flatten(net):
  shape = net.shape.as_list()
  def prod(x): return x[0]*prod(x[1:]) if x else 1
  return tf.reshape(net, shape=[shape[0] or -1, prod(shape[1:])])


def _get_progressive_mask(training_params, phase_t, phase_progress_t):
  tp = training_params
  depth_to_vars = tp.infogan_cont_unmask_depth_to_vars
  # TOOD: gpu_batch_size = tp.batch_size / tp.num_gpus
  num_vars = tp.infogan_cont_num_vars

  mask = tf.constant([0] * num_vars, dtype=tf.float32)
  for phase, coords in depth_to_vars.items():
    phase_mask = [1 if i in coords else 0 for i in range(num_vars)]
    phase_mask = tf.constant(phase_mask, dtype=tf.float32)

    mask = tf.cond(
      tf.greater_equal(phase_t, tf.constant(phase, dtype=tf.int64)),
      lambda: tf.maximum(mask, phase_mask),
      lambda: mask,
    )
  tf.summary.scalar('progressive_mask_total_ones', tf.reduce_sum(mask),
                    family='progressive_mask')
  for i in range(20):
    if i < num_vars: tf.summary.scalar('progressive_mask_coord_%d' % i, mask[i],
                                       family='progressive_mask')
  mask = tf.stack([mask]*tp.batch_size_per_gpu)
  return mask


def get_gpu_batch_size(tensor, training_params):
  # TODO: this should be computed dynamically, not based on tp.batch_size.
  return training_params.batch_size / training_params.num_gpus
  #  return tf.shape(tensor)[0] / training_params.num_gpus


def _split_structured_input(tp, inp, num_vars, var_name_prefix):
  gpu_batch_size = get_gpu_batch_size(inp, tp)
  #  assert inp.shape.is_compatible_with([gpu_batch_size, num_vars])
  res = {}
  for coord in range(num_vars):
    t = tf.slice(inp, [0, coord], [gpu_batch_size, 1])
    t = tf.identity(t, name=var_name_prefix + str(coord))
    res[coord] = t
  return res


def _split_structured_continuous_input(tp, inp, phase, phase_progress):
  if inp is None: return {}

  if tp.infogan_cont_unmask_depth_to_vars is not None:
    inp = inp * _get_progressive_mask(tp, phase, phase_progress)

  return _split_structured_input(tp, inp, tp.infogan_cont_num_vars,
                                 'cont_')

def _split_structured_categorical_input(tp, inp):
  if inp is None: return {}
  d = _split_structured_input(tp, inp, tp.infogan_cat_num_vars, 'cat_')
  gpu_batch_size = get_gpu_batch_size(inp, tp)
  res = {}
  for key, val in d.items():
    name = val.op.name.rpartition('/')[-1] + '_one_hot'
    one_hot = tf.one_hot(val, depth=tp.infogan_cat_dim)
    one_hot.shape.assert_is_compatible_with(
      [gpu_batch_size, 1, tp.infogan_cat_dim])
    one_hot = one_hot[:,0,:]
    one_hot = tf.identity(one_hot, name=name)
    one_hot.shape.assert_is_compatible_with(
      [gpu_batch_size, tp.infogan_cat_dim])
    res[key] = one_hot
  return res


def _block_id_to_structured_vars_internal(phase_to_coords, coord_to_tensor, tp):
  if phase_to_coords is None:
    phase_to_coords = {2: list(coord_to_tensor.keys())}

  coord_to_phase = {}
  for phase, coords in phase_to_coords.items():
    for coord in coords:
      coord_to_phase[coord] = phase

  highest_block = int(math.log(tp.image_side, 2))
  res = {block_id: [] for block_id in range(2, highest_block+1)}
  for coord, tensor in coord_to_tensor.items():
    phase = coord_to_phase[coord]
    if phase in res:
      res[phase].append(tensor)
  return res


def _block_id_to_structured_vars(generator_inputs, tp, phase, phase_progress):
  cont_vars = _split_structured_continuous_input(
    tp, generator_inputs['structured_continuous_input'],
    phase, phase_progress)
  cat_vars = _split_structured_categorical_input(
    tp, generator_inputs['structured_categorical_input'])

  cont = _block_id_to_structured_vars_internal(
    tp.infogan_cont_depth_to_vars,
    cont_vars,
    tp,
  )
  cat = _block_id_to_structured_vars_internal(
    tp.infogan_cat_depth_to_vars,
    cat_vars,
    tp,
  )
  assert sorted(cont.keys()) == sorted(cat.keys())
  res = {}
  for block_id in cont.keys():
    res[block_id] = cat[block_id] + cont[block_id]
  return res


