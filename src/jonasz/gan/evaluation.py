import tensorflow as tf
import time
import numpy as np
import math
import os
from tensorflow.contrib import gan as tfgan
from tensorflow.python.training import basic_session_run_hooks
from jonasz.progressive_infogan import progressive_infogan_losses
from tensorflow.contrib.gan.python.eval.python import summaries as tfgan_summaries
from tensorflow.python.training import saver
from jonasz.lib import util
from tensorflow.python.training import training_util

class EvalParams(util.Params):
  def get_allowed_params_with_defaults(self):
    return dict(
      eval_dir=None,  # By default creates a training_dir/eval subdir.
      use_ema_params=True,
      inception_num_images = 16000,
      frechet_num_images = 5000,
      msssim_num_pairs = 2000,
      isolation_num_images = None,  # NOTE: only works with progressive infogan.
      infogan_metrics_num_iters = 3000,
    )


def _inception_logits(gan):
  # Let's make sure we're not evaluating too many images at a time, so we don't
  # run out of memory. Note that during training we might be working with 128
  # low res images, but in preprocess_image they are upscaled.
  inp = gan.generated_data[:16]
  generated_images_preprocessed = tfgan.eval.preprocess_image((inp+1.)*128.)
  return tfgan.eval.run_inception(generated_images_preprocessed)


def _inception_final_pool(gan):
  # Let's make sure we're not evaluating too many images at a time, so we don't
  # run out of memory. Note that during training we might be working with 128
  # low res images, but in preprocess_image they are upscaled.
  max_imgs = 16
  generated_images_preprocessed = tfgan.eval.preprocess_image(
    (gan.generated_data[:max_imgs]+1.)*128.)
  real_images_preprocessed = tfgan.eval.preprocess_image(
    (gan.real_data[:max_imgs]+1.)*128.)

  def inception(x):
    return tfgan.eval.run_inception(x, output_tensor='pool_3:0')
  return (inception(generated_images_preprocessed),
          inception(real_images_preprocessed))

def _write_summaries_internal(eval_dir, sess, cur_global_step, summaries=None,
                              feed_dict=None):
  writer = tf.summary.FileWriter(eval_dir)
  summaries = summaries or tf.get_collection(tf.GraphKeys.SUMMARIES)
  summaries = filter(sess.graph.is_fetchable, summaries)
  for x in sess.run(summaries, feed_dict=feed_dict):
    writer.add_summary(x, global_step=cur_global_step)
  writer.flush()
  writer.close()

def _write_summaries(eval_dir, sess, cur_global_step, summaries=None,
                     feed_dict=None, ema_feed_dict=None):
  _write_summaries_internal(eval_dir, sess, cur_global_step,
                            summaries=summaries, feed_dict=feed_dict)
  if ema_feed_dict:
    ema_eval_dir = os.path.join(eval_dir, 'ema')
    tf.gfile.MakeDirs(ema_eval_dir)
    _write_summaries_internal(ema_eval_dir, sess, cur_global_step,
                              summaries=summaries, feed_dict=ema_feed_dict)


def _accumulate_n(tensor, sess, n):
  res = []
  i = 0
  have = 0
  while have < n:
    cur = sess.run(tensor)
    have += len(cur)
    res.append(cur)
    if i % 10 == 0:
      tf.logging.info('Accumulating tensor values: %d / %d', have, n)
    i += 1
  return np.concatenate(res)[:n]


# Note: We're evaluating logits_t on sess so that we can use more datapoints
# for the calculation of the inception distance, while still fitting in the
# memory. Perhaps there's a better way of doing this?

# TODO: For cifar, both _calc* functions work on modified (augumented) data.
# That probably shouldn't be the case.
def _maybe_calc_inception_score(gan, sess, params):
  if not params.inception_num_images:
    return
  tf.logging.info('Calculating inception score.')
  logits_t = _inception_logits(gan)
  logits_t = logits_t[:16]
  logits = _accumulate_n(logits_t, sess, params.inception_num_images)

  inception_score_t = tfgan.eval.classifier_score_from_logits(
    tf.constant(logits))

  tf.summary.scalar('inception_score', inception_score_t,
                    family='quality')


def _maybe_calc_frechet_inception_distance(gan, sess, eval_params):
  if not eval_params.frechet_num_images:
    return
  tf.logging.info('Calculating Frechet inception distance.')
  inception_gen_t, inception_real_t = _inception_final_pool(gan)

  inception_gen = _accumulate_n(inception_gen_t, sess,
                                eval_params.frechet_num_images)
  inception_real = _accumulate_n(inception_real_t, sess,
                                 eval_params.frechet_num_images)

  frechet_dist_t = tfgan.eval.frechet_classifier_distance_from_activations(
    tf.constant(inception_real), tf.constant(inception_gen))

  tf.summary.scalar('frechet_distance', frechet_dist_t,
                    family='quality')


def calc_mean(vals):
  mean = np.mean(vals)
  se = np.std(vals, ddof=1) / len(vals)
  return mean, se


def _maybe_calc_infogan_isolation(gan, sess, training_params, version='msssim'):
  if not training_params.infogan_cont_num_vars:
    return
  if not training_params.eval_params.isolation_num_images:
    return
  assert version in ['msssim', 'mse']
  tf.logging.info('Calculating infogan %s isolation' % version)
  assert set(gan.generator_inputs.keys()) == {
    'noise',
    'structured_categorical_input',
    'structured_continuous_input',
    'one_hot_labels',
  }
  assert gan.generator_inputs['structured_categorical_input'] is None
  noise = gan.generator_inputs['noise']
  code = gan.generator_inputs['structured_continuous_input']

  # Graph creation is quite costly, so doing it outside of msssim func to make
  # it run fast.
  _dynamic_scale=2.
  _shape=[None, training_params.image_side, training_params.image_side, 3]
  _imgs1_placeholder = tf.placeholder(tf.float32, shape=_shape)
  _imgs2_placeholder = tf.placeholder(tf.float32, shape=_shape)
  _imgs1_t = tf.image.resize_images(
    _imgs1_placeholder, size=[256, 256],
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  _imgs2_t = tf.image.resize_images(
    _imgs2_placeholder, size=[256, 256],
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  _sim = tf.image.ssim_multiscale(_imgs1_t, _imgs2_t, _dynamic_scale)

  def msssim(a, b):
    a = a.copy()
    b = b.copy()
    def prepare(imgs):
      # Assuming dynamic range [-1., 1.]
      #  assert -1. <= np.min(imgs) <= -0.2, np.min(imgs)
      #  assert 0.2 <= np.percentile(imgs, 95) <= 1., np.percentile(imgs, 95)
      imgs += 1. # Now in [0., 2.]
      assert _dynamic_scale == 2.
      return imgs

    a = prepare(a)
    b = prepare(b)
    sim = sess.run(_sim, feed_dict={_imgs1_placeholder: a,
                                    _imgs2_placeholder: b})
    assert sim.shape == (training_params.batch_size_per_gpu,)
    return list(sim)

  def mse(a, b):
    res = -np.mean((a-b)**2., axis=(1,2,3))
    assert res.shape == (training_params.batch_size_per_gpu,)
    return list(res)

  isolation_per_coord = {}
  for coord in range(training_params.infogan_cont_num_vars):
    ceil_div = lambda a, b: (a+b-1)/b
    reps = ceil_div(training_params.eval_params.isolation_num_images,
                    training_params.batch_size_per_gpu)

    all_isos = []
    for _ in range(reps):
      fixed_noise, fixed_code = sess.run([noise, code])
      def _code_at_coord_val(coord, val):
        cur_code = fixed_code.copy()
        cur_code[:,coord] = np.array([val] * cur_code.shape[0])
        return cur_code
      def _imgs_at_coord_val(coord, val):
        cur_code = _code_at_coord_val(coord, val)
        imgs = sess.run(gan.generated_data, feed_dict={noise: fixed_noise,
                                                       code: cur_code})
        return imgs
      mid_imgs = _imgs_at_coord_val(coord, 0.)
      for val in (-3, -2, -1, 1, 2, 3):
        cur_imgs = _imgs_at_coord_val(coord, training_params.noise_stddev*val)
        if version == 'msssim':
          all_isos.extend(list(msssim(cur_imgs, mid_imgs)))
        else:
          assert version == 'mse'
          all_isos.extend(list(mse(cur_imgs, mid_imgs)))

    isolation, se = calc_mean(all_isos)
    isolation_per_coord[coord] = isolation
    tf.logging.info('coord %d, %s isolation %f (se %f)',
                    coord, version, isolation, se)
    tf.summary.scalar('infogan_%s_isolation_coord_%d' % (version, coord),
                      tf.constant(isolation),
                      family='infogan_%s_isolation_per_coord' % version)

  for block in training_params.block_ids:
    coords = block_to_corresponding_cont_coords(training_params, block)
    block_isolation = np.mean([isolation_per_coord[coord] for coord in coords])
    tf.summary.scalar('infogan_%s_isolation_block_%d' % (version, block),
                      tf.constant(block_isolation),
                      family='infogan_%s_isolation' % version)

  tf.summary.scalar('infogan_%s_isolation' % version,
                    tf.constant(np.mean(isolation_per_coord.values())),
                    family='infogan_%s_isolation' % version)


def block_to_corresponding_cont_coords(tp, block_id):
  architecture = tp.infogan_cont_depth_to_vars
  unmask       = tp.infogan_cont_unmask_depth_to_vars
  activate     = tp.infogan_cont_loss_phase_to_active_coords

  if unmask is None and activate is None:
    return architecture[block_id]

  # When both are present, they need to agree.
  if unmask is not None and activate is not None:
    assert unmask[block_id] == activate[block_id]

  # When more complex (unmask / activate) settings are present, all vars
  # need to be fed to layer 2; or archtecture needs to align with the complex
  # settings.
  assert (sorted(architecture[2]) == range(tp.infogan_cont_num_vars) or
          architecture == (unmask or activate))

  return (unmask or activate)[block_id]


def _maybe_calc_infogan_metrics(gan_model_dict, sess, training_params):
  tp = training_params
  if not tp.eval_params.infogan_metrics_num_iters:
    return
  if not tp.infogan_cont_num_vars:
    return
  # tensors
  mi_penalty_per_coord_t = {}
  abs_error_per_coord_t = {}
  # coord -> array of actual values
  mi_penalty_per_coord = {}
  abs_error_per_coord = {}

  for coord in range(tp.infogan_cont_num_vars):
    mip, abs_err = (progressive_infogan_losses.
                    unweighted_mutual_information_penalty_per_coord(
                      gan_model_dict, tp, coord))
    mi_penalty_per_coord_t[coord] = mip
    abs_error_per_coord_t[coord] = abs_err
    mi_penalty_per_coord[coord] = []
    abs_error_per_coord[coord] = []

  for i in range(tp.eval_params.infogan_metrics_num_iters):
    if i % 10 == 0:
      tf.logging.info('_maybe_calc_infogan_metrics, iter %d / %d',
                      i, tp.eval_params.infogan_metrics_num_iters)
    cur_mi_penalty_per_coord, cur_abs_error_per_coord = sess.run([
      mi_penalty_per_coord_t, abs_error_per_coord_t])

    for coord in range(tp.infogan_cont_num_vars):
      mi_penalty_per_coord[coord].append(cur_mi_penalty_per_coord[coord])
      abs_error_per_coord[coord].append(cur_abs_error_per_coord[coord])

  mean_mi_penalty_per_coord = {}
  mean_abs_error_per_coord = {}
  for coord in range(tp.infogan_cont_num_vars):
    mean, se = calc_mean(mi_penalty_per_coord[coord])
    mean_mi_penalty_per_coord[coord] = mean
    tf.logging.info('Mutual information penalty, coord %d, val %f, se %f',
                    coord, mean, se)
    tf.summary.scalar('mutual_information_penalty_coord_%d' % coord,
                      mean, family='mutual_information_penalty_per_coord')

    mean, se = calc_mean(abs_error_per_coord[coord])
    mean_abs_error_per_coord[coord] = mean
    tf.logging.info('InfoGAN absolute prediction error, coord %d, val %f, se %f',
                    coord, mean, se)
    tf.summary.scalar('infogan_absolute_prediction_error_%d' % coord,
                      mean, family='infogan_absolute_prediction_error_per_coord')

  tf.summary.scalar('mutual_information_penalty_overall_mean',
                    np.mean(mean_mi_penalty_per_coord.values()))
  tf.summary.scalar('infogan_absolute_prediction_error_overall_mean',
                    np.mean(mean_abs_error_per_coord.values()))

  for block in tp.block_ids:
    coords = block_to_corresponding_cont_coords(tp, block)
    tf.summary.scalar('mutual_information_penalty_block_%d_mean' % block,
                      np.mean([mean_mi_penalty_per_coord[coord]
                               for coord in coords]),
                      family='mutual_information_penalty_per_block')
    tf.summary.scalar('infogan_absolute_prediction_error_block_%d_mean' % block,
                      np.mean([mean_abs_error_per_coord[coord]
                               for coord in coords]),
                      family='infogan_absolute_prediction_error_per_block')




class _RestoreGANSession(object):
  def __init__(self, model_dir = None, vars_to_restore=None,
               use_ema_params=None, gan=None):
    assert use_ema_params is not None
    self.model_dir = model_dir
    if use_ema_params:
      assert vars_to_restore is None
      assert gan is not None
      tmp_params_ema = tf.train.ExponentialMovingAverage(decay=0.9)
      self.vars_to_restore = tmp_params_ema.variables_to_restore(
        moving_avg_variables=gan.generator_variables)
    else:
      self.vars_to_restore = vars_to_restore
    tf.logging.info('_RestoreGANSession variables_to_restore:\n')
    for key, val in sorted((self.vars_to_restore or {}).items()):
      tf.logging.info('%s: %s', key, val.op.name)

  def __enter__(self):
    config=tf.ConfigProto(allow_soft_placement=True)
    #  config.gpu_options.per_process_gpu_memory_fraction = 0.3
    self.sess = tf.Session(config=config)
    return_val = self.sess.__enter__()
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(coord=self.coord)

    if self.model_dir:
      tf.logging.info('Model dir is %s', self.model_dir)
      saver_for_restore = saver.Saver(var_list=self.vars_to_restore)
      checkpoint_path = saver.latest_checkpoint(self.model_dir)
      tf.logging.info('Restoring checkpoints from %s', checkpoint_path)
      saver_for_restore.restore(self.sess, checkpoint_path)

    return return_val

  def __exit__(self, *args):
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.__exit__(*args)


def _get_side_for_dynamic_mse(i, steps, img_side):
  phases = int(math.log(img_side, 2))
  assert img_side == 2**phases
  return 2**(phases * i / steps + 1)


class maybe_gpu_tower_scope(object):
  def __init__(self, use_gpu_tower_scope):
    if use_gpu_tower_scope:
      # Not particularly pretty: make sure global step is not created under
      # 'gpu_tower' scope later on.
      tf.train.get_or_create_global_step()
      self.variable_scope = tf.variable_scope('gpu_tower')
    else:
      self.variable_scope = None

  def __enter__(self):
    if self.variable_scope:
      return self.variable_scope.__enter__()

  def __exit__(self, *args):
    if self.variable_scope:
      self.variable_scope.__exit__(*args)


def _draw_images(training_params, gan_fn, how_many=32, type_='real',
                 rescale_list=()):
  eval_params = training_params.eval_params
  assert type_ in ['real', 'generated']
  with tf.Graph().as_default():
    with tf.device('/gpu:0'):
      with maybe_gpu_tower_scope(training_params.use_gpu_tower_scope):
        gan = gan_fn(is_training=False)
      imgs = []

      with _RestoreGANSession(training_params.output_dir,
                              use_ema_params=eval_params.use_ema_params,
                              gan=gan) as sess:
        i = 0
        while len(imgs) < how_many:
          if i % 10 == 0:
            tf.logging.info('Draw %s images %d / %d',
                            type_, len(imgs), how_many)
          i += 1

          if type_ == 'real':
            imgs_t = gan.real_data
          else:
            assert type_ == 'generated'
            imgs_t = gan.generated_data

          for rescale in rescale_list:
            imgs_t = tf.image.resize_images(
              imgs_t, size=[rescale, rescale],
              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

          cur_imgs = sess.run(imgs_t)
          imgs.extend(list(cur_imgs))

      return np.stack(imgs[:how_many]).astype(np.float32)




def _calc_msssim_diversity_from_images(images):
  # Assuming dynamic range [-1., 1.]
  #  assert -1. <= np.percentile(images, 5) <= -0.2, np.percentile(images, 5)
  #  assert 0.2 <= np.percentile(images, 95) <= 1., np.percentile(images, 95)
  images += 1. # Now in [0., 2.]
  dynamic_range = 2.

  sims = []
  with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(dtype=tf.float32, shape=[1,256,256,3])
    y = tf.placeholder(dtype=tf.float32, shape=[1,256,256,3])
    sim = tf.image.ssim_multiscale(x, y, dynamic_range)
    for a, b in zip(images[::2], images[1::2]):
      cur_sim = sess.run(sim, feed_dict={x:a.reshape([1,256,256,3]),
                                         y:b.reshape([1,256,256,3])})
      sims.append(cur_sim)
  mean, se = calc_mean(sims)
  return 1.-mean, se



def _maybe_calc_msssim_diversity(gan_fn, training_params):
  params = training_params.eval_params
  if not params.msssim_num_pairs:
    tf.logging.info('Skipping msssim diversity')
    return

  tf.logging.info('Calculating msssim diversity')

  # First we scale down to current image side, so real data is the same
  # resolution. Then we scale to 256, so that msssim works well for small images
  # as well.
  rescale_list=[training_params.image_side, 256]
  real_imgs = _draw_images(training_params, gan_fn,
                           how_many=2*params.msssim_num_pairs, type_='real',
                           rescale_list=rescale_list)
  msssim_diversity_real, se_real = _calc_msssim_diversity_from_images(real_imgs)
  del real_imgs
  tf.logging.info('msssim_diversity_real %s (se %f)',
                  msssim_diversity_real, se_real)

  gen_imgs = _draw_images(training_params, gan_fn,
                          how_many=2*params.msssim_num_pairs,
                          type_='generated',
                          rescale_list=rescale_list)
  msssim_diversity_gen, se_gen = _calc_msssim_diversity_from_images(gen_imgs)
  tf.logging.info('msssim_diversity_gen %s (se %f)',
                  msssim_diversity_gen, se_gen)
  del gen_imgs

  # All this just to write summaries.
  with tf.Graph().as_default():
    tf.summary.scalar('msssim_diversity_real',
                      tf.constant(msssim_diversity_real),
                      family='quality')
    tf.summary.scalar('msssim_diversity_gen',
                      tf.constant(msssim_diversity_gen),
                      family='quality')
    training_global_step = training_util.get_or_create_global_step()
    # Recovering the global step.
    with _RestoreGANSession(training_params.output_dir,
                            use_ema_params=False) as sess:
      cur_global_step = sess.run(training_global_step)
      eval_dir = (params.eval_dir
                  or os.path.join(training_params.output_dir, 'eval'))
      _write_summaries(eval_dir, sess, cur_global_step)


def evaluate(gan_fn, training_params):
  eval_params = training_params.eval_params
  tf.logging.info('Running evaluation, training params %s.',
                  str(training_params))
  if training_params.eval_params is None:
    return
  eval_dir = (training_params.eval_params.eval_dir
              or os.path.join(training_params.output_dir, 'eval'))
  with util.TFFileLogger(eval_dir, is_gcloud=training_params.is_gcloud):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info('Starting evaluation')
    tf.logging.info('Training params: %s', training_params)

    _maybe_calc_msssim_diversity(gan_fn, training_params)

    with tf.Graph().as_default():
      with maybe_gpu_tower_scope(training_params.use_gpu_tower_scope):
        gan_model_dict = gan_fn(is_training=False, return_dict=True)
        gan = gan_model_dict['gan_model']
      # TODO: why generator scope? Probably due to the way I create optimizers
      # for training. Fix this.
      #  with tf.variable_scope('generator'):
      global_step = training_util.get_or_create_global_step()

      with _RestoreGANSession(training_params.output_dir,
                              use_ema_params=eval_params.use_ema_params,
                              gan=gan) as sess:
        cur_global_step = sess.run(global_step)
        tf.logging.info('Current global_step: %d', cur_global_step)

        _maybe_calc_infogan_metrics(gan_model_dict, sess, training_params)
        _maybe_calc_infogan_isolation(gan, sess, training_params, 'msssim')
        _maybe_calc_infogan_isolation(gan, sess, training_params, 'mse')
        _maybe_calc_inception_score(gan, sess, training_params.eval_params)
        _maybe_calc_frechet_inception_distance(gan, sess,
                                               training_params.eval_params)

        _write_summaries(eval_dir, sess, cur_global_step)

    return cur_global_step
