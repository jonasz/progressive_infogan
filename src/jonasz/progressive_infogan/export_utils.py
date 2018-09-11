import os
import re
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.training import saver
from jonasz.progressive_infogan import networks
from jonasz import constants
from jonasz.lib import util
import importlib

def _define_flags():
  global FLAGS
  flags = tf.flags
  flags.DEFINE_string('experiment_module', None,
                      ('Specifies for which experiment to work with. Format',
                       'as in an import statemtent: jonasz.experiments...'))
  flags.DEFINE_boolean('export_with_ema_params', False,
                       'Whether to load moving average versions of variables.')
  flags.DEFINE_integer('num_newest_checkpoints', None,
                     ('If provided, variables are averaged across the given number'
                      ' of latest checkpoints. Otherwise, the newest checkpoint'
                      ' is used.'))
  flags.DEFINE_string('subdir', None,
                      ('Optional, custom subdir to save the saved_model into.'))
  flags.DEFINE_integer('ignore_checkpoints_after', None,
                      ('Optional, useful to load an older version.'))
  FLAGS = flags.FLAGS


def _restore_average_from_multiple_checkpoints(saver_for_restore,
                                               sess, checkpoint_paths):
  tf.logging.info('_restore_average_from_multiple_checkpoints')
  vars_list = tf.trainable_variables()
  tot_vars = None
  for checkpoint_path in checkpoint_paths:
    tf.logging.info('Restoring from %s', checkpoint_path)
    saver_for_restore.restore(sess, checkpoint_path)
    cur_vars = {v.name: sess.run(v) for v in vars_list}
    if tot_vars:
      for key, val in cur_vars.items():
        tot_vars[key] += val
    else:
      tot_vars = cur_vars.copy()
  den = float(len(checkpoint_paths))
  for var in vars_list:
    val = tot_vars[var.name] / den
    sess.run(var.assign(val))


def export_newest_savedmodel(training_params,
                             export_with_ema_params=False,
                             checkpoint_paths=None,
                             subdir=None):
  # TODO: Fix this. We shouldn't need train here.
  # Inline import to work around circular dependency.
  from jonasz.progressive_infogan import train
  if not subdir:
    subdir = 'saved_model'
    if export_with_ema_params:
      subdir += '_ema'
    if checkpoint_paths:
      subdir += '_averaged'
  cur_time = str(int(time.time()))
  tmpexport_dir = os.path.join(training_params.output_dir, subdir,
                               '.tmp_' + cur_time)
  export_dir = os.path.join(training_params.output_dir, subdir, cur_time)
  if tf.gfile.Exists(export_dir):
    tf.logging.warning(
      'Exporting cancelled. Export dir already exists: %s', export_dir)
    return None

  with tf.Graph().as_default() as g:
    serialized_examples = tf.placeholder(tf.string)
    features = tf.parse_example(
        serialized_examples,
        {
            'noise': tf.FixedLenFeature([training_params.noise_size],
                                        tf.float32),
            # For uniformity of saved model interfaces, we always require 200
            # continuous and 200 categorical variables, and trim it to the right
            # length later on. Makes working with many models in ipynb easier.
            'structured_continuous_input': tf.FixedLenFeature(
              [200], tf.float32, default_value=[-1e9]*200),
            'structured_categorical_input': tf.FixedLenFeature(
              [200], tf.int64, default_value=[int(-1e9)]*200),
        }
    )
    noise = features['noise']
    structured_cont = None
    if training_params.infogan_cont_num_vars:
      structured_cont = features['structured_continuous_input'][
        :,:training_params.infogan_cont_num_vars]
    structured_cat = None
    if training_params.infogan_cat_num_vars:
      structured_cat = features['structured_categorical_input'][
        :,:training_params.infogan_cat_num_vars]
      structured_cat = structured_cat % training_params.infogan_cat_dim
    generator_inputs = {
        'noise': features['noise'],
        'structured_continuous_input': structured_cont,
        'structured_categorical_input': structured_cat,
        # TODO: Get rid of labels here.
        'one_hot_labels': tf.constant(
            0., shape=[training_params.batch_size, 10])
    }

    global_step = tf.train.get_or_create_global_step()
    phase, phase_progress = train._get_phase(training_params, global_step)
    # The variable scope mimics the behavior of tfgan.gan_train. Needed for
    # the saver below to see the proper variable names.
    scope = ('gpu_tower/Generator' if training_params.use_gpu_tower_scope
             else 'Generator')
    with tf.variable_scope(scope):
      generated_images = networks.generator_fn(
        generator_inputs=generator_inputs,
        training_params=training_params,
        is_training=False,
        phase=phase,
        phase_progress=phase_progress,
      )

    session_config = tf.ConfigProto(
      allow_soft_placement=True,
    )
    with tf.Session(config=session_config) as sess:
      if export_with_ema_params:
        tmp_params_ema = tf.train.ExponentialMovingAverage(
          decay=training_params.generator_params.ema_decay_for_visualization)
        variables_to_restore = tmp_params_ema.variables_to_restore()
        #  tf.logging.info('variables_to_restore: ' + str(variables_to_restore))
        saver_for_restore = tf.train.Saver(variables_to_restore)
      else:
        saver_for_restore = saver.Saver()

      if checkpoint_paths:
        _restore_average_from_multiple_checkpoints(
          saver_for_restore,
          sess,
          checkpoint_paths,
        )
        checkpoint_path = '<multiple pahts>'
      else:
        checkpoint_path = saver.latest_checkpoint(training_params.output_dir)
        assert checkpoint_path
        saver_for_restore.restore(sess, checkpoint_path)

      tmp_dir = '/tmp/tmpdir_export_%d' % int(100*time.time())
      builder = tf.saved_model.builder.SavedModelBuilder(tmpexport_dir)
      signature_def_map = {
          'generate': tf.saved_model.signature_def_utils.predict_signature_def(
              {'serialized_examples': serialized_examples},
              {'images': generated_images},
          ),
      }
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=signature_def_map)
      builder.save(as_text=True)
  tf.gfile.Rename(tmpexport_dir, export_dir)
  tf.logging.info('Exported saved model (%s) to %s',
                  checkpoint_path.rpartition('/')[-1],
                  export_dir)
  return export_dir


def _maybe_restore_from_path(sess, path, vars_to_restore=None,
                             checkpoint_num=None):
  tf.logging.info('maybe_restore from %s', path)
  saver_for_restore = saver.Saver(var_list=vars_to_restore)
  if checkpoint_num:
    checkpoint_path = os.path.join(path,
                                   'model.ckpt-%06d' % checkpoint_num)
  else:
    checkpoint_path = saver.latest_checkpoint(path)
  if checkpoint_path:
    tf.logging.info('Restoring checkpoints from %s', checkpoint_path)
    saver_for_restore.restore(sess, checkpoint_path)
  else:
    tf.logging.info("No checkpoints found.")


def _handle_legacy_vars_to_restore(vars_to_restore, path):
  available_vars = tf.contrib.framework.list_variables(path)
  available_vars = {var for var, shape in available_vars}

  old_vars_to_restore = vars_to_restore
  vars_to_restore = dict()
  for var in old_vars_to_restore:
    name = var.op.name
    new_name = name[10:]
    if (name not in available_vars
        and name.startswith('gpu_tower/')
        and new_name in available_vars):
      tf.logging.info('vars_to_restore: renaming %s to %s' % (name, new_name))
      vars_to_restore[new_name] = var
    elif name in available_vars:
      vars_to_restore[name] = var
    else:
      tf.logging.info('Missing variable: %s', name)

  return vars_to_restore


def maybe_restore(sess, training_params=None, vars_to_restore=None,
                  allow_partial_restore=False, path=None,
                  checkpoint_num=None):
  if path is None:
    path = training_params.output_dir
  if vars_to_restore is None:
    vars_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

  if not tf.gfile.Exists(os.path.join(path,
                                      'checkpoint')):
    tf.logging.info('No checkpoints available. Nothing restored.')
    return

  new_vars_to_restore = _handle_legacy_vars_to_restore(vars_to_restore,path)

  if len(new_vars_to_restore) == len(vars_to_restore):
    _maybe_restore_from_path(sess, path, new_vars_to_restore,
                             checkpoint_num=checkpoint_num)
  elif allow_partial_restore:
    found = len(new_vars_to_restore)
    tf.logging.info('\n\n\n\n\n\n')
    tf.logging.warning('NOTE: Not all variables found in checkpoint.' +
                       ' Will attempt partial restore.')
    tf.logging.info('Found %d / %d variables', found, len(vars_to_restore))
    tf.logging.info('\n\n\n\n\n\n')
    assert found > 0
    time.sleep(3)
    _maybe_restore_from_path(sess, path, new_vars_to_restore,
                             checkpoint_num=checkpoint_num)
  else:
    raise RuntimeError, ('Not all variables found in the checkpoint. ' +
                         'See logs above.')


def checkpoint(sess, cur_global_step, training_params):
  ckpt_path = os.path.join(training_params.output_dir,
                           'model.ckpt-%06d' % cur_global_step)
  if tf.gfile.Exists(ckpt_path):
    tf.logging.info('Checkpoint %s already exists.', checkpoint_path)
  else:
    checkpoint_path = tf.train.Saver().save(sess, ckpt_path)
    tf.logging.info('Exported checkpoint to %s', checkpoint_path)

  export_newest_savedmodel(training_params)
  if training_params.generator_params.ema_decay_for_visualization:
    export_newest_savedmodel(training_params,
                             export_with_ema_params=True)


def should_checkpoint(training_params, steps_passed, secs_passed):
  if (training_params.checkpoint_every_n_steps
      and steps_passed % training_params.checkpoint_every_n_steps == 0):
    return True
  if (training_params.checkpoint_every_n_secs
      and secs_passed >= training_params.checkpoint_every_n_secs):
    return True
  return False



def write_summaries(model_dir, sess, cur_global_step, summaries=None,
                    feed_dict=None, allow_failures=False):
  writer = tf.summary.FileWriter(
    model_dir,
    graph=sess.graph if cur_global_step < 10 else None)
  summaries = summaries or tf.get_collection(tf.GraphKeys.SUMMARIES)
  summaries = filter(sess.graph.is_fetchable, summaries)
  try:
    for x in sess.run(summaries, feed_dict=feed_dict):
        writer.add_summary(x, global_step=cur_global_step)
  except Exception, e:
    if allow_failures:
      tf.logging.warning('Exception when writing summary: %s', str(e))
    else:
      raise
  writer.flush()
  writer.close()


def _extract_step(path):
  match = re.match(r'.*model.ckpt-([0-9]*).meta', path)
  assert match, 'Failed to extract checkpoint step from path %s' % path
  return int(match.group(1))


def _newest_checkpoints(training_params, num_checkpoints=5,
                        ignore_checkpoints_after=None):
  all_checkpoints = sorted(tf.gfile.Glob(
    os.path.join(training_params.output_dir, 'model.ckpt-*.meta')))
  if ignore_checkpoints_after:
    all_checkpoints = [x for x in all_checkpoints
                       if _extract_step(x) <= ignore_checkpoints_after]
  all_checkpoints = map(lambda x: x[:-5], all_checkpoints)
  return all_checkpoints[-num_checkpoints:]


def main(*args):
  module = importlib.import_module(FLAGS.experiment_module)
  training_params = module.training_params()
  with util.TFFileLogger(training_params.output_dir):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info('export_utils, flags: %s', str(FLAGS.__flags))
    tf.logging.info('Training params:\n' + str(training_params))
    if FLAGS.ignore_checkpoints_after:
      assert FLAGS.num_newest_checkpoints, ('ignore_checkpoints_after only '
                                            'works in combination with '
                                            'num_newest_checkpoints')
    if FLAGS.num_newest_checkpoints:
      checkpoint_paths = _newest_checkpoints(training_params,
                                             FLAGS.num_newest_checkpoints,
                                             FLAGS.ignore_checkpoints_after)
      tf.logging.info('checkpoint_paths:\n%s', '\n'.join(checkpoint_paths))
    else:
      checkpoint_paths=None

    export_newest_savedmodel(
      training_params,
      export_with_ema_params=FLAGS.export_with_ema_params,
      checkpoint_paths=checkpoint_paths,
      subdir=FLAGS.subdir)


if __name__ == '__main__':
  _define_flags()
  tf.app.run()
