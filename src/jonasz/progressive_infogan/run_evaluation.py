import os
import datetime
from jonasz.progressive_infogan import create_animation
import importlib
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
from jonasz.progressive_infogan import train
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


def _find_phase(training_params, checkpoint_num):
  for phase in sorted(training_params.dynamic_steps_per_phase.keys()):
    tp_phase = train._specialize_training_params_for_phase(
      training_params, phase)

    if tp_phase.max_steps >= checkpoint_num:
      return phase
  assert False, 'Couldnt find the phase corresponding to checkpoint num %d' %( 
    checkpoint_num)


@util.tf_logging_decorator
def _run_evaluation_with_logging(training_params, step, original_output_dir):
  phase = _find_phase(training_params, step)
  tf.logging.info('Found the phase corresponding to step %s: %s',
                  step, phase)
  tf.logging.info('Downloading the checkpoint for step %d', step)
  train._download_checkpoint(original_output_dir,
                             training_params.output_dir,
                             num_checkpoint=step)
  tp_phase = train._specialize_training_params_for_phase(training_params, phase)
  del training_params
  tf.logging.info('Running evaluation.')
  train._evaluation(tp_phase)
  tf.logging.info('Exporting savedmodel.')
  export_utils.export_newest_savedmodel(tp_phase, export_with_ema_params=True)


def get_output_dir(output_dir_base, job_id, step):
  return os.path.join(output_dir_base,
                      job_id.replace('/', '_') + '__step%d' % step)


def run_evaluation(training_module_name, job_id, step, output_dir_base,
                   eval_params=None):
  """
  Params:
    training_module_name: (str) 'jonasz.experiments.<date>.<name>'
    job_id: (str) either gcloud job id, or local path relative to
          constants.TRAINING_OUTPUT_BASE_DIR
    step: identifies the checkpoint to load for evaluation.
    output_dir_base: where to save evaluation results (and tmp files)
    eval_params: Optional
  """
  if job_id.startswith('job___'):
    original_output_dir = 'gs://seer-of-visions-ml2/%s' % job_id
  else:
    original_output_dir = os.path.join(constants.TRAINING_OUTPUT_BASE_DIR,
                                       job_id)

  output_dir = get_output_dir(output_dir_base, job_id, step)
  success_file = os.path.join(output_dir, 'run_evaluation_success')
  if tf.gfile.Exists(success_file):
    print 'File run_evaluation_success exists, returning', success_file
    return

  module = importlib.import_module(training_module_name)
  training_params = module.training_params(
    output_dir=output_dir,
  )
  training_params.overwrite(
    continue_from=None,  # Not really necessary, I think.
  )
  if eval_params:
    training_params.overwrite(
      eval_params=eval_params
    )
  _run_evaluation_with_logging(training_params, step, original_output_dir)

  with tf.gfile.Open(success_file, 'w') as f:
    f.write('%s' % (str(datetime.datetime.now())))


# Note: currently creates animation for the newest savedmodel, not for a given
# step.
