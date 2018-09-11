"""Miscellaneous utilities."""

import tempfile
import traceback
import logging
import subprocess
import signal
import collections
import datetime
import numpy as np
import os
import re
import random
import time
import copy
import grpc
from grpc.beta import implementations as grpc_beta_implementations
from grpc._cython import cygrpc
import tensorflow as tf
from jonasz import constants


def CHW_to_HWC(img):
  """Converts a np arrray from shape [c, h, w] to [h, w, c]."""
  c, h, w = img.shape
  new_img = img.reshape([c, h*w]).T
  new_img = new_img.reshape([h, w, c])
  return new_img

def imshow(img, minv=None, maxv=None, turn_off_axis=True):
  from matplotlib import pyplot as plt  # Import here so we don't have to list
                                        # this as a requirement in setup.py.
  assert minv is not None  # Otherwise we distort the images.
  assert maxv is not None
  minv = minv or min(img.flatten())
  maxv = maxv or max(img.flatten())
  img = (img - minv) / (maxv - minv)

  plt.tight_layout(pad=0., h_pad=0., w_pad=0.)
  fig = plt.imshow(img, vmin=0., vmax=1., interpolation='none')
  if turn_off_axis:
    # This is useful for the purpose of creating animations, but somehow causes
    # problems when working in a notebook. (Some imgs are not displayed.)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axis('off')
  return fig


def show_imgs(imgs, data_type='HWC', cols=5, minv=None, maxv=None,
              size=15):
  from matplotlib import pyplot as plt  # Import here so we don't have to list
                                        # this as a requirement in setup.py.
  assert data_type in ('HWC', 'CHW')
  if not isinstance(imgs, (list, tuple)):
    imgs = [imgs]

  n = len(imgs)
  rows = (n + cols-1) / cols
  siz = float(size) / cols
  plt.figure(figsize=(siz*cols, siz*rows))
  for i, img in enumerate(imgs):
    size = 1
    if data_type == 'CHW':
      img = CHW_to_HWC(img)
    plt.subplot(rows, cols, i+1).axis('off')
    imshow(img, minv=minv, maxv=maxv, turn_off_axis=False)
  plt.show()


def random_crop(img, max_shift=7, can_flip=False, data_type='CHW'):
    assert data_type == 'CHW'
    c, h, w = img.shape
    shift_h = random.randint(-max_shift, max_shift)
    shift_w = random.randint(-max_shift, max_shift)
    cropped = np.zeros_like(img)

    i_a_0 = max(0, shift_h)
    i_x_0 = max(0, -shift_h)
    i_a_1 = min(h, h+shift_h)
    i_x_1 = min(h, h-shift_h)

    j_a_0 = max(0, shift_w)
    j_x_0 = max(0, -shift_w)
    j_a_1 = min(w, w+shift_w)
    j_x_1 = min(w, w-shift_w)

    for channel in range(c):
        cropped[channel, i_a_0:i_a_1, j_a_0:j_a_1] = (
            img[channel, i_x_0:i_x_1, j_x_0:j_x_1])

    if can_flip and random.randint(0, 1)==1:
        cropped = np.flip(cropped, 2)

    return cropped


class Params(object):
  def __init__(self, **kwargs):
    params = self.get_allowed_params_with_defaults()
    assert_set_covered(kwargs, params)
    params.update(kwargs)
    self.params = params

    self.validate()

  def validate(self):
    pass

  def overwrite(self, **kwargs):
    self.params.update(kwargs)

  def __getattr__(self, name):
    return self.params[name]

  def default_param(self, key, val):
    self.params[key] = self.params.get(key, val)

  def __getstate__(self):
    return self.params

  def __setstate__(self, vals):
    self.params = vals.copy()

  def __repr__(self):
    res = self.__class__.__name__ + '(\n'
    for key, val in sorted(self.params.items()):
      cur = '%s=%r,' % (key, val)
      cur = ('  ' + t for t in cur.split('\n'))
      cur = list(filter(None, cur))
      cur = '\n'.join(cur) + '\n'
      res += cur
    res += ')'
    return res

  def get_allowed_params_with_defaults(self):
    return {'x': 2, 'y': 3}
    raise NotImplementedError

  def __deepcopy__(self, memo):
    return self.__class__(**copy.deepcopy(self.params))


TIME_START = None
def reset_timer():
    global TIME_START
    TIME_START = time.time()

def time_elapsed(log=True):
    elapsed = int(time.time() - TIME_START)
    hours = elapsed / 60 / 60
    minutes = (elapsed / 60) % 60
    seconds = elapsed % 60
    if log:
      print ('time elapsed: %d hours %d minutes %d seconds' % (
          hours, minutes, seconds))
    return elapsed


def assert_set_covered(covered, covered_by):
  unexpected = set(covered) - set(covered_by)
  assert not unexpected, unexpected


def get_time_str(microseconds=False):
  fmt = '%y_%m_%d___%H_%M_%S'
  if microseconds: fmt += '___%f'
  return datetime.datetime.now().strftime(fmt)


def get_new_dir(base_path, suffix):
  return os.path.join(base_path, get_time_str() + '___' + suffix)


def serialized_example(float_features, int_features):
    example = tf.train.Example()
    for key, val in float_features.iteritems():
        example.features.feature[key].float_list.value.extend(
            np.array(val).reshape(-1))
    for key, val in int_features.iteritems():
        example.features.feature[key].int64_list.value.extend(
            np.array(val).reshape(-1))
    return example.SerializeToString()


def _chunks(iterable, chunk_size):
  first_unused = 0
  while first_unused < len(iterable):
    yield iterable[first_unused:first_unused+chunk_size]
    first_unused += chunk_size


class _DummyCtxMgr():
    def __init__(self, *args):
      self.args = args
    def __enter__(self):
      return self.args
    def __exit__(self, exc_type, exc_value, traceback):
      pass


def tensorflow_model_server_predict(host_port=None,
                                    model_id=None,
                                    signature_name=None,
                                    serialized_examples=None,
                                    serialized_examples_tensor_name=None,
                                    version=None,
                                    batch_size=256,
                                    retries=10):
  """Queries a tensorflow_model_server, potentially spawning it first.

  Args:
    host_port: tensorflow_model_server address. If None, will spawn a new
        tms process, and kill it afterwards.
    model_id: If host_port is given - the name of the model to query. Otherwise
        a path to the model saved_model dir.
    signature_name: model's signature to query against.
    serialized_exapmles: list of serialized tf.train.Example.
    serialized_examples_tensor_name: when querying the model, supply the
        serialized tf exmaples into a tensor with this name.
    version: saved_model version or None. None means: query the newest
        available version.
    batch_size: if number of serialized examples exceeds batch_size, the model
        will be queried multiple times, each time with at most batch_size
        examples.
    retries: it takes some time for the tms to spawn and load the model. We
        do a couple retries, each time waiting for 2 secs.

  """
  from tensorflow_serving.apis import predict_pb2
  from tensorflow_serving.apis import prediction_service_pb2

  if host_port is None:
    versions = [] if version is None else [version]
    models = {model_id: versions}
    mgr = TFModelServer(models=models)
  else:
    mgr = _DummyCtxMgr(None, host_port)

  trial_num = 0
  with mgr as (popen, (host, port)):
    while trial_num < retries:
      if trial_num >= 2:
        print '(Re)trying (%d) to query tensorflow_model_server.' % trial_num

      if popen is not None and popen.poll() is not None:
        raise RuntimeError(
          'tensorflow_model_server exited with code %d' % popen.returncode)

      trial_num += 1
      try:
        channel = grpc_beta_implementations.Channel(grpc.insecure_channel(
          target='%s:%s' % (host, port),
          options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                   (cygrpc.ChannelArgKey.max_receive_message_length, -1)]))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        results = collections.defaultdict(list)
        for i, chunk in enumerate(_chunks(serialized_examples, batch_size)):
          if i > 9 and i % 10 == 0:
            print 'Prediction for batch %d / %d' % (
                i, (len(serialized_examples) + batch_size - 1) / batch_size)
          request = predict_pb2.PredictRequest()
          request.model_spec.name = model_id
          if version is not None:
            request.model_spec.version.value = version
          request.model_spec.signature_name = signature_name
          request.inputs[serialized_examples_tensor_name].CopyFrom(
              tf.contrib.util.make_tensor_proto(chunk))

          result_future = stub.Predict.future(request, 15.0)
          while result_future.done() is False:
            time.sleep(0.01)
          result = result_future.result()
          for key, tensor in result.outputs.iteritems():
              results[str(key)] = np.concatenate([
                  np.array(results[str(key)]),
                  np.array(tensor.float_val),
                  np.array(tensor.double_val),
                  np.array(tensor.int_val),
                  np.array(tensor.string_val),
                  np.array(tensor.int64_val),
                  np.array(tensor.string_val),
              ])
        return results
      except Exception, e:
        if trial_num > 3:
          print 'Failed to query tensorflow_model_server:', e
        time.sleep(2.)

    raise Exception('Failed to query tensorflow_model_server. Double check ' +
                    'the model you passed in exists.')


def _create_model_config_file(models):
  """
  Args:
    models: dict from model path to list of versions (potentially empty)
  """
  config_file = tempfile.NamedTemporaryFile(delete=False)
  config_file.write('model_config_list: {\n')
  for path, versions in models.items():
    config_file.write("""
    config: {
      name: '%s',
      base_path: '%s',
      model_platform: 'tensorflow'
    """ % (path, path))

    if versions:
      config_file.write("""
          model_version_policy: {
            specific: {
      """)
      for version in versions:
        config_file.write("versions: %d\n" % version)
      config_file.write("}}\n")

    config_file.write("},")

  config_file.write('}\n')
  config_file.close()
  print config_file.name
  return config_file.name


def start_tensorflow_model_server(port=8100, models=None, interruptible=False):
  """
  Args:
    models: dict from model path to list of versions (potentially empty)
  """
  old_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES') or None
  if old_cuda_visible_devices:
    del os.environ['CUDA_VISIBLE_DEVICES']
  print 'starting tensorflow_model_server'
  model_config_file = _create_model_config_file(models)
  if not interruptible:
    preexec_fn = lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
  else:
    preexec_fn = None
  p = subprocess.Popen([
    'tensorflow_model_server',
    '--port=' + str(port),
    '--model_config_file=' + model_config_file,
    '--per_process_gpu_memory_fraction=0.7'],
		# Make sure interrupts in notebooks don't kill the server
    preexec_fn=preexec_fn,
  )
  if old_cuda_visible_devices:
    os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
  return p


class TFModelServer(object):
  def __init__(self, port=8100, models=None, initial_sleep=None,
               interruptible=False):
    """
    Args:
      models: dict from model path to list of versions (potentially empty)
    """
    self.port = port
    self.models = models
    self.subprocess = None
    self.initial_sleep = initial_sleep
    self.interruptible = interruptible

  def __enter__(self):
    self.subprocess = start_tensorflow_model_server(
      port=self.port, models=self.models, interruptible=self.interruptible)
    if self.initial_sleep:
      print 'Waiting for tensorflow_model_server to start'
      time.sleep(self.initial_sleep)
    return self.subprocess, ('localhost', self.port)

  def __exit__(self, exc_type, exc_value, traceback):
    self.subprocess.kill()


class _TFLoggingFilter(logging.Filter):
  forbidden_list = [
    'Summary name .* is illegal; using .* instead.',
  ]
  def filter(self, record):
    for forbidden in self.forbidden_list:
      if re.match(forbidden, record.msg):
        return False
    return True

class TFFileLogger(object):
  """ Context manager for saving tensorflow logs into file.

  Usage:

  with TFFileLogger(training_directory):
    run_training(...)
  """
  def __init__(self, dir_path, is_gcloud):
    self.dir_path = dir_path
    self.logging_handler = None
    self.is_gcloud = is_gcloud

  def __enter__(self):
    if not tf.gfile.Exists(self.dir_path):
      tf.gfile.MakeDirs(self.dir_path)
    self.tensorflow_logger = logging.getLogger('tensorflow')
    self.tensorflow_logger.addFilter(_TFLoggingFilter())

    if not self.is_gcloud:
      self.logging_handler = logging.FileHandler(
          os.path.join(self.dir_path, 'tensorflow.log'))
      self.logging_handler.setLevel(logging.DEBUG)
      self.tensorflow_logger.addHandler(self.logging_handler)

    formatter = logging.Formatter('@@@ [%(asctime)s %(name)s.%(levelname)s] %(message)s')
    for handler in self.tensorflow_logger.handlers:
      handler.setFormatter(formatter)

  def __exit__(self, *args):
    tf.logging.error('Exception reported by TFFileLogger:')
    tf.logging.error(traceback.format_exc())
    if self.logging_handler:
      self.tensorflow_logger.removeHandler(self.logging_handler)


def tf_logging_decorator(f):
  def new_f(training_params, *args, **kwargs):
    with TFFileLogger(training_params.output_dir,
                      is_gcloud=training_params.is_gcloud):
      tf.logging.set_verbosity(tf.logging.DEBUG)
      tf.logging.info('logging_decorator calling %s', f.__name__)
      tf.logging.info('Training params:\n' + str(training_params))
      f(training_params, *args, **kwargs)
  return new_f


def get_optimizer(params, scope, global_step=None):
  """Creates and returns a tf.train.Optimizer.

  Args:
    params.optimzier: (str, float)
    params.learning_rate_decay_rate: float or None
    params.learning_rate_decay_steps: int or None
    scope: str
  """
  if global_step is None:
    global_step = tf.train.get_or_create_global_step()
  with tf.variable_scope(scope):
    optimizer, initial_learning_rate = params.optimizer

    if params.learning_rate_decay_steps is None:
      assert params.learning_rate_decay_rate is None
      lr = initial_learning_rate
    else:
      assert params.learning_rate_decay_rate is not None
      lr = tf.train.exponential_decay(
          learning_rate=initial_learning_rate,
          global_step=global_step,
          decay_steps=params.learning_rate_decay_steps,
          decay_rate=params.learning_rate_decay_rate,
          staircase=True)
    tf.summary.scalar('learning_rate', lr)

    if optimizer == 'adam':
      return tf.train.AdamOptimizer(learning_rate=lr)
    elif optimizer=='adam_b0_b99':
      return tf.train.AdamOptimizer(learning_rate=lr, beta1=0., beta2=0.99,
                                    epsilon=1e-8)
    elif optimizer=='gdo':
      return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif optimizer=='rmsprop':
      return tf.train.RMSPropOptimizer(lr, decay=.9, momentum=0.1)
    else:
      assert False, 'Unknown opimizer: ' + optimizer


def construct_experiment_output_dir(fname):
  """
  Translates experiment's filename to output directory path, like so:
    From "jonasz/experiments/2018_06_17/post_training_infogan_01.py"
    To ("/home/jonasz/warcaby/jonasz/2018_06_17/"
        "post_training_infogan_01___18_06_18___17_37_11")
  """
  subdir = ''
  for i in [-2, -3]:  # The date dir can have a subdir with experiments.
    date =  fname.split('/')[i]
    if re.match('^\d\d\d\d_\d\d_\d\d$', date):
      break
    else:
      subdir = date
  assert re.match('^\d\d\d\d_\d\d_\d\d$', date), date
  experiment = fname.rpartition('/')[-1][:-3]
  assert re.match('^[a-z0-9_]*$', experiment), experiment
  return os.path.join(constants.TRAINING_OUTPUT_BASE_DIR, date, subdir,
                      experiment + '___'  + get_time_str()
                     )
