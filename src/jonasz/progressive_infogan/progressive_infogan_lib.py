import matplotlib
import tensorflow as tf
import numpy as np
import random
from jonasz.lib import util
import matplotlib.animation
import matplotlib.pyplot as plt
import PIL, PIL.Image, PIL.ImageDraw
import multiprocessing

class GenInput(object):

  def __init__(self, noise, cont_structured_input, cat_structured_input):
    self.noise = noise
    self.cont_structured_input = cont_structured_input
    self.cat_structured_input = cat_structured_input

  @classmethod
  def random(cls, noise_stddev):
    return cls(
      noise=cls._create_noise(64, noise_stddev),
      cont_structured_input=cls._create_noise(200, noise_stddev),
      cat_structured_input=cls._create_cat_structured_input(),
    )

  def copy(self):
    return self.__class__(
      noise=self.noise.copy(),
      cont_structured_input=self.cont_structured_input.copy(),
      cat_structured_input=self.cat_structured_input.copy(),
    )

  def as_dict_key(self):
    return (tuple(self.noise),
            tuple(self.cont_structured_input),
            tuple(self.cat_structured_input))

  @classmethod
  def _create_noise(cls, noise_size, noise_stddev):
    return noise_stddev*np.random.randn(noise_size)

  @classmethod
  def _create_cat_structured_input(cls):
    return np.random.randint(0, 10, size=200)

  def as_serialized_example(self):
    return util.serialized_example(
      {
        'noise': self.noise,
        'structured_continuous_input': self.cont_structured_input,
        'labels': []
      },
      {
        'structured_categorical_input': self.cat_structured_input,
      })


def random_request(num_images=None, noise_stddev=None, seed=None):
  assert num_images is not None
  assert noise_stddev is not None
  if seed is not None:
    np.random.seed(seed)
  return [GenInput.random(noise_stddev) for i in range(num_images)]


def _query_tf_model_server(gen_inputs,
                           model_id=None,
                           version=None,
                           signature='generate',
                           batch_size=None,
                           host_port=None):
  assert batch_size is not None
  assert model_id is not None
  assert len(gen_inputs) % batch_size == 0, len(gen_inputs)
  return util.tensorflow_model_server_predict(
    host_port=host_port,
    model_id=model_id,
    signature_name=signature,
    serialized_examples=[gi.as_serialized_example() for gi in gen_inputs],
    serialized_examples_tensor_name='serialized_examples',
    version=version,
    batch_size=batch_size,
  )


def gen_inputs_to_images(gen_inputs,
                         model_id=None,
                         version=None,
                         embed_in_random_request=False,
                         batch_size=None,
                         host_port=None,
                         noise_stddev=None):
  """
  Args:
    gen_inputs:
    model_id,
    version:
    embed_in_random_request: if True, each gen_input will be evaluated in a
      separate batch. Each batch will be identical, with the exception of the
      first item. This is useful if we are creating an animation for a model
      that uses batch norm or other batch-wise techniques.
    batch_size:
    host_port: Optional. TF Model Server.
    noise_stddev: Required when embed_in_random_request is True.

  Returns:
    A list of HWC np arrays.
  """
  assert batch_size is not None
  if embed_in_random_request:
    assert noise_stddev is not None
    np.random.seed(13)  # Let's make sure random_request is uniform across
                        # queries.
    random_batch = random_request(num_images=batch_size,
                                  noise_stddev=noise_stddev) * len(gen_inputs)
    print 'Got %d gen_inputs, embedding in random batch size %d' % (
      len(gen_inputs), len(random_batch))
    for i, gen_input in enumerate(gen_inputs):
      random_batch[i*batch_size] = gen_input
    gen_inputs = random_batch

  missing = (batch_size - len(gen_inputs) % batch_size) % batch_size
  if missing:
    #  tf.logging.info('Num inputs (%d) not divisible by batch size, adding %d '
    #                  'inputs.', len(gen_inputs), missing)
    gen_inputs += random_request(num_images=missing, noise_stddev=noise_stddev)


  result_dict = _query_tf_model_server(gen_inputs=gen_inputs,
                                       model_id=model_id,
                                       version=version,
                                       batch_size=batch_size,
                                       host_port=host_port)

  # Need to infer the resolution.
  resolution = int((len(result_dict['images']) / 3 / len(gen_inputs))**0.5)
  shape = [len(gen_inputs), resolution, resolution, 3]
  res = result_dict['images'].reshape(shape)
  res = list(res)
  if missing:
    res = list(res)[:-missing]
  if embed_in_random_request:
    return res[::batch_size]
  else:
    return res
