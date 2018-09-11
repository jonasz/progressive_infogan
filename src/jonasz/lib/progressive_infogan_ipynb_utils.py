import tensorflow as tf
import numpy as np
import os
import scipy
import random
from jonasz.lib import util
from jonasz.cifar10 import cifar10_dataset
from jonasz import constants
from matplotlib import pyplot as plt
from jonasz.progressive_infogan import progressive_infogan_lib
import ipywidgets
import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import time

# To be filled in by the user.
# See src/notebooks/jonasz/notebooks/generator.ipynb
MODELS = { }


TF_MODEL_SERVER_SUBPROCESS = None
HOST_PORT = ('localhost', 9141)
def restart_server():
  global TF_MODEL_SERVER_SUBPROCESS, HOST_PORT
  if TF_MODEL_SERVER_SUBPROCESS is not None:
    print 'killing previous subprocess, pid', TF_MODEL_SERVER_SUBPROCESS.pid
    TF_MODEL_SERVER_SUBPROCESS.kill()
  #  HOST_PORT = ('localhost', random.randint(8000, 10000))
  models_to_versions = {
    model: options.get('versions', []) for model, options in MODELS.items()
  }

  print 'HOST_PORT', HOST_PORT
  for key, val in models_to_versions.items(): print key, val
  TF_MODEL_SERVER_SUBPROCESS = util.start_tensorflow_model_server(
    port=HOST_PORT[1],
    models=models_to_versions,
  )
  print 'tensorflow model server running, pid', TF_MODEL_SERVER_SUBPROCESS.pid


def get_images(model, gen_inputs):
  batch_size = MODELS[model]['batch_size']
  imgs = progressive_infogan_lib.gen_inputs_to_images(gen_inputs,
                                                      model_id=model,
                                                      batch_size=batch_size,
                                                      host_port=HOST_PORT,
                                                      noise_stddev=1.)
  return imgs



def manual_animation(model, gen_inp, coords=range(64,80)):
  _input = copy.deepcopy(gen_inp)


  fig = plt.figure(figsize=(9, 9))
  ax = fig.add_subplot(111)
  imshow_obj = [None]
  #  last_update = time.time()

  def f(**kwargs):
    coords = sorted(kwargs.items())
    c = [val for key, val in coords]
    print c
    inp = copy.deepcopy(_input)
    for key, val in coords:
      coord = int(key[1:])  # key is 'c29'
      inp.cont_structured_input[coord] = val
    img = memoized_get_img(model, inp)
    img = (img + 1.) / 2
#     img = np.clip(img, -1., 1.)


    #  global imshow_obj, last_update
    #  elapsed = time.time() - last_update
    if imshow_obj[0] is None:
      imshow_obj[0] = ax.imshow(img)  # ,minv=-1., maxv=1., interpolation='none')
      plt.show()
  #   elif elapsed > 0.1:
    else:
      #  last_update = time.time()
      imshow_obj[0].set_data(img)
      plt.show()

  # for i in range(num_coords): print _noise[i]
  _kwargs = collections.OrderedDict([
      ('c%02d' % i,
       ipywidgets.FloatSlider(value=_input.cont_structured_input[i],
                              min=-2.6, max=2.6, step=0.2)) #, continuous_update=False))
       for i in coords
  ])
  interactive_plot = ipywidgets.interactive(f, **_kwargs)
  output = interactive_plot.children[-1]
  # output.layout.height = '250px'
  return interactive_plot


def interpolate(model, inp, coord, steps=3):
  inps = []
  for val in np.linspace(-3., 3., steps):
    cur_inp = inp.copy()
    cur_inp.cont_structured_input[coord] = val
    inps.append(cur_inp)
  imgs = get_images(model, inps)
  return imgs


def blur(a, stddev):
  a = a.copy()
  for channel in range(3):
    a[:,:,channel] = scipy.ndimage.filters.gaussian_filter(
      a[:,:,channel], stddev)
  return a


def diff2(a, b):
  a = a.copy()
  b = b.copy()
  res = b-a
  res = np.abs(res)
  res = blur(res, 1.)
  return res

def avg_change(model, coord,
							 percentile_thresholds=[99.5, 99.,98., 95.,90., 80.],
               num_samples=30,
               inputs_for_interpolation=[], interpolation=None):
  gen_inputs = progressive_infogan_lib.random_request(
    num_images=num_samples, noise_stddev=1.)
  left, mid, right = [], [], []
  for gi in gen_inputs:
    gi_left, gi_right = gi.copy(), gi.copy()
    gi_left.cont_structured_input[coord] = -2.5
    gi_right.cont_structured_input[coord] = 2.5
    mid.append(gi)
    left.append(gi_left)
    right.append(gi_right)

  imgs_left = get_images(model, left)
  imgs_mid = get_images(model, mid)
  imgs_right = get_images(model, right)

  diffs = [diff2(a, b) for (a, b) in zip(imgs_left, imgs_right)]
  bigleft = sum(imgs_left) / len(imgs_left)
  bigright = sum(imgs_right) / len(imgs_right)
  bigmid = sum(imgs_mid) / len(imgs_mid)

  bigdiff_rgb = sum(diffs) / len(diffs)
  bigdiff_norm = np.linalg.norm(bigdiff_rgb, axis=2, keepdims=True)

  bigchange_imgs = []
  for percentile_threshold in percentile_thresholds:
    thresh = np.percentile(bigdiff_norm, percentile_threshold)
    bigchange = np.where(bigdiff_norm > thresh, bigright, np.ones_like(bigright)-.2)
    bigchange_imgs.append(bigchange)



  util.show_imgs(bigchange_imgs, cols=len(bigchange_imgs), minv=-1., maxv=1.,
                 interpolation=interpolation)
  util.show_imgs([bigleft,
                  bigright,
                  bigmid],
                 cols=4, minv=-1., maxv=1, interpolation=interpolation)

  util.show_imgs(imgs_left[:5], minv=-1., maxv=1., cols=5,
                 interpolation=interpolation)
  util.show_imgs(imgs_right[:5], minv=-1., maxv=1., cols=5,
                 interpolation=interpolation)
  for inp in inputs_for_interpolation:
    util.show_imgs(interpolate(model, inp, coord, steps=5),
                   minv=-1., maxv=1., cols=5,
                   interpolation=interpolation)
