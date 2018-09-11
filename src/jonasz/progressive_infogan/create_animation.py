#  python jonasz/progressive_infogan/create_animation.py \
#  --saved_model_path gs://seer-of-visions-ml2/job___2018_07_24___23_24_11___c7_mirror_04_cropped___4ef8ebe/saved_model_ema \
#  --export_path=/tmp/vid/vid3.mp4 \
#  --noise_stddev=1.0 \
#  --batch_size=16 \
#  --cont_coords=10,16,17,21,25,32,36 \
#  --tensorflow_model_server_port=8500 \
#  --grid_size=2
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import random
import gc
from jonasz.lib import util
import matplotlib.animation
import matplotlib.pyplot as plt
import PIL, PIL.Image, PIL.ImageDraw
import multiprocessing
from jonasz.progressive_infogan import progressive_infogan_lib

tf.flags.DEFINE_string('saved_model_path', None, 'The model to use.')
tf.flags.DEFINE_integer('version', -1, 'The model\'s version to use.')
tf.flags.DEFINE_float('noise_stddev', None, 'How to sample the input to G.')
tf.flags.DEFINE_integer('num_cont_coords', None,
                        'Will animate range(num_cont_coords) coords.')
tf.flags.DEFINE_string('cont_coords', None, 'Comma separated list of integers')
tf.flags.DEFINE_integer('num_cat_coords', None,
                        'Will animate range(num_cat_coords) coords.')
tf.flags.DEFINE_integer('batch_size', None,
                        'Batch size to use when querying the model.')
tf.flags.DEFINE_string('export_path', None,
                       'Where to save the animation. /path/to/vid.mp4')
tf.flags.DEFINE_integer('tensorflow_model_server_port', None,
                        'Optional, if you already have a running server.')
tf.flags.DEFINE_integer('grid_size', 1,
                        ('The animation wil consist of grid_size**2 images '
                         'arranged in a grid.'))
tf.flags.DEFINE_boolean('embed_requests_in_random_batch', True,
                        ('Should be true if generator uses batch norm (or '
                         'any batch-wise op) in training mode.'))
tf.flags.DEFINE_integer('coord_resolution', 32,
                        ('The range of possible values for each coord will be '
                         'divided into `coord_resolution` values.'))
tf.flags.DEFINE_integer('blink_num_times', 0,
                        ('For each coord, show the change by rapidly switching '
                         'between the most extreme values of the coord.'))
tf.flags.DEFINE_string('numpy_random_seeds', None,
                       ('e.g. 5,2,3 - each provided seed controls the '
                         'corresponding face.'))
tf.flags.DEFINE_integer('tensorflow_model_server_initial_sleep', None,
                        ('Sometimes it takes longer for server to start up.'))
tf.flags.DEFINE_integer('resolution', 800, 'Resolution of the resulting video.')
tf.flags.DEFINE_integer('coords_batch', None,
                        'Split the output vid into parts, each with # coords')
tf.flags.DEFINE_boolean('adjust_colors', False,
                        ('If true, we try to adjust all iamges in animation '
                         'so that the distribution of values in each channel '
                         'is the same'))
FLAGS = tf.flags.FLAGS



def map_channel(ch, ref):
  sorted_ch = np.sort(ch.flatten())
  sorted_ref = np.sort(ref.flatten())
  m = {x: y for (x, y) in zip(sorted_ch, sorted_ref)}
  res = np.array([m[val] for val in ch.flatten()])
  res = np.reshape(res, ch.shape)
  return res


def adjust_img_to_ref(img, ref):
  img = img.copy()
  for channel in range(img.shape[2]):
    img[:,:,channel]= map_channel(img[:,:,channel], ref[:,:,channel])
  return img


def _animate(imgs, interval=20, size=8, export_path=None, resolution=800):
  fps = 1000/interval
  tf.logging.info('_animate %d imgs', len(imgs))
  fig = plt.figure(figsize=(size, size))
  ims = [[util.imshow(img, minv=-1., maxv=1.)] for img in imgs]
  tf.logging.info('_animate: imshow finished')
  ani = matplotlib.animation.ArtistAnimation(
    fig, ims, interval=interval, blit=False, repeat_delay=2000)
  tf.logging.info('_animate: ArtistAnimation created')


  if export_path is not None:
    tf.logging.info('_animate: saving to file, %s', export_path)
    writer=matplotlib.animation.AVConvWriter(
      fps=fps,
      codec='libx264',
      extra_args=['-preset', 'superfast',
                  '-crf', '20',
                  '-movflags', '+faststart',
                 ],
    )
    dpi = resolution / size
    ani.save(export_path, writer=writer, dpi=dpi)
    tf.logging.info('_animate: ani.save finished')
    res = None
  else:
    from IPython.display import HTML
    res = HTML(ani.to_html5_video())
    tf.logging.info('_animate: html5video finished')
  fig.clf()
  plt.cla()
  plt.close()
  return res


def _write_on_img((img, text)):
    img_side = img.shape[0]
    txt = PIL.Image.fromarray(np.uint8((img+1.)*128.))
    draw = PIL.ImageDraw.Draw(txt)
    draw.text((1,1), str(text))

    img = np.array(txt.getdata())
    img = img.reshape(img_side, img_side, 3) / 128. - 1
    return img


def _animate_single_image(
    saved_model_path,
    noise_stddev,
    cat_coords=None,
    cont_coords=None,
    coord_resolution=32,
    host_port=None,
    batch_size=None,
    embed_in_random_request=True,
    gen_input=None,
    version=None,
    blink_num_times=0,
    adjust_colors=False,
  ):
  assert saved_model_path is not None
  assert noise_stddev is not None
  assert cat_coords is not None
  assert cont_coords is not None
  assert host_port is not None, 'Without running server itll take forever'
  assert batch_size is not None
  assert cont_coords or cat_coords
  assert not cat_coords

  def _eval_gen_inputs(gen_inputs):
    imgs = progressive_infogan_lib.gen_inputs_to_images(
      gen_inputs=gen_inputs,
      model_id=saved_model_path,
      version=version,
      embed_in_random_request=embed_in_random_request,
      batch_size=batch_size,
      host_port=host_port,
      noise_stddev=noise_stddev,
    )
    return imgs

  if gen_input is None:
    gen_input = progressive_infogan_lib.GenInput.random(noise_stddev=noise_stddev)
  if adjust_colors:
    ref_img = _eval_gen_inputs([gen_input])[0]
  else:
    ref_img = None

  def _imgs(coord, vals, amplitude):
    gen_inputs = []
    txts = []
    for val in vals:
      cur_gen_input = gen_input.copy()
      cur_gen_input.cont_structured_input[coord] = val
      gen_inputs.append(cur_gen_input)
      progress = (val / amplitude + 1.) / 2.
      progress_txt = '------++++++'[:int(round(12*progress, 0))]
      txts.append('%2d@%s' % (coord, progress_txt))
    imgs = _eval_gen_inputs(gen_inputs)
    if ref_img is not None:
      for i in range(len(imgs)):
        imgs[i] = adjust_img_to_ref(imgs[i], ref_img)
    return zip(gen_inputs, txts, imgs)


  out = []
  for coord in cont_coords:
    tf.logging.info('Calculating images for coord %d', coord)
    # TODO: maybe 3.*stddev is too much?
    amplitude = 3.*noise_stddev
    all_imgs = _imgs(coord,
                     np.linspace(-amplitude, amplitude, coord_resolution),
                     amplitude)
    right = all_imgs[coord_resolution/2:]
    left = all_imgs[:coord_resolution/2]

    for _ in range(blink_num_times):
      out.extend([left[0]] * 5)
      out.extend([right[-1]] * 5)

    out.extend([right[0]] * 3)
    out.extend(right)
    out.extend(list(reversed(right)))
    out.extend(list(reversed(left)))
    out.extend(left)
    out.extend([left[-1]] * 3)

  return zip(*out)


def _make_grid_img(imgs, grid_size):
  assert len(imgs) == grid_size**2
  lines = []
  for i in range(grid_size):
    lines.append(np.concatenate(imgs[i*grid_size:(i+1)*grid_size], axis=0))
  res = np.concatenate(lines, axis=1)
  return res

def create_animation(
    saved_model_path,
    noise_stddev,
    cat_coords=None,
    cont_coords=None,
    grid_size=1,
    coord_resolution=30,
    frame_duration=50,  # 20FPS
    host_port=None,
    export_path=None,
    batch_size=None,
    embed_in_random_request=True,
    numpy_random_seeds=[],
    version=None,
    resolution=800,
    blink_num_times=0,
    adjust_colors=False,
  ):
  assert 1 <= grid_size <= 5
  tf.logging.info('Cont coords: %s', cont_coords)
  tf.logging.info('Cat coords: %s', cat_coords)

  tf.logging.info('adjust_colors: %s', adjust_colors)

  # We create the initial images right after numpy.random.seed, and so the
  # faces we'll get depend solely on the seed, and not other parameters,
  # like number of times we call numpy.random.* in other places.
  gen_inputs = []
  for i in range(grid_size**2):
    seed, = numpy_random_seeds[i:i+1] or [random.randint(0, 1000000)]
    tf.logging.info('Video %d, numpy_random_seed %d', i, seed)
    np.random.seed(seed)
    gen_inputs.append(progressive_infogan_lib.GenInput.random(noise_stddev))

  all_imgs = []
  for i in range(grid_size**2):
    tf.logging.info('Animating image %d / %d', i, grid_size**2)
    _, txts, cur_imgs = _animate_single_image(
      saved_model_path=saved_model_path,
      noise_stddev=noise_stddev,
      cat_coords=cat_coords,
      cont_coords=cont_coords,
      coord_resolution=coord_resolution,
      host_port=host_port,
      batch_size=batch_size,
      embed_in_random_request=embed_in_random_request,
      gen_input=gen_inputs[i],
      version=version,
      blink_num_times=blink_num_times,
      adjust_colors=adjust_colors,
    )
    all_imgs.append(cur_imgs)
  del cur_imgs
  all_imgs = zip(*all_imgs)
  tf.logging.info('Created %d images', len(all_imgs))
  tf.logging.info('Composing images into a grid')
  # Memory efficient - overwrite as we modify.
  for i in range(len(all_imgs)):
    all_imgs[i] = _make_grid_img(all_imgs[i], grid_size)
  tf.logging.info('Writing on images')
  pool = multiprocessing.Pool(processes=8)
  all_imgs = pool.map(_write_on_img, zip(all_imgs, txts))
  pool.close()
  pool.join()

  tf.logging.info('Animating')
  ani = _animate(
    all_imgs,
    interval=frame_duration,
    export_path=export_path,
    resolution=resolution,
  )


def _make_batches(data, batch_size):
  if batch_size is None:
    batch_size = len(data)

  i = 0
  while True:
    start = i*batch_size
    end = start + batch_size
    if start >= len(data):
      break
    yield data[start:end]
    i += 1


def main(*args):
  with util.TFFileLogger('/tmp/create_animation', is_gcloud=True):
    cat_coords=list(reversed(range(FLAGS.num_cat_coords or 0)))
    assert cat_coords == [], cat_coords
    tf.logging.set_verbosity(tf.logging.DEBUG)
    assert FLAGS.saved_model_path
    assert FLAGS.export_path is not None
    if FLAGS.version >= 0:
      version = FLAGS.version
      versions = [version]
    else:
      version = None
      versions = []

    if FLAGS.tensorflow_model_server_port is None:
      mgr = util.TFModelServer(
                          port=random.randint(8000, 10000),
                          models={FLAGS.saved_model_path: versions},
                          initial_sleep=FLAGS.tensorflow_model_server_initial_sleep,
                          interruptible=True)
    else:
      port = FLAGS.tensorflow_model_server_port
      mgr = util._DummyCtxMgr(
        None, ('localhost', FLAGS.tensorflow_model_server_port))

    assert (FLAGS.num_cont_coords is None) != (FLAGS.cont_coords is None)
    if FLAGS.num_cont_coords:
      cont_coords = list(reversed(range(FLAGS.num_cont_coords or 0)))
    else:
      cont_coords = FLAGS.cont_coords.split(',')
      cont_coords = map(int, cont_coords)

    if FLAGS.numpy_random_seeds:
      numpy_random_seeds = map(int, FLAGS.numpy_random_seeds.split(','))
    else:
      numpy_random_seeds = []

    with mgr as (_, (host, port)):
      for i, cont_coords in enumerate(_make_batches(cont_coords,
                                                    FLAGS.coords_batch)):
        gc.collect()
        path, ext = FLAGS.export_path.rsplit('.', 1)
        cur_export_path = path + '_part%02d' % i + '.' + ext
        tf.logging.info('coords_batch = %d, first batch: %s. Exp path: %s',
                        FLAGS.coords_batch, cont_coords, cur_export_path)
        create_animation(
          saved_model_path=FLAGS.saved_model_path,
          version=version,
          noise_stddev=FLAGS.noise_stddev,
          cat_coords=[],
          cont_coords=cont_coords,
          grid_size=FLAGS.grid_size,
          host_port=(host, port),
          batch_size=FLAGS.batch_size,
          export_path=cur_export_path,
          embed_in_random_request=FLAGS.embed_requests_in_random_batch,
          coord_resolution=FLAGS.coord_resolution,
          numpy_random_seeds=numpy_random_seeds,
          resolution=FLAGS.resolution,
          blink_num_times=FLAGS.blink_num_times,
          adjust_colors=FLAGS.adjust_colors,
        )


if __name__ == '__main__':
  tf.app.run()
