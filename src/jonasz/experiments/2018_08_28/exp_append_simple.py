DESCRIPTION = __file__ + """
For each block from 2 to 6, the structured code is appended to the block's
input.
"""
import os
import re
import tensorflow as tf
from jonasz.progressive_infogan import train
from jonasz.progressive_infogan import networks
from jonasz.lib import util
from jonasz import constants
from jonasz.gan import evaluation
from jonasz.nvidia_celeb import celeba_hq_dataset


def training_params(is_gcloud=False, output_dir=None):

  if not output_dir:
    output_dir=util.construct_experiment_output_dir(__file__)

  num_gpus = 1
  stop_after = 7
  dynamic_batch_size = {
      2: 128,
      3: 128,
      4: 64,
      5: 32,
      6: 16,
      7: 6,
      8: 3,
  }
  imgs_per_phase = 384000
  dynamic_steps_per_phase = {
      phase: max(imgs_per_phase / batch_size, 6000)
      for phase, batch_size in dynamic_batch_size.items()
  }
  dynamic_steps_per_phase[7] *= 2
  return train.TrainingParams(
    description=DESCRIPTION,
    is_gcloud                     = is_gcloud,
    num_gpus                      = num_gpus,
    dataset_params                = celeba_hq_dataset.get_dataset_params(
                                        is_gcloud=is_gcloud,
                                        crop_at_center=True),
    checkpoint_every_n_steps      = None,
    checkpoint_every_n_secs       = 2*60*60,
    dynamic_steps_per_phase       = dynamic_steps_per_phase,
    dynamic_batch_size            = dynamic_batch_size,
    stop_after                    = stop_after,
    eval_every_n_secs             = 48*60*60,
    write_summaries_every_n_steps = 700,
    infogan_summary_reps          = 0,
    output_dir                    = output_dir,
    allow_initial_partial_restore = True,
    noise_size                    = 64,
    noise_stddev                  = 1.,
    summary_grid_size             = 3,
    #  allow_simultaneous_steps     = False,

    infogan_cont_weight = 10.,
    infogan_cont_depth_to_num_vars = {
      2: 16,
      3: 16,
      4: 16,
      5: 16,
      6: 16,
      7: 0,
      8: 0,
    },

    generator_params=networks.GeneratorParams(
      #  debug_mode = True,
      channels_at_4x4             = 2048,
      #  channels_max                = 448,  # Works
      #  channels_max                = 496,  # Fails at phase 8
      channels_max                = 480,
      optimizer                   = ('adam_b0_b99', 0.0005),
      ema_decay_for_visualization = .999,
      weight_norm                 = 'equalized',
      norm                        = 'batch_norm_in_place',
      norm_per_gpu                = True,
      double_conv                 = True,
      conditioning                = False,
      infogan_input_method    = 'append',
    ),


    discriminator_params=networks.DiscriminatorParams(
      #  debug_mode = True,
      channels_at_2x2         = 4096,
      channels_max            = 512,
      conditioning            = False,
      optimizer               = ('adam_b0_b99', 0.0005),
      weight_norm             = 'equalized',
      norm                    = None,
      norm_per_gpu            = True,
      double_conv             = True,
      second_conv_channels_x2 = True,
      fromrgb_use_n_img_diffs = 2,
      #  elastic_block_input     = True,
      #  infogan_max_pool        = True,
    ),

    use_gpu_tower_scope = True,
  )

def main(*args):
  train.run_training(training_params())

if __name__ == '__main__':
  tf.app.run()
