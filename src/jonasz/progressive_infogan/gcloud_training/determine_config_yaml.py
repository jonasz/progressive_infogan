#! /usr/bin/python

import tensorflow as tf
import importlib

def main(*args):
  module = args[0][1]
  output_dir = args[0][2]
  assert module
  assert output_dir
  experiment_module = importlib.import_module(module)
  training_params = experiment_module.training_params(is_gcloud=True,
                                                      output_dir=output_dir)
  if training_params.num_gpus == 1:
    print 'src/jonasz/progressive_infogan/gcloud_training/config.yaml'
  elif training_params.num_gpus == 4:
    print 'src/jonasz/progressive_infogan/gcloud_training/config_4gpus.yaml'
  elif training_params.num_gpus == 8:
    print 'src/jonasz/progressive_infogan/gcloud_training/config_8gpus.yaml'
  else:
    assert False, training_params.num_gpus

if __name__ == '__main__':
  tf.app.run()
