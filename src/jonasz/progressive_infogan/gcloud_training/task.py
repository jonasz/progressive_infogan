import argparse
import tensorflow as tf
from jonasz.progressive_infogan import train
import importlib


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--experiment_module',
      help='Experiment module to take training_params from.',
      required=True
  )
  parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='unused',
      default='',
  )
  args = parser.parse_args()
  arguments = args.__dict__
  output_dir = arguments.pop('output_dir')

  experiment_module = importlib.import_module(arguments['experiment_module'])
  training_params = experiment_module.training_params(is_gcloud=True,
                                                      output_dir=output_dir)
  train.run_training(training_params)
