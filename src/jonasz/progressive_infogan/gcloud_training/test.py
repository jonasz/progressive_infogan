import tensorflow as tf
import importlib

def main(*args):
  print args
  module = args[0][1]
  output_dir = args[0][2]
  assert module
  assert output_dir
  experiment_module = importlib.import_module(module)
  training_params = experiment_module.training_params(is_gcloud=True,
                                                      output_dir=output_dir)
  print str(training_params)

if __name__ == '__main__':
  tf.app.run()
