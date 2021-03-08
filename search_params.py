import argparse
import os
from subprocess import check_call
import sys

import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/base_model', help='Directory containing params.json')
parser.add_argument('--input_dir', default='data/from_kaggle', help="Directory containing the dataset")

def launch_training_job(parent_dir, input_dir, train_file, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        input_dir: (string) directory containing the dataset
        train_file: (string) file for training 
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    fmt = "{python} {train} --model_dir={model_dir} --input_dir {input_dir}"
    cmd = fmt.format(python=PYTHON, train=train_file, model_dir=model_dir, input_dir=input_dir)

    print(cmd)
    check_call(cmd, shell=True)

if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # Model files
    #files = ['1d-cnn-train.py', 'dnn-train.py', 'tabnet-train.py']
    files = ['1d-cnn-train.py']

    # Perform hypersearch over one parameter
    learning_rates = [1e-4, 1e-3, 1e-2]
    ncompo_genes_values = [50, 250, 500]
    smoothing_values = [0, 0.001, 0.005]

    for train_file in files:

      for smoothing in smoothing_values:
          params.smoothing = smoothing
          job_name = "smoothing_{}".format(smoothing)
          launch_training_job(args.parent_dir, args.input_dir, train_file, job_name, params)
    
      if (0): 
        for ncompo_genes in ncompo_genes_values:
            params.ncompo_genes = ncompo_genes
            job_name = "ncompo_genes_{}".format(ncompo_genes)
            launch_training_job(args.parent_dir, args.input_dir, train_file, job_name, params)

        for learning_rate in learning_rates:
            params.learning_rate = learning_rate
            job_name = "learning_rate_{}".format(learning_rate)
            launch_training_job(args.parent_dir, args.input_dir, train_file, job_name, params)


