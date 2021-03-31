#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import traceback
import argparse
from sklearn.model_selection import ParameterGrid
from rxn_modeling.models.trainer import RxnTrainer


def main(config_file, output_path):
    os.makedirs(output_path, exist_ok=True)
    try:
        # Read in any hyperparameters that the user passed with the training job
        # Depending on how you set the hyperparameters

        try:
            with open(config_file, 'r') as tc:
                temp = json.load(tc)
                hp_id = int(temp['id'])
                config_file = temp['config_file']
                with open(config_file, 'r') as fd:
                    config = json.load(fd)
                training_params = ParameterGrid(config)[hp_id]
        except Exception as ee:
            with open(config_file, 'r') as tc:
                training_params = json.load(tc)

            if isinstance(training_params, list):
                grid = ParameterGrid(training_params)
                print(f'>> There are {len(grid)} experiments bundled together... Taking the first one.')
                training_params = grid[0]

        if isinstance(training_params, list):
            # assert len(training_params) == 1
            training_params = training_params[0]

        # the function below does all the data loading, run, validate and test the algo
        RxnTrainer.run_experiment(**training_params, output_path=output_path)

    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    # These are the paths where SageMaker mounts interesting things in your container.
    # by default we assume that only one channel of input data will be used and its name will be training.
    prefix = '/opt/ml/'
    output_path = os.path.join(prefix, 'model')
    config_file = os.path.join(prefix, 'input/config/hyperparameters.json')
    input_path = os.path.join(prefix, 'input/data/training')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config_file', type=str, default=config_file,
                        help="""The name/path of the config file (in json format) that contains all the
                            parameters for the experiment. This config file should be at the same
                            location as the current train file""")
    parser.add_argument('-i', '--input_path', type=str, default=None,
                        help="""location that contains the train files.""")
    parser.add_argument('-o', '--output_path', type=str, default=output_path,
                        help="""location for saving the training results (model artifacts and output files).
                                If not specified, results are stored in the folder "results" at the same level as  .""")
    args = parser.parse_args()

    main(config_file=args.config_file, output_path=args.output_path)
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
