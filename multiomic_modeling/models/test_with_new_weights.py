import argparse
import os
import random
import numpy as np
from argparse import Namespace

import optuna
from optuna.study import StudyDirection
from packaging import version
from multiomic_modeling.models.trainer import MultiomicTrainer
from multiomic_modeling.models.base import PatientPruner

import pytorch_lightning as pl
import torch

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")
def temp():
    model_params = {
        "d_input_enc": 2000, 
        "lr": 6.033193735866575e-05,
        "nb_classes_dec": 33,
        "early_stopping": True,
        "dropout": 0.16171970479206027,
        "weight_decay": 5.4598394312421854e-05,
        "activation": "relu",
        "optimizer": "Adam",
        "lr_scheduler": "cosine_with_restarts",
        "loss": "ce",
        "n_epochs": 500, 
        "batch_size": 256,
        "class_weights":[4.03557312, 0.85154295, 0.30184775, 1.18997669, 8.25050505,
                0.72372851, 7.73484848, 1.81996435, 0.62294082, 0.61468995,
                4.07992008, 0.49969411, 1.07615283, 1.85636364, 0.7018388 ,
                0.84765463, 0.60271547, 0.62398778, 4.26750261, 0.61878788,
                1.89424861, 1.98541565, 0.65595888, 2.05123054, 1.37001006,
                0.77509964, 0.76393565, 2.67102681, 0.64012539, 2.94660895,
                0.64012539, 6.51355662, 4.64090909],
        "d_model_enc_dec": 512,
        "n_heads_enc_dec": 16,
        "n_layers_enc": 10,
        "n_layers_dec": 1
    }
    d_ff_enc_dec_value = model_params["d_model_enc_dec"] * 4
    model_params["d_ff_enc_dec"] = d_ff_enc_dec_value

    fit_params = {
        "nb_ckpts":1, 
        "verbose":1
    }

    predict_params = {
        "nb_ckpts":1, 
        "scores_fname": "transformer_scores.json"
    }

    training_params = {
        "model_params": model_params,
        "fit_params": fit_params,
        "predict_params": predict_params,
        "data_size": int('2000'),
        "dataset_views_to_consider": 'all',
        "seed": 42
    }
    
    model = MultiomicTrainer.run_experiment(**training_params, output_path='/home/maoss2/scratch')
    
if __name__ == '__main__':
    temp()