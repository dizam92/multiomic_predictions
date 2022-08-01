import argparse
import os
import random
import numpy as np
from argparse import Namespace

import optuna
from optuna.study import StudyDirection
from packaging import version
from multiomic_modeling.models.trainer import MultiomicTrainer
# from multiomic_modeling.models.base import PatientPruner
from optuna.pruners import PatientPruner, MedianPruner

import pytorch_lightning as pl
import torch

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")


def objective(trial: optuna.trial.Trial, 
              d_input_enc: int, 
              dataset_views_to_consider: str, 
              data_size: int, 
              output_path: str,
              random_seed: int) -> float:
    """ Main fonction to poptimize with Optuna """
    model_params = {
        "d_input_enc": int(d_input_enc), 
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "nb_classes_dec": 33,
        "early_stopping": True,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5), # 0.1, 0.5
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-1, log=True), # 1e-8, 1e-2
        "activation": "relu",
        "optimizer": "Adam",
        "lr_scheduler": "cosine_with_restarts",
        "loss": "ce",
        "n_epochs": 500, # augmenter ca since i have more data
        "batch_size": 256,
        # "batch_size": trial.suggest_categorical("batch_size", [256, 512]), # [128, 256, 512]
        "class_weights":[4.03557312, 0.85154295, 0.30184775, 1.18997669, 8.25050505,
                0.72372851, 7.73484848, 1.81996435, 0.62294082, 0.61468995,
                4.07992008, 0.49969411, 1.07615283, 1.85636364, 0.7018388 ,
                0.84765463, 0.60271547, 0.62398778, 4.26750261, 0.61878788,
                1.89424861, 1.98541565, 0.65595888, 2.05123054, 1.37001006,
                0.77509964, 0.76393565, 2.67102681, 0.64012539, 2.94660895,
                0.64012539, 6.51355662, 4.64090909],
        "d_model_enc_dec": trial.suggest_categorical("d_model_enc_dec", [128, 256, 512]), # [32, 64, 128, 256, 512]
        "n_heads_enc_dec": trial.suggest_categorical("n_heads_enc_dec", [8, 16]), # fixed heads
        "n_layers_enc": trial.suggest_categorical("n_layers_enc", [2, 4, 6]), # [2, 4, 6, 8, 10, 12]
        "n_layers_dec": trial.suggest_categorical("n_layers_dec", [1, 2]) # [1, 2, 4, 6]
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
        "data_size": int(data_size),
        "dataset_views_to_consider": dataset_views_to_consider,
        "exp_type": "normal",
        "seed": int(random_seed)
    }

    model = MultiomicTrainer.run_experiment(**training_params, output_path=output_path)
    # return model.trainer.callback_metrics["val_multi_acc"].item()
    retour = model.trainer.callback_metrics["val_ce"].item()
    return retour


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna version of Transformer model.")
    parser.add_argument('-d', '--d_input_enc', type=int, default=2000)
    parser.add_argument('-d_view', '--dataset_views_to_consider', type=str, default='all')
    parser.add_argument('-o', '--output_path', type=str, default='/home/maoss2/scratch/optuna_test_output_2000')
    parser.add_argument('-s', '--data_size', type=int, default=2000)
    parser.add_argument('-db_name', '--db_name', type=str, default='experiment_data_2000')
    parser.add_argument('-study_name', '--study_name', type=str, default='experiment_data_2000')
    parser.add_argument('-seed', '--seed', type=int, default=42)
    args = parser.parse_args()
    assert args.d_input_enc == args.data_size, 'must be the same size'
    if os.path.exists(args.output_path): pass
    else: os.mkdir(args.output_path)
    
    storage_db = optuna.storages.RDBStorage(
                url=f"sqlite:///{args.output_path}/{args.db_name}_{args.seed}.db" # url="sqlite:///:memory:" quand le lien est relatif
            )
    study = optuna.create_study(study_name=args.study_name, 
                                storage=storage_db, 
                                direction="minimize", # direction="maximize", 
                                pruner=PatientPruner(MedianPruner(), patience=10), 
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, 
                                           args.d_input_enc, 
                                           args.dataset_views_to_consider, 
                                           args.data_size, 
                                           args.output_path,
                                           args.seed), 
                   n_trials=100, timeout=54000, catch=(ReferenceError,)) #15h 54000 #12h 43200 #24h  86400 # add the catching of the reference error 
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
