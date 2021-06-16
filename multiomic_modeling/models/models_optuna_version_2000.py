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


def objective(trial: optuna.trial.Trial) -> float:
    """ Main fonction to poptimize with Optuna """
    model_params = {
        "d_input_enc": 2000, 
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        "nb_classes_dec": 33,
        "early_stopping": True,
        "dropout": trial.suggest_float("dropout", 0.15, 0.4), # 0.1, 0.5
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True), # 1e-8, 1e-2
        "activation": "relu",
        "optimizer": "Adam",
        "lr_scheduler": "cosine_with_restarts",
        "loss": "ce",
        "n_epochs": 300,
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]), # [128, 256, 512]
        "class_weights":[4.1472332 , 0.87510425, 0.30869373, 1.2229021 , 8.47878788,
            0.7000834 , 7.94886364, 1.87032086, 0.63379644, 0.63169777,
            4.19280719, 0.40417951, 1.08393595, 1.90772727, 0.72125795,
            0.87110834, 0.59523472, 0.61243251, 4.38557994, 0.63169777,
            1.94666048, 2.04035002, 0.67410858, 2.08494784, 1.40791681,
            0.79654583, 0.74666429, 2.74493133, 0.65783699, 3.02813853,
            0.65445189, 6.6937799 , 4.76931818],
        "d_model_enc_dec": trial.suggest_categorical("d_model_enc_dec", [64, 128, 256, 512]), # [32, 64, 128, 256, 512]
        "n_heads_enc_dec": trial.suggest_categorical("n_heads_enc_dec", [8, 16]),
        "n_layers_enc": trial.suggest_categorical("n_layers_enc", [4, 6, 8, 10, 12]), # [2, 4, 6, 8, 10, 12]
        "n_layers_dec": trial.suggest_categorical("n_layers_dec", [1, 2, 4]) # [1, 2, 4, 6]
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
        "data_size": 2000,
        "dataset_views_to_consider": "all",
        "type_of_model": "transformer",
        "complete_dataset": False,
        "seed": 42
    }

    model = MultiomicTrainer.run_experiment(**training_params, trial=trial, output_path='/home/maoss2/scratch/optuna_test_output_2000')
    return model.trainer.callback_metrics["val_multi_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna version of Transformer model.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    # pruning = True
    # pruner: optuna.pruners.BasePruner = (
    #     optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8000) if args.pruning else optuna.pruners.NopPruner()
    # ) # i checked this so the MedianPruner is ok but i should add the minimum step parameter
    
    storage_db = optuna.storages.RDBStorage(
                url="sqlite:////home/maoss2/scratch/optuna_test_output_2000/experiment_data_2000.db" # url="sqlite:///:memory:" quand le lien est relatif
            )
    study = optuna.create_study(study_name='experiment_data_2000', 
                                storage=storage_db, 
                                direction="maximize", 
                                pruner=PatientPruner(patience=10), 
                                load_if_exists=True)
    study.optimize(objective, n_trials=30, timeout=225000)
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
