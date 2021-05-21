import argparse
import os
import random
import numpy as np
from argparse import Namespace

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
from multiomic_modeling.models.trainer import MultiomicTrainer
from multiomic_modeling.data.data_loader import MultiomicDataset, SubsetRandomSampler, multiomic_dataset_builder

import pytorch_lightning as pl
import torch

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")

training_params_global = []
def objective(trial: optuna.trial.Trial) -> float:
    """ Main fonction to poptimize with Optuna """
    model_params = {
        "d_input_enc": 2000, #TODO: modifier ceci to 5k or 10k when i wanna test on other dataset
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "nb_classes_dec": 33,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 0.1, 0.5),
        "activation": "relu",
        "optimizer": "Adam",
        "lr_scheduler": "cosine_with_restarts",
        "loss": "ce",
        "n_epochs": 200,
        "batch_size": 128,
        "class_weights":[4.1472332 , 0.87510425, 0.30869373, 1.2229021 , 8.47878788,
            0.7000834 , 7.94886364, 1.87032086, 0.63379644, 0.63169777,
            4.19280719, 0.40417951, 1.08393595, 1.90772727, 0.72125795,
            0.87110834, 0.59523472, 0.61243251, 4.38557994, 0.63169777,
            1.94666048, 2.04035002, 0.67410858, 2.08494784, 1.40791681,
            0.79654583, 0.74666429, 2.74493133, 0.65783699, 3.02813853,
            0.65445189, 6.6937799 , 4.76931818],
        "d_model_enc_dec": trial.suggest_categorical("n_heads_enc_dec", [128, 256, 512, 1024, 2048]),
        "n_heads_enc_dec": trial.suggest_categorical("n_heads_enc_dec", [128, 256, 512, 1024, 2048]),
        "n_layers_enc_dec": trial.suggest_int("n_layers_enc_dec", 1, 10, step=1)
    }
    d_ff_enc_dec_value = model_params["d_model_enc_dec"] * 4
    model_params["d_ff_enc_dec"] = d_ff_enc_dec_value

    fit_params = {
        "nb_ckpts":1, 
        "verbose":1
    }

    predict_params = {
        "nb_ckpts":1, 
        "scores_fname": "transformer_scores.json" # change the name when we doing 2K 5K or 10k?
    }

    training_params = {
        "model_params": model_params,
        "fit_params": fit_params,
        "predict_params": predict_params,
        "dataset_views_to_consider": "all",
        "type_of_model": "transformer",
        "complete_dataset": False,
        "seed": trial.suggest_int("seed", 42, 1000)
    }
    training_params_global.append(training_params)
    # TODO: Change the outputpath for each exp
    model = MultiomicTrainer.run_experiment(**training_params, trial=trial, output_path='/home/maoss2/scratch/optuna_test_output_2000')
    # model = MultiomicTrainer.run_experiment(**training_params, trial=trial, output_path='/home/maoss2/scratch/optuna_test_output_5000')
    # model = MultiomicTrainer.run_experiment(**training_params, trial=trial, output_path='/home/maoss2/scratch/optuna_test_output_10000')
    # model = MultiomicTrainer.run_experiment(trial=trial, **training_params, output_path='./')
    
    return model.trainer.callback_metrics["val_multi_acc"].item()

def detailed_objective(trial):
    """ For the evaluation on the test set """
    all_params = training_params_global[0]
    random.seed(all_params['seed'])
    np.random.seed(all_params['seed'])
    torch.manual_seed(all_params['seed'])
    dataset = MultiomicDataset(views_to_consider=all_params["dataset_views_to_consider"], 
                                   type_of_model=all_params["type_of_model"], 
                                   complete_dataset=all_params["complete_dataset"])
    train, valid, test = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
    model = MultiomicTrainer(Namespace(**all_params['model_params']))
    scores_fname = os.path.join(all_params['fit_params']['output_path'], 
                                all_params['predict_params'].get('scores_fname', "naive_scores.txt"))
    scores = model.score(dataset=test, 
                         artifact_dir=all_params['fit_params']['output_path'], 
                         nb_ckpts=all_params['predict_params'].get('nb_ckpts', 1), 
                         scores_fname=scores_fname) # scores_fname
    return scores['acc'], scores['prec'], scores['rec'], scores['f1_score']

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

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if bool(args.pruning) else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=None, n_jobs=-1) # modifier le nbre de trials ici

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    scores_test = detailed_objective(study.best_trial)
    print(f'Acc {scores_test[0]}\tPrec {scores_test[1]}\tRec {scores_test[2]}\tF1_score {scores_test[3]}')