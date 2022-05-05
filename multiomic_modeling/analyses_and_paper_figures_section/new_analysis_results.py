import pickle
import os
import json
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
home_path = '/home/maoss2/scratch'
# results_order are always in this order 'acc', 'prec', 'rec', 'f1_score', 'mcc_score'

class ResultsAnalysis:
    
    @staticmethod
    def read_baselines_pickle_files(fichier: str):
        results = pickle.load(open(fichier, 'rb'))
        test_scores = results['test_scores'] # dict {'acc', 'prec', 'rec', 'f1_score'}
        train_scores = results['train_metrics']
        return test_scores, train_scores
    
    @staticmethod
    def baselines_analysis_reports(directory: str = '/home/maoss2/scratch/baselines_results',
                                   output_file: str = 'baselines_reports.md'):
        os.chdir(directory) 
        real_output_file = f'{home_path}/{output_file}'
        repo_list = os.listdir()
        dt_repo_list = [repo for repo in repo_list if repo.startswith('dt')]
        rf_repo_list = [repo for repo in repo_list if repo.startswith('rf')]
        svm_repo_list = [repo for repo in repo_list if repo.startswith('svm')]
        
        dt_test_metrics = [list(ResultsAnalysis().read_baselines_pickle_files(fichier=f)[0].values()) for f in dt_repo_list]
        rf_test_metrics = [list(ResultsAnalysis().read_baselines_pickle_files(fichier=f)[0].values()) for f in rf_repo_list]
        svm_test_metrics = [list(ResultsAnalysis().read_baselines_pickle_files(fichier=f)[0].values()) for f in svm_repo_list]
        
        dt_metrics_mean = np.round(np.mean(np.array((dt_test_metrics)), axis=0), 2)
        dt_metrics_std = np.round(np.std(np.array((dt_test_metrics)), axis=0), 2)
        
        rf_metrics_mean = np.round(np.mean(np.array((rf_test_metrics)), axis=0), 2)
        rf_metrics_std = np.round(np.std(np.array((rf_test_metrics)), axis=0), 2)
        
        svm_metrics_mean = np.round(np.mean(np.array((svm_test_metrics)), axis=0), 2)
        svm_metrics_std = np.round(np.std(np.array((svm_test_metrics)), axis=0), 2)
        
        with open(real_output_file, 'w') as fd:
            fd.write('results_order are always in this order acc, prec, rec, f1_score, mcc_score')
            fd.write('| Algo| Mean | Std|\n')
            fd.write('| ------------- | ------------- |  -------------:|\n')
            fd.write(f'| DT | {dt_metrics_mean}  | {dt_metrics_std}|\n')
            fd.write(f'| RF | {rf_metrics_mean}  | {rf_metrics_std}|\n')
            fd.write(f'| SVM | {svm_metrics_mean}  | {svm_metrics_std}|\n')
        
    @staticmethod
    def read_optuna_json_output_file(sub_root: str):
        try:
            test_scores = json.load(open(f'{sub_root}/transformer_scores.json'))
        except FileNotFoundError: 
            test_scores = { "acc": 0, "prec": 0, "rec": 0, "f1_score": 0, "mcc_score": 0
                           }
        return test_scores
                            
    @staticmethod
    def read_each_optuna_repo(directory: str):
        os.chdir(directory)
        repo_list = os.listdir()
        repo_list = [el for el in repo_list if not el.startswith('experiment')]
        match_index_to_repo_position_in_the_list = {idx: repo for idx, repo in enumerate(repo_list)}
        test_metrics = [list(ResultsAnalysis().read_optuna_json_output_file(sub_root=repo).values()) for repo in repo_list]
        test_metrics = [el for el in test_metrics if el != [0, 0, 0, 0, 0]] # remove the experiment that did not finished
        # calculer le argmax sur l'accuracy et retourner juste ce truc
        best_idx = np.argmax(np.array(test_metrics)[:, 0], axis=0)
        return test_metrics[best_idx], match_index_to_repo_position_in_the_list[best_idx]
        
    @staticmethod
    def optuna_analysis_reports(directory: str = '/home/maoss2/scratch/optuna_data_aug_repo/',
                                output_file: str = 'data_aug_optuna_reports.md' ):
        os.chdir(directory)
        repo_list = os.listdir()
        results_dict = {}
        for repo in repo_list:
            rs = repo.split('_')[-1]
            best_metrics, best_metrics_repo_name = ResultsAnalysis().read_each_optuna_repo(directory=repo)
            results_dict[rs] = [best_metrics, best_metrics_repo_name]
            os.chdir('../')
        test_metrics = [results_dict[k][0] for k in results_dict]
        best_idx = np.argmax(np.array(test_metrics)[:, 0], axis=0)
        best_rs = list(results_dict.keys())[best_idx]
        test_metrics_mean = np.round(np.mean(np.array((test_metrics)), axis=0), 2)
        test_metrics_std = np.round(np.std(np.array((test_metrics)), axis=0), 2)        
        real_output_file = output_file.replace('.md', f'_{best_rs}.md')
        real_output_file = f'{home_path}/{real_output_file}'
        with open(real_output_file, 'w') as fd:
            fd.write('results_order are always in this order acc, prec, rec, f1_score, mcc_score')
            fd.write('| Best Seed | FIle Name| Best Value | - |\n')
            fd.write('| ------------- | ------------- |------------- |  -------------:|\n')
            fd.write(f'| {best_rs} | {results_dict[best_rs][-1]} | {results_dict[best_rs][0]}  | - |\n')
            fd.write('| - | - | Mean | Std |\n')
            fd.write(f'| - | - | {test_metrics_mean} | {test_metrics_std} |\n')
    
        
if __name__ ==  '__main__':
    # Baselines
    ResultsAnalysis().baselines_analysis_reports()
    # Optuna ModelAug
    ResultsAnalysis().optuna_analysis_reports()
    