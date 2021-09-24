import pickle
import os
import json
import numpy as np
import pandas as pd
from glob import glob

class BaseLinesAnalysis:
    
    @staticmethod
    def read_baselines_pickle_files(fichier: str):
        """
        Lit et extrait les infromations principales dans un fichier pickle de type dt_cancerName_dataSize_scores.pck
        """
        results = pickle.load(open(fichier, 'rb'))
        test_scores = results['test_scores'] # dict {'acc', 'prec', 'rec', 'f1_score'}
        train_scores = results['train_metrics']
        return test_scores, train_scores

    @staticmethod
    def build_results_baselines_file_output(repo: str = '/home/maoss2/scratch', 
                                            fichier_output: str = '/home/maoss2/scratch/baselines_results.md'):
        """
        Créé un fichier .md pour afficher les résultats. 
        Ce fichier doit etre manuellement reclassé pour un ordre chronologique. pour comparer plus tard comme il faut
        """
        os.chdir(repo)
        with open(fichier_output, 'w') as fd:
            # fd.writelines([el + '\n' for el in all_filenames])
            fd.write('| FileName | Test Metrics | Train Metrics |\n')
            fd.write('| ------------- |:-------------:| -------------:|\n')
            for fichier in glob('*.pck'):
                test_scores, train_scores = BaseLinesAnalysis.read_baselines_pickle_files(fichier)
                fd.write(f'|{fichier} |{test_scores} |{train_scores} |\n')
        
    @staticmethod
    def build_results_optuna_models(repo: str = '/home/maoss2/scratch', 
                                    repo_fichier_output: str = '/home/maoss2/scratch/'):
        os.chdir(repo)
        root_list_repo = os.listdir()
        root_list_repo = [root_repo for root_repo in root_list_repo if root_repo.find('optuna') != -1]
        root_list_repo = [root_repo for root_repo in root_list_repo if not root_repo.endswith('md')]
        for root_repo in root_list_repo:
            saving_file_name_repo = f'{repo_fichier_output}{root_repo}.md'
            os.chdir(root_repo)
            sub_root_repos_list = os.listdir()
            with open(saving_file_name_repo, 'w') as fd:
                fd.write('| FileName | Test Metrics |\n')
                fd.write('| ------------- | -------------:|\n')
                for sub_root in sub_root_repos_list: 
                    if not sub_root.startswith('experiment'): # on evite de lire la DB
                        try:
                            test_scores = json.load(open(f'{sub_root}/transformer_scores.json'))
                            fd.write(f'|{sub_root} |{test_scores} |\n')
                        except FileNotFoundError: 
                            fd.write(f'|{sub_root} |NONE|\n')
            os.chdir('../')
            
if __name__ == '__main__':
    base_lines_analyses = BaseLinesAnalysis()
    base_lines_analyses.build_results_baselines_file_output()
    base_lines_analyses.build_results_optuna_models()