import pickle
import os
import json
import argparse
import sys
import numpy as np
# from scipy.stats import sem
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from itertools import combinations
from typing import Tuple
from multiomic_modeling.data.data_loader import MultiomicDatasetDataAug, MultiomicDatasetNormal, MultiomicDatasetBuilder, SubsetRandomSampler
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from multiomic_modeling.models.trainer import *
from multiomic_modeling.torch_utils import to_numpy
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_theme()
# home_path = '/home/maoss2/scratch'
home_path = '/home/maoss2/PycharmProjects/multiomic_predictions/reports_dir'
# results_order are always in this order 'acc', 'prec', 'rec', 'f1_score', 'mcc_score'
best_config_file_path_normal_data_aug_2000 = '/scratch/maoss2/optuna_data_aug_output_2000/64c01d9cc9220b7fb39c2740272c1a02faff77e0/config.json' # 96.042
best_config_file_path_normal_normal_2000 = '/scratch/maoss2/optuna_normal_output_2000/ec18a0b2ca27de64e673e9dc9dfb9596970c130d/config.json' # 91.595

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
            fd.write('results_order are always in this order acc, prec, rec, f1_score, mcc_score\n')
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
                                output_file: str ='data_aug_optuna_reports.md' ):
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
        # test_metrics_std = np.round(np.std(np.array((test_metrics)), axis=0) / / np.sqrt(test_metrics), 2)  
        test_metrics_sem = np.round((np.std(np.array((test_metrics)) , axis=0) / np.sqrt(np.array(test_metrics).shape[0])), 2)
        real_output_file = output_file.replace('.md', f'_{best_rs}.md')
        real_output_file = f'{home_path}/{real_output_file}'
        with open(real_output_file, 'w') as fd:
            fd.write('results_order are always in this order acc, prec, rec, f1_score, mcc_score\n')
            fd.write('| Best Seed | FIle Name| Best Value | - |\n')
            fd.write('| ------------- | ------------- |------------- |  -------------:|\n')
            fd.write(f'| {best_rs} | {results_dict[best_rs][-1]} | {results_dict[best_rs][0]}  | - |\n')
            fd.write('| - | - | Mean | Std |\n')
            fd.write(f'| - | - | {test_metrics_mean} | {test_metrics_sem} |\n')
    
    # TODO: IT's not done/finish yet: must go back here to complete it
    @staticmethod
    def build_reports_on_dataset(data_size: int = data_size, dataset_views_to_consider: str = '3_main_omics'):
        dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider=dataset_views_to_consider)
        train, test, valid = MultiomicDatasetBuilder().multiomic_data_normal_builder(dataset=dataset, 
                                                                                         test_size=0.2, 
                                                                                         valid_size=0.1, 
                                                                                         random_state=seed)
        dataset_augmented = MultiomicDatasetDataAug(train_dataset=train, data_size=data_size, views_to_consider=dataset_views_to_consider)
        train_augmented = MultiomicDatasetBuilder.multiomic_data_aug_builder(augmented_dataset=dataset_augmented)
        train_data_loader = MultiomicDatasetBuilder().multiomic_dataset_loader(dataset=train, batch_size=len(train))
        train_augmented_data_loader = MultiomicDatasetBuilder().multiomic_dataset_loader(dataset=train_augmented, batch_size=len(train_augmented))
        test_data_loader = MultiomicDatasetBuilder().multiomic_dataset_loader(dataset=test, batch_size=len(test))
        valid_data_loader = MultiomicDatasetBuilder().multiomic_dataset_loader(dataset=valid, batch_size=len(valid))
        
        
        
class FiguresArticles:
    def __init__(self, data_size: int = 2000, dataset_views_to_consider: str = 'all'):
        super(FiguresArticles, self).__init__()
        self.dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider=dataset_views_to_consider)
    
    @staticmethod
    def figure_1():
        print('The data augmentation process. See readme: It was done on another site')
        
    @staticmethod
    def figure_2():
        print('The transformer view. See readme: It was done on another site')
    
    @staticmethod
    def bar_plot(x: np.array, 
                 y: np.array, 
                 title: str = '', 
                 write_on_bars: bool = False,
                 rotate_xticks_labels: bool = False,
                 fig_name: str = 'plot_number_of_samples_with_n_omics', 
                 x_label: str = 'Number of omics data available per samples', 
                 y_label: str = 'Number of samples') -> None:
        fig, axes = plt.subplots(figsize=(11.69, 8.27))
        axes = sns.barplot(x=x, y=y)
        axes.set_xlabel(x_label, fontweight='bold', loc="center") # fontsize=16, 
        axes.set_ylabel(y_label, fontweight='bold', loc="center") # fontsize=16, 
        if title != '': axes.set_title(f'{title}', size=15, fontweight="bold")
        # axes.set(xlabel=x_label, ylabel=y_label)
        if rotate_xticks_labels: plt.xticks(fontsize=8, rotation=315) #-225
        if write_on_bars: 
            y_percentage = (y / sum(y)) * 100
            y_percentage = [str(np.round(el, 3)) for el in y_percentage]
            axes.bar_label(axes.containers[0], y_percentage) # if we want just the number remove the y_percentage which is the label here
        fig.savefig(f'{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{fig_name}.pdf')
        plt.close(fig)
    
    def build_supplementary_figures(self) -> None:
        temporaire_dict = defaultdict(dict)
        for patient_name in self.dataset.all_patient_names:
            cpt = 0; list_of_omics_per_patients = []
            if patient_name in self.dataset.views[0]['patient_names']: cpt+=1; list_of_omics_per_patients.append('c')
            if patient_name in self.dataset.views[1]['patient_names']: cpt+=1; list_of_omics_per_patients.append('me')
            if patient_name in self.dataset.views[2]['patient_names']: cpt+=1; list_of_omics_per_patients.append('mi')
            if patient_name in self.dataset.views[3]['patient_names']: cpt+=1; list_of_omics_per_patients.append('r')
            if patient_name in self.dataset.views[4]['patient_names']: cpt+=1; list_of_omics_per_patients.append('p')
            temporaire_dict[patient_name] = [cpt, list_of_omics_per_patients]
        values_temporaire_dict = np.asarray(list(temporaire_dict.values()))
        comptes = values_temporaire_dict[:,0]
        comptes_list_of_omics_per_patients = values_temporaire_dict[:,1]
        x_comptes, y_comptes = np.unique(comptes, return_counts=True)
        x_comptes_list_of_omics_per_patients, y_comptes_list_of_omics_per_patients = np.unique(comptes_list_of_omics_per_patients, return_counts=True)
        x_comptes_list_of_omics_per_patients = np.array(['_'.join(el) for el in x_comptes_list_of_omics_per_patients])
        self.bar_plot(x=x_comptes, 
                    y=y_comptes,
                    title='Samples number with n views', 
                    write_on_bars=True,
                    rotate_xticks_labels=False,
                    fig_name='plot_number_of_samples_with_n_omics', 
                    x_label='Number of omics data available per samples', 
                    y_label='Number of samples')
        self.bar_plot(x=x_comptes_list_of_omics_per_patients, 
                    y=y_comptes_list_of_omics_per_patients, 
                    title='Samples number for each omic combination',
                    write_on_bars=False,
                    rotate_xticks_labels=True,
                    fig_name='plot_number_of_samples_for_each_combination_available', 
                    x_label='Omics data combination', 
                    y_label='Number of samples')   
        cancer_names_arrays = np.asarray([self.dataset.label_encoder.inverse_transform([i])[0] for i in self.dataset.all_patient_labels])
        x_cancer_names_arrays, y_cancer_names_arrays = np.unique(cancer_names_arrays, return_counts=True)
        self.bar_plot(x=x_cancer_names_arrays, 
                      y=y_cancer_names_arrays, 
                      title='Samples Number for each cancer type',
                      write_on_bars=False,
                      rotate_xticks_labels=True,
                      fig_name='plot_number_of_samples_per_cancer', 
                      x_label='Cancer names', 
                      y_label='Number of samples')   

class BuildMetricsComparisonBar:
    @staticmethod
    def compute_new_diverging_stacked_bar_chart(title: str = '', 
                                                fig_name: str = '',
                                                output_path: str = './', 
                                                write_on_bars : bool = True):
        # f1_scores
        modified_values = [0.20,0.98,0.98,0.94,0.88,0.64,0.93,0.80,0.63,0.95,0.82,0.84,0.86,0.95,0.93,0.99,0.83,0.72,1.00,0.78,1.00,0.96,0.99,0.37,0.97,1.00,0.92,0.98,0.99,1.00,0.95,0.74,1.00]
        original_values = [0.89,0.98,1.00,0.96,0.92,0.89,1.00,0.94,0.93,0.99,0.98,0.98,0.94,0.97,0.98,0.99,0.93,0.92,0.96,0.96,0.99,0.97,1.00,0.54,0.97,0.99,0.98,1.00,1.00,0.98,0.95,0.67,1.00]
        modified_values = np.asarray(modified_values) * 100
        original_values = np.asarray(original_values) * 100
        # recall (most of the time it's equal accuracy here)
        # modified_values = [1.00,0.96,0.96,0.93,1.00,0.50,1.00,0.82,0.47,0.95,0.73,0.78,0.79,1.00,0.99,0.99,0.74,0.65,1.00,0.66,1.00,0.92,0.98,0.67,0.97,1.00,0.94,0.96,1.00,1.00,0.97,0.88,1.00]
        # original_values = [0.92,0.98,1.00,0.95,0.86,0.85,1.00,0.94,0.93,0.97,1.00,0.98,0.93,1.00,1.00,1.00,0.95,0.87,1.00,0.95,1.00,0.97,1.00,0.62,0.98,1.00,0.95,1.00,1.00,1.00,0.98,0.75,1.00]
        # modified_values = np.asarray(modified_values) * 100
        # original_values = np.asarray(original_values) * 100
        
        cancer_labels=['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 
               'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 
               'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 
               'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
        values_to_plot = modified_values - original_values
        classes = cancer_labels
        fig, axes = plt.subplots(figsize=(16, 5))
        # self.colors = [color_palette()[0]] * len(self.colors)
        axes = sns.barplot(x=values_to_plot, y=classes, palette=sns.color_palette(n_colors = 33))
        axes.set_xlabel(f'Scores variation', loc="center") 
        # axes.set_ylabel('Cancer Label', loc="center")
        axes.yaxis.set_visible(False)
        axes.legend(labels=cancer_labels, labelcolor=sns.color_palette(n_colors = 33), fontsize='xx-small') # , labelcolor='linecolor'
        if write_on_bars: 
            axes.bar_label(axes.containers[0]) 
        axes.set_xlim([-80, 10])
        if title != '': axes.set_title(f'{title}', size=10, fontweight="bold")
        fig.savefig(f'{output_path}/{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{output_path}/{fig_name}.pdf')
        plt.close(fig)
        
    @staticmethod
    def compute_diverging_stacked_bar_chart(fichier: str, 
                                            targeted_metric: str, 
                                            title: str,
                                            output_path: str, 
                                            fig_name: str,
                                            write_on_bars: bool = False):
        with open(fichier) as f: 
            lines = f.readlines()
            lines = [el.split('|') for el in lines] # [['', cle, value, '\n'], []]
            dict_view_off =  {
                line[1].strip('__'): json.loads(line[2].strip(' ').replace('\'', '\"')) for line in lines
            }
        base_value = dict_view_off['none'][f'{targeted_metric}']
        values_to_plot = [el[f'{targeted_metric}'] - base_value for el in dict_view_off.values()] 
        classes = list(dict_view_off.keys())
        classes[0] = 'none(baseline)'
        fig, axes = plt.subplots(figsize=(16, 10))
        # axes = sns.barplot(x=values_to_plot, y=classes, linewidth=2.5, facecolor=(1, 1, 1, 0), edgecolor=".2", color="black", saturation=.5)
        axes = sns.barplot(x=values_to_plot, y=classes, color="black")
        # axes.set_xlabel(f'Difference in value from baseline model {targeted_metric} metric value', fontweight='bold', loc="center", fontsize=20) 
        # axes.set_ylabel('Views turned off', fontweight='bold', loc="center", fontsize=20)
        axes.set_xlabel(f'Difference in value from baseline model {targeted_metric} metric value', fontweight='bold', loc="center", fontsize='xx-large') 
        axes.set_ylabel('Views turned off', fontweight='bold', loc="center", fontsize='xx-large')
        if write_on_bars: 
            axes.bar_label(axes.containers[0]) 
        axes.set_xlim([-3, 1])
        if title != '': axes.set_title(f'{title}', fontsize='xx-large', fontweight="bold")
        fig.savefig(f'{output_path}/{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{output_path}/{fig_name}.pdf')
        plt.close(fig)

    @staticmethod
    def compute_baseline_models_comparison_bar_chart_figure(fichier: str, 
                                                            targeted_metric: str,  
                                                            #title: str,
                                                            output_path: str, 
                                                            fig_name: str):
        with open(fichier) as f: 
            lines = f.readlines()
            lines = [el.split('|') for el in lines] # [['', cle, value, '\n'], []]
            dict_view_off =  {
                '_'.join(line[1].strip(' ').split('_')[:2]): json.loads(line[2].strip(' ').replace('\'', '\"')) for line in lines[2:]
            }
        all_cle = [el for el in dict_view_off.keys() if el.endswith('all')]; all_values = [dict_view_off[el] for el in all_cle]
        for dico in all_values: dico['mcc_score'] = np.round(dico['mcc_score'] * 100, 3) 
        all_values_cibles = [dico[targeted_metric] for dico in all_values]
        
        mirna_cle = [el for el in dict_view_off.keys() if el.endswith('mirna')]; mirna_values = [dict_view_off[el] for el in mirna_cle]
        for dico in mirna_values: dico['mcc_score'] = np.round(dico['mcc_score'] * 100, 3) 
        mirna_values_cibles = [dico[targeted_metric] for dico in mirna_values]
        
        rna_cle = [el for el in dict_view_off.keys() if el.endswith('rna') and el.find('mi') == -1]; rna_values = [dict_view_off[el] for el in rna_cle]
        for dico in rna_values: dico['mcc_score'] = np.round(dico['mcc_score'] * 100, 3) 
        rna_values_cibles = [dico[targeted_metric] for dico in rna_values]
        
        protein_cle = [el for el in dict_view_off.keys() if el.endswith('protein')]; protein_values = [dict_view_off[el] for el in protein_cle]
        for dico in protein_values: dico['mcc_score'] = np.round(dico['mcc_score'] * 100, 3) 
        protein_values_cibles = [dico[targeted_metric] for dico in protein_values]
        
        cnv_cle = [el for el in dict_view_off.keys() if el.endswith('cnv')]; cnv_values = [dict_view_off[el] for el in cnv_cle]
        for dico in cnv_values: dico['mcc_score'] = np.round(dico['mcc_score'] * 100, 3) 
        cnv_values_cibles = [dico[targeted_metric] for dico in cnv_values]
        
        methyl_cle = [el for el in dict_view_off.keys() if el.endswith('methyl')]; methyl_values = [dict_view_off[el] for el in methyl_cle]
        for dico in methyl_values: dico['mcc_score'] = np.round(dico['mcc_score'] * 100, 3) 
        methyl_values_cibles = [dico[targeted_metric] for dico in methyl_values]
        
        cles_liste = [all_cle, cnv_cle, methyl_cle, rna_cle, mirna_cle, protein_cle]
        values_liste = [all_values_cibles, cnv_values_cibles, methyl_values_cibles, rna_values_cibles, mirna_values_cibles, protein_values_cibles]
        fig, axes = plt.subplots(figsize=(16, 10), nrows=2, ncols=3)
        fig.suptitle(f'Baseline algorithms (DT, RF, SVM and DNN) metrics perfomance evalution', size=15, fontweight='bold')
        palette = sns.color_palette(n_colors=4)
        
        for idx, ax in enumerate(axes.flat): 
            if idx == 0: ax.bar(x=cles_liste[0], height=values_liste[0], width=0.5, color=palette); ax.set_title("ALL", fontweight="bold")
            if idx == 1: ax.bar(x=cles_liste[1], height=values_liste[1], width=0.5, color=palette); ax.set_title("CNV", fontweight="bold")
            if idx == 2: ax.bar(x=cles_liste[2], height=values_liste[2], width=0.5, color=palette); ax.set_title("METHYL", fontweight="bold")
            if idx == 3: ax.bar(x=cles_liste[3], height=values_liste[3], width=0.5, color=palette); ax.set_title("RNA", fontweight="bold")
            if idx == 4: ax.bar(x=cles_liste[4], height=values_liste[4], width=0.5, color=palette); ax.set_title("MIRNA", fontweight="bold")
            if idx == 5: ax.bar(x=cles_liste[5], height=values_liste[5], width=0.5, color=palette); ax.set_title("PROTEIN", fontweight="bold")
            # ax.set(xlabel='Algorithms', ylabel=f'{targeted_metric}')
            # axes.set_xlabel(f'Difference in value from baseline model {targeted_metric} metric value', fontweight='bold', loc="center") 
            ax.set_ylabel(f'{targeted_metric}', fontweight="bold")
            ax.set_ylim([0, 100])
            ax.bar_label(ax.containers[0])
        # More a note to myself: I can't put the legend" I spent too much time on this BS i give up
            # labels = ['dt', 'rf', 'svm', 'dnn']; fig.legend(labels, loc='upper right', prop={'size': 10}, labelcolor=palette, edgecolor=palette) 
        fig.savefig(f'{output_path}/{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{output_path}/{fig_name}.pdf')
        plt.close(fig)
    
    @staticmethod
    def compute_our_models_comparison_bar_chart_figure(fichier: str, 
                                                       targeted_metric: str, 
                                                       title: str,
                                                       output_path: str, 
                                                       fig_name: str):
        with open(fichier) as f: 
            lines = f.readlines()
            lines = [el.split('|') for el in lines] # [['', cle, value, '\n'], []]
            dict_view_off =  {
                line[1].strip(' '): json.loads(line[2].strip(' ').replace('\'', '\"')) for line in lines
            }
        values_to_plot = [el[f'{targeted_metric}'] for el in dict_view_off.values()] 
        # classes = list(dict_view_off.keys())
        classes = ['normal', 'data_aug', 'multimodal_data_aug']
        fig, axes = plt.subplots(figsize=(16, 10))
        palette = sns.color_palette(n_colors=3)
        axes.bar(x=classes, height=values_to_plot, width=0.5, color=palette)
        axes.set_xlabel(f'MutiOmicTransformerModel (MOTM) and  MutiOmicTransformerModelMultimodal (MOTMM)', fontweight='bold', loc="center") 
        axes.set_ylabel('Metric value in %', fontweight='bold', loc="center")
        axes.bar_label(axes.containers[0])
        # axes.set(xlabel=f'Our model metric comparison: metric {targeted_metric} ', ylabel='')
        if title != '': axes.set_title(f'{title}', size=15, fontweight='bold')
        fig.savefig(f'{output_path}/{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{output_path}/{fig_name}.pdf')
        plt.close(fig)

    @staticmethod
    def plot_all_metrics_together(fichier_our_model: str,
                                  fichier_baseline_model: str, 
                                  targeted_metric: str, 
                                  title: str,
                                  output_path: str, 
                                  fig_name: str):
        with open(fichier_our_model) as f: 
            lines = f.readlines()
            lines = [el.split('|') for el in lines] # [['', cle, value, '\n'], []]
            dict_view_off =  {
                line[1].strip(' '): json.loads(line[2].strip(' ').replace('\'', '\"')) for line in lines
            }
        values_to_plot = [el[f'{targeted_metric}'] for el in dict_view_off.values()] 
        classes = ['motm_on_normal', 'motm_on_data_aug', 'mtom_mm_on_data_aug']
        
        with open(fichier_baseline_model) as f: 
            lines = f.readlines()
            lines = [el.split('|') for el in lines] # [['', cle, value, '\n'], []]
            dict_view_off_base =  {
                '_'.join(line[1].strip(' ').split('_')[:2]): json.loads(line[2].strip(' ').replace('\'', '\"')) for line in lines[2:]
            }
        all_cle = [el for el in dict_view_off_base.keys() if el.endswith('all')]; all_values = [dict_view_off_base[el] for el in all_cle]
        all_values_cibles = [dico[targeted_metric] for dico in all_values]
      
        final_values_to_plot = []; final_values_to_plot.extend(all_values_cibles); final_values_to_plot.extend(values_to_plot)
        final_classes = []; final_classes.extend(all_cle); final_classes.extend(classes)
    
        fig, axes = plt.subplots(figsize=(16, 10))
        palette = sns.color_palette(n_colors=7)
        axes.bar(x=final_classes, height=final_values_to_plot, width=0.5, color=palette)
        axes.set_xlabel(f'Algorithms', fontweight='bold', loc="center") 
        axes.set_ylabel(f'{targeted_metric} value in %', fontweight='bold', loc="center")
        axes.bar_label(axes.containers[0])
        
        if title != '': axes.set_title(f'{title}', size=15, fontweight='bold')
        fig.savefig(f'{output_path}/{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{output_path}/{fig_name}.pdf')
        plt.close(fig)

# AttentionWeightANalysis Section
class InspectedValues(object):
    def __init__(self, func_name: str):
        self.func_name = func_name
        self._returned = None
    def __call__(self, returned):
        self._returned = returned
    def get(self):
        if self._returned is None:
            raise Exception(f"Function {self.func_name} has not been called.")
        return self._returned

class Inspect(object):
    def __init__(self, obj: object, func_name: str):
        self.obj = obj
        self.func_name = func_name
        self.overrided = None
    def __enter__(self):
        inspected = InspectedValues(self.func_name)
        self.overrided = self.obj.__getattribute__(self.func_name)
        def new(*args, **vargs):
            values = self.overrided(*args, **vargs)  # type: ignore
            inspected(values)
            return values
        self.obj.__setattr__(self.func_name, new)
        return inspected
    def __exit__(self, exc_type, exc_value, tb):
        self.obj.__setattr__(self.func_name, self.overrided)

class AttentionWeightsAnalysis:    
    @staticmethod
    def build_examples_per_cancer(data_size: int = 2000) -> Tuple[list, list]:
        dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider='all')
        _, test, _ = MultiomicDatasetBuilder.multiomic_data_normal_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
        
        test_data_loader =  MultiomicDatasetBuilder.multiomic_dataset_loader(dataset=test, batch_size=32, nb_cpus=2)
        list_of_cancer_names = [test.dataset.label_encoder.inverse_transform([i])[0] for i in range(33)]
        list_of_examples = [[] for _ in range(33)] # chaque sous liste va contenir les examples de ce type de cancer  
        for test_data in iter(test_data_loader):
            batch_examples = test_data[0]
            batch_examples_labels = test_data[1]
            for sub_idx, cancer_label in enumerate(batch_examples_labels):
                list_of_examples[int(cancer_label)].append([batch_examples[0][sub_idx], 
                                                            batch_examples[1][sub_idx]])
        return list_of_examples, list_of_cancer_names

    @staticmethod        
    def get_attention_weights(trainer, inputs_list: list) -> list:
        """
        Args:
            trainer, MultiomicTrainer or MultiomicTrainerMultimodal
            inputs_list, list of [[tensor_of_the_views, tensor_of_the_random_mask, tensor_of_the_originall_mask], [], ..., []]
            basically this is a list of all the examples per cancer
        Return:
            output with this shape [number_of_layer * batch_size * number_of_views * number_of_views]
        """
        inputs_list_copy = deepcopy(inputs_list)
        original_data = torch.Tensor([np.asarray(el[0]) for el in inputs_list_copy]).float()
        mask = torch.Tensor([np.asarray(el[1]) for el in inputs_list_copy]).bool()
        inputs = [original_data, mask]
       
        res = []
        for layer in range(trainer.network.encoder.n_layers):
            try:
                attention_module = trainer.network.encoder.net.layers[layer].self_attn  # type: ignore
            except Exception as e:
                raise ValueError(
                    f"Model {trainer.__name__} "
                    + "has no attention module or can't use the default function implementation.",
                    e,
                )
            with Inspect(attention_module, "forward") as returned:
                trainer.network(inputs)
                temp = returned.get()[-1]
                # print(temp.shape)
                # print(type(temp))
                res.append(temp)
        # print(len(res))
        return torch.stack(res, dim=0) # return [number_of_layer * batch_size * number_of_views * number_of_views]

    @staticmethod
    def plot_attentions_weights_per_cancer(cancer_weights, 
                                           output_path: str = './', 
                                           fig_name: str = 'cancer', 
                                           columns_names: list = ['cnv', 'methyl', 'mirna', 'rna', 'protein'], 
                                           exp_type: str = 'normal'):
        torch.manual_seed(42)
        np.random.seed(42)
        final_array = to_numpy(torch.nn.functional.softmax(cancer_weights * 100, dim=-1).mean(dim=0))
        final_array = np.round(final_array, 3)
        print(final_array)
        fig, axes = plt.subplots(figsize=(11.69, 8.27))
        sns.heatmap(final_array, vmin=0, vmax=1, annot=True, linewidths=0.1, 
                        xticklabels=columns_names, yticklabels=columns_names)
        fig.savefig(f'{output_path}/{fig_name}_{exp_type}.pdf')
        plt.close(fig)
    
    #TODO : a modifier
    @staticmethod
    def plot_tsne(list_of_examples_per_cancer: list,
                  list_of_cancer_names: list,
                  trainer_model, 
                  plot_for_all_cancer: bool = True,
                  plot_for_all_examples_for_all_cancer: bool= True,
                  output_path: str = './', 
                  fig_name: str = 'tsne_cancer'):
                  #, columns_names: list = ['cnv', 'methyl_450', 'mirna', 'rna', 'protein']
        # if plot_for_all_cancer:
            #     X = []; y = []
            #     for idx, cancer_list in enumerate(list_of_examples_per_cancer):
            #         cancer_name = list_of_cancer_names[idx]
            #         attention_weights_per_layer_for_cancer_list = AttentionWeightsAnalysis().get_attention_weights(trainer=trainer_model, inputs_list=cancer_list)
            #         # examples_weigths_per_cancer = torch.mean(attention_weights_per_layer_for_cancer_list, dim=0) 
            #             # final_array = to_numpy(torch.nn.functional.softmax(examples_weigths_per_cancer * 100, dim=-1)
            #             # # la moyenne est nécessaire car on fait la moyenne sur les layers de l'output: ainsi on obtient une "seule layer"
            #             # X.append(to_numpy(torch.flatten(attention_weights_per_layer_for_cancer_list)))
            #             # y.append(cancer_name)
            #             # X.append(np.sum(final_array, axis=0)) # devient 1d vector (1, 5)
            #         examples_weigths_per_cancer = torch.mean(attention_weights_per_layer_for_cancer_list, dim=0)
            #         final_array = to_numpy(torch.flatten((torch.nn.functional.softmax(examples_weigths_per_cancer * 100, dim=-1).mean(dim=0))))
            #         X.append(final_array)
            #         y.append(cancer_name)
            #     X = np.asarray(X)
            #     y = np.asarray(y)
            #     tsne = TSNE()
            #     X_embedded = tsne.fit_transform(X)
            #     palette = sns.color_palette("bright", len(y))
            #     fig, axes = plt.subplots(figsize=(11.69, 8.27))
            #     sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
            #     fig.savefig(f'{output_path}/{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{output_path}/{fig_name}.pdf')
            #     plt.close(fig)
        if plot_for_all_examples_for_all_cancer:
            X = []; y = []
            for idx, cancer_list in enumerate(list_of_examples_per_cancer):
                cancer_name = list_of_cancer_names[idx]
                attention_weights_per_layer_for_cancer_list = AttentionWeightsAnalysis.get_attention_weights(trainer=trainer_model, inputs_list=cancer_list)
                examples_weigths_per_cancer = torch.mean(attention_weights_per_layer_for_cancer_list, dim=0)
                # examples_weigths_per_cancer = attention_weights_per_layer_for_cancer_list[-1] #last layer
                # X.extend(to_numpy(torch.sum(examples_weigths_per_cancer, (1))))
                X.extend(to_numpy(torch.flatten(examples_weigths_per_cancer, start_dim=1, end_dim=2)))
                y.extend([cancer_name] * examples_weigths_per_cancer.shape[0])
            X = np.asarray(X)
            y = np.asarray(y)
            tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1, learning_rate=250, init='pca') #init='random'
            X_embedded = tsne.fit_transform(X)
            palette = sns.color_palette("bright", len(y))
            # fig, axes = plt.subplots(figsize=(11.69, 8.27))
            # sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full') #, palette=palette
            df_subset = pd.DataFrame.from_dict({'tsne_col_0': X_embedded[:,0], 
                                                'tsne_col_1': X_embedded[:,1], })
            fig = plt.figure(figsize=(16,10))
            sns.scatterplot(
                x="tsne_col_0", y="tsne_col_1",
                hue=y,
                palette=sns.color_palette("hls", 33),
                data=df_subset,
                legend="full",
                alpha=0.3
            )
            fig.savefig(f'{output_path}/{fig_name}') if fig_name.endswith('pdf') else fig.savefig(f'{output_path}/{fig_name}.pdf')
            plt.close(fig)

def main_attention_weights_plot(config_file: str, 
                                algo_type: str = 'normal', 
                                exp_type: str = 'normal', 
                                output_path: str = './', 
                                data_size: int = 2000):
    list_of_examples_per_cancer, list_of_cancer_names = AttentionWeightsAnalysis.build_examples_per_cancer(data_size=int(data_size)) # length of 33
    with open(config_file, 'r') as f:
        all_params = json.load(f)
    random.seed(all_params['seed'])
    np.random.seed(all_params['seed'])
    torch.manual_seed(all_params['seed'])
    if algo_type == 'normal': trainer_model = MultiomicTrainer(Namespace(**all_params['model_params']))
    elif algo_type == 'multimodal' : trainer_model = MultiomicTrainerMultiModal(Namespace(**all_params['model_params']))
    else: raise f'The algotype: {algo_type} is not implemented' 
    
    for idx, cancer_list in enumerate(list_of_examples_per_cancer):
        cancer_name = list_of_cancer_names[idx]
        if not os.path.exists(f'/scratch/maoss2/{output_path}/{cancer_name}_{exp_type}.pdf'):
            attention_weights_per_layer_for_cancer_list = AttentionWeightsAnalysis.get_attention_weights(trainer=trainer_model, inputs_list=cancer_list)
            # Here we could just extract the last layer information and plot the figure with that?
            # examples_weigths_per_cancer = attention_weights_per_layer_for_cancer_list 
            examples_weigths_per_cancer = torch.mean(attention_weights_per_layer_for_cancer_list, dim=0) # matrice [len(cancer_list) * nb_view * nb_view]:
                # essentiellement on fait la moyenne sur les layer. the dim=0 qui veut dire 1ere dimension qui est le nombre de layer du dec en output. 
                # du coup la matrice obtenu peut etre utilisée ici pour l'étape de clustering par eg (en associant) le label je suppose ca devrait passer
            print(cancer_name)
            print(examples_weigths_per_cancer.shape) 
            AttentionWeightsAnalysis.plot_attentions_weights_per_cancer(cancer_weights=examples_weigths_per_cancer,
                                                                        output_path=output_path, 
                                                                        fig_name=cancer_name, 
                                                                        columns_names=['cnv', 'methyl', 'mirna', 'rna', 'protein'],
                                                                        exp_type=exp_type)
    AttentionWeightsAnalysis.plot_tsne(list_of_examples_per_cancer=list_of_examples_per_cancer,
                                      list_of_cancer_names=list_of_cancer_names,
                                      trainer_model=trainer_model,
                                      plot_for_all_cancer=False,
                                      plot_for_all_examples_for_all_cancer=True,
                                      output_path=output_path,
                                      fig_name=f'all_examples_tsne_{exp_type}')

def main_compute_diverging_stacked_bar_chart():
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_normal_saving_version.md', 
                                                                  targeted_metric='rec', 
                                                                  title='Metrics evolution per view turned off', 
                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                  fig_name='model_normal_rec')
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_data_aug_saving_version.md', 
                                                                  targeted_metric='rec', 
                                                                  title='Metric scores divergence per omic view turned off', 
                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                  fig_name='model_data_aug_rec')
    
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_normal_saving_version.md', 
                                                                  targeted_metric='acc', 
                                                                  title='Metrics evolution per view turned off', 
                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                  fig_name='model_normal_acc')
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='temp.md', 
                                                                  targeted_metric='acc', 
                                                                  title='Metric scores divergence per omic view turned off', 
                                                                  output_path='./', 
                                                                  fig_name='model_data_aug_acc',
                                                                  write_on_bars=True)
    
def main_compute_baseline_models_comparison_bar_chart_figure():
    BuildMetricsComparisonBar.compute_baseline_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md',
                                                                                  targeted_metric='rec',
                                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                                  fig_name='baselines_model_recall')
    BuildMetricsComparisonBar.compute_baseline_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md',
                                                                                  targeted_metric='f1_score',
                                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                                  fig_name='baselines_model_f1_score')
    BuildMetricsComparisonBar.compute_baseline_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md',
                                                                                  targeted_metric='prec',
                                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                                  fig_name='baselines_model_precision')
    BuildMetricsComparisonBar.compute_baseline_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md',
                                                                                  targeted_metric='acc',
                                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                                  fig_name='baselines_model_accuracy')
    BuildMetricsComparisonBar.compute_baseline_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md',
                                                                                  targeted_metric='mcc_score',
                                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                                  fig_name='baselines_model_mcc_score')
    
    BuildMetricsComparisonBar.compute_our_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                                               targeted_metric='rec',
                                                                               title='MOTM and MOTMM evaluation performamce', 
                                                                               output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                               fig_name='our_models_recall')
    BuildMetricsComparisonBar.compute_our_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                                               targeted_metric='f1_score',
                                                                               title='MOTM and MOTMM evaluation performamce', 
                                                                               output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                               fig_name='our_models_f1_score')
    BuildMetricsComparisonBar.compute_our_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                                               targeted_metric='prec',
                                                                               title='MOTM and MOTMM evaluation performamce', 
                                                                               output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                               fig_name='our_models_precision')
    BuildMetricsComparisonBar.compute_our_models_comparison_bar_chart_figure(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                                               targeted_metric='acc',
                                                                               title='MOTM and MOTMM evaluation performamce', 
                                                                               output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                               fig_name='our_models_accuracy')

def main_plot_all_metrics_together():
    BuildMetricsComparisonBar.plot_all_metrics_together(fichier_our_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                        fichier_baseline_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md', 
                                                        targeted_metric='acc', 
                                                        title='Baselines and MOTM models metrics (Accuracy) perfomance evalution', 
                                                        output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                        fig_name='baseline_vs_motm_models_comparison_accuracy')
    BuildMetricsComparisonBar.plot_all_metrics_together(fichier_our_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                        fichier_baseline_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md', 
                                                        targeted_metric='prec', 
                                                        title='Baselines and MOTM models metrics (Precision) perfomance evalution', 
                                                        output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                        fig_name='baseline_vs_motm_models_comparison_precision')
    BuildMetricsComparisonBar.plot_all_metrics_together(fichier_our_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                        fichier_baseline_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md', 
                                                        targeted_metric='rec', 
                                                        title='Baselines and MOTM models metrics (Recall) perfomance evalution', 
                                                        output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                        fig_name='baseline_vs_motm_models_comparison_recall')
    BuildMetricsComparisonBar.plot_all_metrics_together(fichier_our_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/our_model_comparison.md', 
                                                        fichier_baseline_model='/Users/maoss2/PycharmProjects/multiomic_predictions/results/baselines_results.md', 
                                                        targeted_metric='f1_score', 
                                                        title='Baselines and MOTM models metrics (F1_score) perfomance evalution', 
                                                        output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                        fig_name='baseline_vs_motm_models_comparison_f1_score')

def main_compute_new_diverging_stacked_bar_chart():
    BuildMetricsComparisonBar.compute_new_diverging_stacked_bar_chart(title='F1 score score variations after important views removal', 
                                                                      fig_name='new_fig_f1_score.pdf', 
                                                                      output_path='./', 
                                                                      write_on_bars=True)    
 
if __name__ ==  '__main__':
    # Baselines
    ResultsAnalysis().baselines_analysis_reports()
    # Optuna ModelAug
    # ResultsAnalysis().optuna_analysis_reports()
    ResultsAnalysis().optuna_analysis_reports(directory='optuna_normal_3_main_omics_repo/', output_file='normal_3_main_omics_reports.md')
    ResultsAnalysis().optuna_analysis_reports(directory='optuna_normal_all_repo/', output_file='normal_all_reports.md')
    ResultsAnalysis().optuna_analysis_reports(directory='optuna_data_aug_3_main_omics_repo/', output_file='data_aug_3_main_omics_reports.md')
    ResultsAnalysis().optuna_analysis_reports(directory='optuna_data_aug_all_repo/', output_file='data_augl_all_reports.md')
    # Build figures
    fig_article = FiguresArticles(data_size=2000, dataset_views_to_consider='all')
    fig_article.figure_1()
    fig_article.figure_2()
    fig_article.build_supplementary_figures()
    # attention Weight Plot
    # to run this: salloc --time=03:00:00 --nodes=1  --ntasks-per-node=16 --mem=64G --account=rrg-corbeilj-ac
    main_attention_weights_plot(config_file=best_config_file_path_normal_normal_2000, 
                                algo_type='normal', 
                                exp_type='normal', 
                                output_path=best_config_file_path_normal_normal_2000[:-12], 
                                data_size=2000)
    main_attention_weights_plot(config_file=best_config_file_path_normal_data_aug_2000, 
                                algo_type='normal', 
                                exp_type='data_aug', 
                                output_path=best_config_file_path_normal_data_aug_2000[:-12], 
                                data_size=2000)
    