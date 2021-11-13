import pickle
import os
import json
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

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

class BuildMetricsComparisonBar:
    @staticmethod
    def compute_diverging_stacked_bar_chart(fichier: str, 
                                            targeted_metric: str, 
                                            title: str,
                                            output_path: str, 
                                            fig_name: str):
        with open(fichier) as f: 
            lines = f.readlines()
            lines = [el.split('|') for el in lines] # [['', cle, value, '\n'], []]
            dict_view_off =  {
                line[1].strip('__'): json.loads(line[2].strip(' ').replace('\'', '\"')) for line in lines
            }
        base_value = dict_view_off['aucune'][f'{targeted_metric}']
        values_to_plot = [el[f'{targeted_metric}'] - base_value for el in dict_view_off.values()] 
        classes = list(dict_view_off.keys())
        classes[0] = 'none(baseline)'
        fig, axes = plt.subplots(figsize=(16, 10))
        axes = sns.barplot(x=values_to_plot, y=classes)
        axes.set_xlabel(f'Difference in value from baseline model {targeted_metric} metric value', fontweight='bold', loc="center") 
        axes.set_ylabel('Views turned off', fontweight='bold', loc="center")
        # axes.set(xlabel=f'Difference in value from baseline model {targeted_metric} metric value', 
        #          ylabel='Views turned off')
        axes.set_xlim([-100, 5])
        if title != '': axes.set_title(f'{title}', size=15, fontweight="bold")
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

if __name__ == '__main__':
    base_lines_analyses = BaseLinesAnalysis()
    base_lines_analyses.build_results_baselines_file_output()
    base_lines_analyses.build_results_optuna_models()
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_normal_saving_version.md', 
                                                                  targeted_metric='rec', 
                                                                  title='Metrics evolution per view turned off', 
                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                  fig_name='model_normal_rec')
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_data_aug_saving_version.md', 
                                                                  targeted_metric='rec', 
                                                                  title='Metrics evolution per view turned off', 
                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                  fig_name='model_data_aug_rec')
    
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_normal_saving_version.md', 
                                                                  targeted_metric='acc', 
                                                                  title='Metrics evolution per view turned off', 
                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                  fig_name='model_normal_acc')
    BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_data_aug_saving_version.md', 
                                                                  targeted_metric='acc', 
                                                                  title='Metrics evolution per view turned off', 
                                                                  output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
                                                                  fig_name='model_data_aug_acc')
    
    # BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_normal_saving_version.md', 
        #                                                               targeted_metric='mcc_score', 
        #                                                               title='Metrics evolution per view turned off', 
        #                                                               output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
        #                                                               fig_name='model_normal_mcc')
        # BuildMetricsComparisonBar.compute_diverging_stacked_bar_chart(fichier='/Users/maoss2/PycharmProjects/multiomic_predictions/results/naive_scores_temp_data_aug_saving_version.md', 
        #                                                               targeted_metric='mcc_score', 
        #                                                               title='Metrics evolution per view turned off', 
        #                                                               output_path='/Users/maoss2/PycharmProjects/multiomic_predictions/results/', 
        #                                                               fig_name='model_data_aug_mcc')
    
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