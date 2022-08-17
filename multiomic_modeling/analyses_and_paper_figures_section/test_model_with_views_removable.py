import argparse
import sys
import os
from collections import defaultdict
from itertools import combinations
from typing import Tuple
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from multiomic_modeling.models.trainer import *
from multiomic_modeling.data.data_loader import MultiomicDatasetDataAug, MultiomicDatasetNormal, MultiomicDatasetBuilder, SubsetRandomSampler
from multiomic_modeling.torch_utils import to_numpy
from multiomic_modeling.analyses_and_paper_figures_section.results_and_figures_analysis import best_config_file_path_normal_data_aug_2000, best_config_file_path_normal_normal_2000
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import seaborn as sns
sns.set_theme()
home_path = '/home/maoss2/PycharmProjects/multiomic_predictions/reports_dir'
             
class TurnOffViewsDatasetNormal(MultiomicDatasetNormal):
    """ This class create a test dataset specialize for the experiment of testing the model of views turned off. 
        It just change the mask boolean to False for the omics to be turned off. 
        Since it just on the test set it can be used for both models learned on normal and data_aug datasets
    """
    def __init__(self, data_size: int = 2000, views_to_consider: str = 'all', view_to_turn_off: list = ['none']):
        super().__init__(data_size=data_size, views_to_consider=views_to_consider)
        self.view_to_turn_off =  view_to_turn_off
        self._dict_of_the_combinations = {'cnv': 0, 'methyl': 1, 'mirna': 2, 'rna': 3, 'protein': 4}
        
    def __getitem__(self, idx): 
        patient_name = self.all_patient_names[idx]
        patient_label = self.all_patient_labels[idx]
        data = np.zeros((len(self.views), self.nb_features)) # nombre_views X nombre_features
        for i, view in enumerate(self.views):
            if patient_name in view['patient_names']:
                try:
                    data[i] = view['data'][view['patient_names'].get(patient_name, 0)]
                except ValueError:
                    data[i][:view['data'][view['patient_names'].get(patient_name, 0)].shape[0]] = view['data'][view['patient_names'].get(patient_name, 0)]
        mask = np.array([(patient_name in view['patient_names']) for view in self.views])
        original_mask = deepcopy(mask)
        original_data = data.astype(float)
        if self.view_to_turn_off == ['none']: pass
        else: 
            for el in self.view_to_turn_off: mask[self._dict_of_the_combinations[el]] = False
        return (original_data, mask, original_mask), patient_label, patient_name

    
class TestModels:
    def __init__(self, number_of_view_to_consider: int = 5):
        super(TestModels).__init__()
        self.number_of_view_to_consider = number_of_view_to_consider
        
    @property    
    def number_of_view_to_consider(self):
        return self._number_of_view_to_consider
    
    @number_of_view_to_consider.setter
    def number_of_view_to_consider(self, value: int):
        assert 2 <= value <= 5, f'cannot set a value {value} not in 2 <= value <= 5'
        self._number_of_view_to_consider = value
    
    def build_set_of_potential_patients_targets(self, nb_views_per_patients: int = 5) -> list:
        assert nb_views_per_patients in [1,2,3,4,5], f'We should have 1,2,3,4 or 5 combined view per patients. This {nb_views_per_patients} is not correct'
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
        list_patients_with_nb_views = [patient_name for patient_name, value in temporaire_dict.items() if value[0] == nb_views_per_patients]
        return list_patients_with_nb_views
    
    def initialisation(self, 
                       config_file: str = '', 
                       data_size: int = 2000, 
                       dataset_views_to_consider: str = 'all',
                       random_state: int = 42):
        self.dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider=dataset_views_to_consider)
        _, new_test, _ = MultiomicDatasetBuilder.multiomic_data_normal_builder(dataset=self.dataset, test_size=0.2, valid_size=0.1, random_state=random_state)
        self.list_patients_with_nb_views = self.build_set_of_potential_patients_targets(nb_views_per_patients=self.number_of_view_to_consider)
        position_test_set_indices_to_retain = []
        patients_name_to_retain = []
        for idx in range(len(new_test.indices)):
            original_mask_patient_in_test_set = new_test[idx][0][-1]
            original_patient_name = new_test[idx][2]
            if sum(original_mask_patient_in_test_set) == self.number_of_view_to_consider and original_patient_name in self.list_patients_with_nb_views: 
                if original_patient_name not in patients_name_to_retain:
                    position_test_set_indices_to_retain.append(idx)
                    patients_name_to_retain.append(original_patient_name)
        old_indices = deepcopy(new_test.indices)
        self.new_test_indices = list(np.asarray(old_indices)[position_test_set_indices_to_retain])
        assert config_file != '', 'must have a config file (from the best model ultimately)'
        with open(config_file, 'r') as f:
            self.all_params = json.load(f)
        random.seed(self.all_params['seed'])
        np.random.seed(self.all_params['seed'])
        torch.manual_seed(self.all_params['seed'])
        self.trainer_model = MultiomicTrainer(Namespace(**self.all_params['model_params']))
    
    def test_scores(self, 
                    save_file_name: str = 'naive_scores', 
                    data_size: int = 2000, 
                    views_to_consider: str = 'all', 
                    view_to_turn_off: list = ['none'], 
                    random_state: int = 42):
        # assert view_to_turn_off in ['none', 'protein', 'methyl', 'mirna', 'rna', 'cnv'], f'the value {view_to_turn_off} is not defined and must be in [protein, methyl, mirna, rna, cnv]'
        test_dataset = TurnOffViewsDatasetNormal(data_size=data_size,
                                                 views_to_consider=views_to_consider,
                                                 view_to_turn_off=view_to_turn_off)
        _, self.test, _ = MultiomicDatasetBuilder.multiomic_data_normal_builder(dataset=test_dataset, test_size=0.2, valid_size=0.1, random_state=random_state)

        self.test.indices = self.new_test_indices
        scores_fname = os.path.join(self.all_params['fit_params']['output_path'], f'{save_file_name}_{views_to_consider}.txt')
        scores = self.trainer_model.score(dataset=self.test, 
                                          artifact_dir=self.all_params['fit_params']['output_path'], 
                                          nb_ckpts=self.all_params['predict_params'].get('nb_ckpts', 1), 
                                          scores_fname=scores_fname)    
        #print('The scores on all the 5 views are', scores)
        self.write_results_to_file(save_file_name=save_file_name, view_to_turn_off=view_to_turn_off, scores=scores)

    def write_results_to_file(self, save_file_name, view_to_turn_off, scores):
        save_file_name = f'{save_file_name}_saving_version'
        name_of_view_to_be_turned_off = '_'.join(view_to_turn_off) 
        with open(f'{save_file_name}.md', 'a+') as fd:
            fd.write(f'|__{name_of_view_to_be_turned_off}__| {scores} |\n')
        
if __name__ == "__main__":
    list_of_views_to_turn_off = ['protein', 'methyl', 'mirna', 'rna', 'cnv']
    # list_of_views_to_turn_off_copy = deepcopy(list_of_views_to_turn_off)
    # for i in range(2, 5):
    #     list_of_views_to_turn_off.extend(list(combinations(list_of_views_to_turn_off_copy, i)))
    list_of_views_to_turn_off.insert(0, 'none')
    list_of_views_to_turn_off = [[el] if type(el) == str else list(el) for el in list_of_views_to_turn_off]

    data_aug_model_test = TestModels(number_of_view_to_consider=5)
    data_aug_model_test.initialisation(config_file=best_config_file_path_normal_data_aug_2000,
                                       data_size=2000, 
                                       dataset_views_to_consider='all', 
                                       random_state=42)
    for view_off in list_of_views_to_turn_off:
        print(f'view to be off is: {view_off}')
        data_aug_model_test.test_scores(save_file_name='naive_scores_temp_data_aug', 
                                        data_size=2000, 
                                        views_to_consider='all', 
                                        view_to_turn_off=view_off, 
                                        random_state=42)
        
    print('---------------Begining the Normal data exp---------------------------\n')
    
    data_normal_test = TestModels(number_of_view_to_consider=5)
    data_normal_test.initialisation(config_file=best_config_file_path_normal_normal_2000,
                                       data_size=2000, 
                                       dataset_views_to_consider='all')
    for view_off in list_of_views_to_turn_off:
        print(f'view to be off is: {view_off}')
        data_normal_test.test_scores(save_file_name='naive_scores_temp_normal', 
                                     data_size=2000, 
                                     views_to_consider='all', 
                                     view_to_turn_off=view_off)
         