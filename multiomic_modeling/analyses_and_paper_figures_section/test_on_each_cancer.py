import argparse
import sys
from collections import defaultdict
from itertools import combinations
from typing import Tuple
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from multiomic_modeling.models.trainer import *
from multiomic_modeling.data.data_loader import MultiomicDatasetDataAug, MultiomicDatasetNormal, MultiomicDatasetBuilder, SubsetRandomSampler
from multiomic_modeling.torch_utils import to_numpy
from multiomic_modeling.analyses_and_paper_figures_section.attention_weights_analysis import best_config_file_path_normal_data_aug_2000, best_config_file_path_normal_normal_2000
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import seaborn as sns
sns.set_theme()

# TODO: La meme classe mais au test set on met les vues de caque cancer à off?              
class TurnOffViewsDatasetNormal(MultiomicDatasetNormal):
    """ This class create a test dataset specialize for the experiment of testing the model of views turned off. 
        It just change the mask boolean to False for the omics to be turned off. 
        Since it just on the test set it can be used for both models learned on normal and data_aug datasets
    """
    def __init__(self, data_size: int = 2000, 
                 views_to_consider: str = 'all',
                 cancer_targeted: int = 0):
        super().__init__(data_size=data_size, views_to_consider=views_to_consider)
        self.cancer_targeted = cancer_targeted
        self._dict_of_the_combinations = {'cnv': 0, 'methyl': 1, 'mirna': 2, 'rna': 3, 'protein': 4}
        self.dict_cancer_to_views = {0:['cnv','rna'], 1:['rna'], 2:['rna'],  3:['cnv', 'rna'], 4:['rna'], 5:['rna'], 
                            6:['mirna','rna'], 7:['rna'], 8:['cnv'], 9:['cnv','rna'], 10:['mirna','rna'], 
                            11:['methyl','rna'], 12:['mirna','rna'], 13:['mirna','rna'], 14:['rna'], 15:['cnv','mirna','rna'], 
                            16:['cnv','mirna','rna'], 17:['cnv','mirna','rna'], 18:['cnv','rna'], 19:['cnv','mirna','rna'],
                            20:['cnv','rna'], 21:['mirna','rna'], 22:['rna'], 23:['cnv','rna'], 24:['rna'], 25:['rna'], 
                            26:['rna'], 27:['mirna','rna'], 28:['rna'], 29:['mirna','rna'], 30:['rna'], 31:['rna'], 32:['cnv','mirna','rna']
                            }
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
        if patient_label == self.cancer_targeted:
            for el in self.dict_cancer_to_views[patient_label]: mask[self._dict_of_the_combinations[el]] = False
        return (original_data, mask, original_mask), patient_label, patient_name

        
class TestMOTOnSamplesWithAllExamples:
    def __init__(self, number_of_view_to_consider: int = 5):
        super(TestMOTOnSamplesWithAllExamples).__init__()
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
                       dataset_views_to_consider: str = 'all'):
        self.dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider=dataset_views_to_consider)
        _, new_test, _ = MultiomicDatasetBuilder.multiomic_data_normal_builder(dataset=self.dataset, test_size=0.2, valid_size=0.1)
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
        print(f'Length Original test set : {len(old_indices)} \n Length Test set with all 5 omics present: {len(self.new_test_indices)}')
        self.test = new_test
        self.test.indices = self.new_test_indices
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
                    views_to_consider: str = 'all'):
        scores_fname = os.path.join(self.all_params['fit_params']['output_path'], f'{save_file_name}_{views_to_consider}.txt')
        scores = self.trainer_model.score(dataset=self.test, 
                                          artifact_dir=self.all_params['fit_params']['output_path'], 
                                          nb_ckpts=self.all_params['predict_params'].get('nb_ckpts', 1), 
                                          scores_fname=scores_fname)    

def main_test_MOT_on_samples_with_all_examples():
    data_aug_model_test = TestMOTOnSamplesWithAllExamples(number_of_view_to_consider=5)
    data_aug_model_test.initialisation(config_file=best_config_file_path_normal_data_aug_2000,
                                       data_size=2000, 
                                       dataset_views_to_consider='all')
    data_aug_model_test.test_scores(save_file_name='mot_data_aug_result_on_test_set_with_all_5_omics', 
                                    data_size=2000, 
                                    views_to_consider='all')
        
    print('---------------Begining the Normal data exp---------------------------\n')
    
    data_normal_test = TestMOTOnSamplesWithAllExamples(number_of_view_to_consider=5)
    data_normal_test.initialisation(config_file=best_config_file_path_normal_normal_2000,
                                    data_size=2000, 
                                    dataset_views_to_consider='all')
    data_normal_test.test_scores(save_file_name='mot_normal_result_on_test_set_with_all_5_omics', 
                                 data_size=2000, 
                                 views_to_consider='all')

class TestMOTOnEachCancerWithSpecificOmicsTurnedOff(): 
    def initialisation(self, 
                       config_file: str = '', 
                       data_size: int = 2000, 
                       dataset_views_to_consider: str = 'all'):
        self.dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider=dataset_views_to_consider)
        _, new_test, _ = MultiomicDatasetBuilder.multiomic_data_normal_builder(dataset=self.dataset, test_size=0.2, valid_size=0.1)

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
                    # view_to_turn_off: list = ['none'],
                    targeted_cancer: int = 0):
        test_dataset = TurnOffViewsDatasetNormal(data_size=data_size,
                                                 views_to_consider=views_to_consider,
                                                 cancer_targeted=targeted_cancer)
        _, self.test, _ = MultiomicDatasetBuilder.multiomic_data_normal_builder(dataset=test_dataset, test_size=0.2, valid_size=0.1)
        # index_original_in_new_test = np.arange((len(self.test.indices)))
        # index_cancer_targeted = [idx for idx in index_original_in_new_test if self.test[idx][-2] == targeted_cancer]
        # self.new_test_indices = list(np.asarray(self.test.indices)[index_cancer_targeted])
        # self.test.indices = self.new_test_indices
        cancer_name_targeted = self.test.dataset.label_encoder.inverse_transform([targeted_cancer])[0]
        # name_of_view_to_be_turned_off = '_'.join(view_to_turn_off)
        scores_fname = os.path.join(self.all_params['fit_params']['output_path'], f'{save_file_name}_{views_to_consider}_{cancer_name_targeted}.txt')
        scores = self.trainer_model.score(dataset=self.test, 
                                          artifact_dir=self.all_params['fit_params']['output_path'], 
                                          nb_ckpts=self.all_params['predict_params'].get('nb_ckpts', 1), 
                                          scores_fname=scores_fname)    

def main_test_MOT_on_each_cancer_with_specific_omics_turned_off():
    for cancer_number in range(33):
        data_aug_model_test = TestMOTOnEachCancerWithSpecificOmicsTurnedOff()
        data_aug_model_test.initialisation(config_file=best_config_file_path_normal_data_aug_2000,
                                    data_size=2000, 
                                    dataset_views_to_consider='all')
        data_aug_model_test.test_scores(save_file_name='scores', 
                                        data_size=2000, 
                                        views_to_consider='all', 
                                        # view_to_turn_off=view_off,
                                        targeted_cancer=cancer_number)


if __name__ == "__main__":
    main_test_MOT_on_samples_with_all_examples()
         