import argparse
import sys
from multiomic_modeling.models.trainer import *
from multiomic_modeling.data.data_loader import MultiomicDataset, multiomic_dataset_builder, multiomic_dataset_loader
from multiomic_modeling.torch_utils import to_numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
sns.set_theme()

# TODO: J<ai une hesitation: est-ce que je dois reconstruire un dataset uniquement de subset d'un seul type de vue? 
    #  ou bien je recupere juste les exemples avec une seule vue disponible dans ce que je veux?
class TestMultiOmicModels:
    def __init__(self, config_file: str, 
                    algo_type: str = 'normal', 
                    data_size: int = 2000, 
                    views_to_consider: str = 'all',
                    save_file_name: str = 'naive_scores') -> None:
        with open(config_file, 'r') as f:
            self.all_params = json.load(f)
        if algo_type == 'normal': trainer_model = MultiomicTrainer(Namespace(**self.all_params['model_params']))
        elif algo_type == 'multimodal' : trainer_model = MultiomicTrainerMultiModal(Namespace(**self.all_params['model_params']))
        else: raise ValueError(f'The algotype: {algo_type} is not implemented')
        if views_to_consider == 'mirna': dataset = MultiomicDataset(data_size=743, views_to_consider='mirna')
        else: dataset = MultiomicDataset(data_size=int(data_size), views_to_consider=views_to_consider)
        _, self.test, _ = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
        self.test_data_loader =  multiomic_dataset_loader(dataset=self.test, batch_size=32, nb_cpus=2)
