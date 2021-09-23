import argparse
import sys
from collections import defaultdict
from typing import Tuple
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from multiomic_modeling.models.trainer import *
from multiomic_modeling.data.data_loader import MultiomicDataset, multiomic_dataset_builder, multiomic_dataset_loader
from multiomic_modeling.torch_utils import to_numpy
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import seaborn as sns
sns.set_theme()

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

# (data_augmentation, mask, original_data, original_mask), patient_label
class AttentionWeightsAnalysis:    
    @staticmethod
    def build_examples_per_cancer(data_size: int = 2000) -> Tuple[list, list]:
        dataset = MultiomicDataset(data_size=data_size, views_to_consider='all')
        _, test, _ = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
        test_data_loader =  multiomic_dataset_loader(dataset=test, batch_size=32, nb_cpus=2)
        list_of_cancer_names = [test.dataset.label_encoder.inverse_transform([i])[0] for i in range(33)]
        list_of_examples = [[] for _ in range(33)] # chaque sous liste va contenir les examples de ce type de cancer  
        for test_data in iter(test_data_loader):
            batch_examples = test_data[0]
            batch_examples_labels = test_data[1]
            for sub_idx, cancer_label in enumerate(batch_examples_labels):
                list_of_examples[int(cancer_label)].append([batch_examples[0][sub_idx], 
                                                            batch_examples[1][sub_idx], 
                                                            batch_examples[2][sub_idx],
                                                            batch_examples[3][sub_idx]])
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
        data_augmentation = torch.Tensor([np.asarray(el[0]) for el in inputs_list_copy]).float()
        mask = torch.Tensor([np.asarray(el[1]) for el in inputs_list_copy]).bool()
        original_data = torch.Tensor([np.asarray(el[2]) for el in inputs_list_copy]).bool()
        original_mask = torch.Tensor([np.asarray(el[3]) for el in inputs_list_copy]).bool()
        inputs = [data_augmentation, mask, original_data, original_mask]
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
    def plot_attentions_weights_per_batch(batch_weights, 
                                        output_path: str = './', 
                                        fig_name: str = 'batch_', 
                                        columns_names: list = ['cnv', 'methyl_450', 'mirna', 'rna', 'protein']):
        final_array = to_numpy(torch.nn.functional.softmax(batch_weights * 100, dim=-1).mean(dim=0))
        fig, axes = plt.subplots(figsize=(11.69, 8.27))
        sns.heatmap(final_array, vmin=0, vmax=1, annot=True, linewidths=0.1, 
                        xticklabels=columns_names, yticklabels=columns_names)
        fig.savefig(f'{output_path}/new_plot_{fig_name}.pdf')
        plt.close(fig)
    

def main_plot(config_file: str, algo_type: str = 'normal', output_path: str = './', data_size: int = 2000):
    weights_analysis_object = AttentionWeightsAnalysis()
    list_of_examples_per_cancer, list_of_cancer_names = weights_analysis_object.build_examples_per_cancer(data_size=int(data_size)) # length of 33
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
        if not os.path.exists(f'/scratch/maoss2/{output_path}/new_plot_{cancer_name}.pdf'):
            attention_weights_per_layer_for_cancer_list = weights_analysis_object.get_attention_weights(trainer=trainer_model, inputs_list=cancer_list)
            batch_examples_weigths_for_cancer_list = torch.mean(attention_weights_per_layer_for_cancer_list, dim=0) # matrice [len(cancer_list) * nb_view * nb_view]
            print(cancer_name)
            print(batch_examples_weigths_for_cancer_list.shape) 
            weights_analysis_object.plot_attentions_weights_per_batch(batch_weights=batch_examples_weigths_for_cancer_list, 
                                                                      output_path=output_path, 
                                                                      fig_name=cancer_name, 
                                                                      columns_names=['cnv', 'methyl_450', 'mirna', 'rna', 'protein'])

best_config_file_path_data_aug_2000 = '/scratch/maoss2/optuna_data_aug_output_2000/ca6d8d29acdba33accae3bcab8f62ddfe699cd11/config.json'
best_config_file_path_data_aug_5000 = '/'
best_config_file_path_data_aug_10000 = '/'

best_config_file_path_multimodal_2000 = '/scratch/maoss2/optuna_multimodal_output_2000/e4b1ba5abbeb3f2062245a335e4afc54b587a1a5/config.json'
best_config_file_path_multimodal_5000 = '/'
best_config_file_path_multimodal_10000 = '/'

# to run this: salloc --time=03:00:00 --nodes=1  --ntasks-per-node=16 --mem=64G --account=rrg-corbeilj-ac
# main_plot(config_file=best_config_file_path_data_aug_2000, algo_type='normal', 
    #         output_path=best_config_file_path_data_aug_2000[:-12], data_size=2000)
    # main_plot(config_file=best_config_file_path_data_aug_5000, algo_type='normal', 
    #         output_path=best_config_file_path_data_aug_5000[:-12], data_size=5000)
    # main_plot(config_file=best_config_file_path_data_aug_10000, algo_type='normal', 
    #         output_path=best_config_file_path_data_aug_10000[:-12], data_size=10000)

    # main_plot(config_file=best_config_file_path_multimodal_2000, algo_type='multimodal', 
    #         output_path=best_config_file_path_multimodal_2000[:-12], data_size=2000)
    # main_plot(config_file=best_config_file_path_multimodal_5000, algo_type='multimodal', 
    #         output_path=best_config_file_path_multimodal_5000[:-12], data_size=5000)
    # main_plot(config_file=best_config_file_path_multimodal_10000, algo_type='multimodal', 
    #         output_path=best_config_file_path_multimodal_10000[:-12], data_size=10000)
            
class NewMultiomicDataset(MultiomicDataset):
    def __init__(self, data_size: int = 2000, views_to_consider: str = 'all'):
        super().__init__(data_size=data_size, views_to_consider=views_to_consider)
    
    # Essentiellement this is the same thing juste que ca retourne le nom du patient aussi
    def __getitem__(self, idx): 
        idx = idx % self.data_len_original  # pour contrer le fait que la longueur du dataset pourrait etre supérieure à l'idx samplé
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
        # The next 2 lines are just here for debug in the future: if we have a pb with the gradient it might be due to the fact there a exempales w/o views
            # patient_name_with_matrix_vide = []
            # if np.all((data == 0)): patient_name_with_matrix_vide.append([patient_name, patient_label])
        original_mask = deepcopy(mask)
        nb_views = np.sum(mask)
        if nb_views > 1:
            # TODO: We might want or need to play here to 'turn off' a certain precise view...
            n_views_to_drop = np.random.choice(nb_views - 1)
            if n_views_to_drop >= 1:
                mask[np.random.choice(np.flatnonzero(mask), size=n_views_to_drop)] = 0
        original_data = deepcopy(data.astype(float))
        data_augmentation = data.astype(float) * mask.reshape(-1, 1) # on met à zéro la vue ou les vues qu'on a dit de drop
        return (data_augmentation, mask, original_data, original_mask), patient_label, patient_name

class TestMultiomicDataset(MultiomicDataset):
    def __init__(self, data_size: int = 2000, views_to_consider: str = 'all', view_to_turn_off: str = 'aucune'):
        super().__init__(data_size=data_size, views_to_consider=views_to_consider)
        self.view_to_turn_off =  view_to_turn_off
        
    def __getitem__(self, idx): 
        idx = idx % self.data_len_original  # pour contrer le fait que la longueur du dataset pourrait etre supérieure à l'idx samplé
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
        original_data = deepcopy(data.astype(float))
        #TODO: faire les combination de views pour turn off les views......
        if self.view_to_turn_off == 'aucune': pass
        if self.view_to_turn_off == 'cnv': mask[0] = False
        if self.view_to_turn_off == 'methyl': mask[1] = False
        if self.view_to_turn_off == 'mirna': mask[2] = False
        if self.view_to_turn_off == 'rna': mask[3] = False
        if self.view_to_turn_off == 'protein': mask[4] = False
        return (original_data, mask, original_data, original_mask), patient_label

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
    
    def initialisation(self, config_file: str = '', algo_type: str = 'normal', data_size: int = 2000, dataset_views_to_consider: str = 'all'):
        self.dataset = MultiomicDataset(data_size=data_size, views_to_consider=dataset_views_to_consider)
        new_dataset = NewMultiomicDataset(data_size=data_size, views_to_consider=dataset_views_to_consider)
        _, new_test, _ = multiomic_dataset_builder(dataset=new_dataset, test_size=0.2, valid_size=0.1)
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
        assert config_file != '', 'must have a config file (from the best model ultimately'
        with open(config_file, 'r') as f:
            self.all_params = json.load(f)
        random.seed(self.all_params['seed'])
        np.random.seed(self.all_params['seed'])
        torch.manual_seed(self.all_params['seed'])
        if algo_type == 'normal': self.trainer_model = MultiomicTrainer(Namespace(**self.all_params['model_params']))
        elif algo_type == 'multimodal' : self.trainer_model = MultiomicTrainerMultiModal(Namespace(**self.all_params['model_params']))
        else: raise f'The algotype: {algo_type} is not implemented'
    
    def test_scores(self, save_file_name: str = 'naive_scores', data_size: int = 2000, views_to_consider: str = 'all', view_to_turn_off: str = 'aucune'):
        assert view_to_turn_off in ['aucune', 'protein', 'methyl', 'mirna', 'rna', 'cnv'], f'the value {view_to_turn_off} is not defined and must be in [protein, methyl, mirna, rna, cnv]'
        test_dataset = TestMultiomicDataset(data_size=data_size, 
                                            iews_to_consider=views_to_consider, 
                                            view_to_turn_off=view_to_turn_off)
        _, self.test, _ = multiomic_dataset_builder(dataset=test_dataset, test_size=0.2, valid_size=0.1)
        self.test.indices = self.new_test_indices
        scores_fname = os.path.join(self.all_params['fit_params']['output_path'], f'{save_file_name}_{views_to_consider}.txt')
        scores = self.trainer_model.score(dataset=self.test, artifact_dir=self.all_params['fit_params']['output_path'], nb_ckpts=self.all_params['predict_params'].get('nb_ckpts', 1), scores_fname=scores_fname)    
        print('The scores on all the 5 views are', scores)
        
data_aug_model_test = TestModels(number_of_view_to_consider=5)
data_aug_model_test.initialisation(config_file=best_config_file_path_data_aug_2000, 
                                   algo_type='normal', data_size=2000, dataset_views_to_consider='all')
for view_off in ['aucune', 'protein', 'methyl', 'mirna', 'rna', 'cnv']:
    print(f'view to be off is: {view_off}')
    data_aug_model_test.test_scores(save_file_name='naive_scores_temp', data_size=2000, 
                       views_to_consider='all', view_to_turn_off=view_off)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the attention weights figure")
    parser.add_argument('-p', '--config_file', type=str, default='',
                        help="""The name/path of the config file (in json format) that contains all the
                            parameters for the experiment. This config file should be at the same
                            location as the current train file""")
    parser.add_argument('-ds', '--data_size', type=str, default='2000',
                        help="""Data size for loading the right data""")
    parser.add_argument('-o', '--output_path', type=str, default='',
                        help="""location for saving the training results (model artifacts and output files).
                                If not specified, results are stored in the folder "results" at the same level as  .""")
    args = parser.parse_args()
    main_plot(config_file=args.config_file, data_size=args.data_size, output_path=args.output_path)
    # test_trainer_models_on_different_views(config_file=args.config_file, data_size=args.data_size, save_file_name='naive_scores')
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)

    