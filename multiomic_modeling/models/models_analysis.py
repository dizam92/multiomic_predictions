import argparse
import sys
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
    def build_examples_per_cancer(data_size: int = 2000) -> tuple[list, list]:
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
        attention_weights_per_layer_for_cancer_list = weights_analysis_object.get_attention_weights(trainer=trainer_model, inputs_list=cancer_list)
        batch_examples_weigths_for_cancer_list = torch.mean(attention_weights_per_layer_for_cancer_list, dim=0) # matrice [len(cancer_list) * nb_view * nb_view]
        print(cancer_name)
        print(batch_examples_weigths_for_cancer_list.shape)
        weights_analysis_object.plot_attentions_weights_per_batch(batch_weights=batch_examples_weigths_for_cancer_list, 
                                                                output_path=output_path, 
                                                                fig_name=cancer_name, 
                                                                columns_names=['cnv', 'methyl_450', 'mirna', 'rna', 'protein'])
        
def test_trainer_models_on_different_views(config_file: str, algo_type: str = 'normal', data_size: int = 2000, save_file_name: str = 'naive_scores'):
    views_to_consider_list = ['all', 'cnv', 'methyl', 'rna_iso', 'mirna'] 
    for views_to_consider in views_to_consider_list:
        if views_to_consider == 'mirna': dataset = MultiomicDataset(data_size=743, views_to_consider=views_to_consider)
        else: dataset = MultiomicDataset(data_size=int(data_size), views_to_consider=views_to_consider)
        _, test, _ = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
        test_data_loader =  multiomic_dataset_loader(dataset=test, batch_size=32, nb_cpus=2)
        with open(config_file, 'r') as f:
            all_params = json.load(f)
        random.seed(all_params['seed'])
        np.random.seed(all_params['seed'])
        torch.manual_seed(all_params['seed'])
        if algo_type == 'normal': trainer_model = MultiomicTrainer(Namespace(**all_params['model_params']))
        elif algo_type == 'multimodal' : trainer_model = MultiomicTrainerMultiModal(Namespace(**all_params['model_params']))
        else: raise f'The algotype: {algo_type} is not implemented'
        # scores_fname = os.path.join(all_params['fit_params']['output_path'], all_params['predict_params'].get('scores_fname', "naive_scores.txt"))
        scores_fname = os.path.join(all_params['fit_params']['output_path'], f'{save_file_name}_{views_to_consider}.txt')
        scores = trainer_model.score(dataset=test, artifact_dir=all_params['fit_params']['output_path'], nb_ckpts=all_params['predict_params'].get('nb_ckpts', 1), scores_fname=scores_fname)    

best_config_file_path_data_aug_2000 = '/'
best_config_file_path_data_aug_5000 = '/'
best_config_file_path_data_aug_10000 = '/'

best_config_file_path_multimodal_2000 = '/'
best_config_file_path_multimodal_5000 = '/'
best_config_file_path_multimodal_10000 = '/'


main_plot(config_file=best_config_file_path_data_aug_2000, algo_type='normal', 
          output_path=best_config_file_path_data_aug_2000[:-12], data_size=2000)
main_plot(config_file=best_config_file_path_data_aug_5000, algo_type='normal', 
          output_path=best_config_file_path_data_aug_5000[:-12], data_size=5000)
main_plot(config_file=best_config_file_path_data_aug_10000, algo_type='normal', 
          output_path=best_config_file_path_data_aug_10000[:-12], data_size=10000)

main_plot(config_file=best_config_file_path_multimodal_2000, algo_type='multimodal', 
          output_path=best_config_file_path_multimodal_2000[:-12], data_size=2000)
main_plot(config_file=best_config_file_path_multimodal_5000, algo_type='multimodal', 
          output_path=best_config_file_path_multimodal_5000[:-12], data_size=5000)
main_plot(config_file=best_config_file_path_multimodal_10000, algo_type='multimodal', 
          output_path=best_config_file_path_multimodal_10000[:-12], data_size=10000)

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
    test_trainer_models_on_different_views(config_file=args.config_file, data_size=args.data_size, save_file_name='naive_scores')
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
