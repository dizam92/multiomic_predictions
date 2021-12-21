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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import seaborn as sns
sns.set_theme()

best_config_file_path_normal_data_aug_2000 = '/scratch/maoss2/optuna_data_aug_output_2000/64c01d9cc9220b7fb39c2740272c1a02faff77e0/config.json' # 96.042
best_config_file_path_normal_normal_2000 = '/scratch/maoss2/optuna_normal_output_2000/ec18a0b2ca27de64e673e9dc9dfb9596970c130d/config.json' # 91.595

# best_config_file_path_multimodal_data_aug_2000 = '/scratch/maoss2/optuna_multimodal_data_aug_output_2000/e4e38101b615967fa3fed7462ac08bc88ff1b116/config.json'


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
        final_array = to_numpy(torch.nn.functional.softmax(cancer_weights * 100, dim=-1).mean(dim=0))
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

def main_plot(config_file: str, 
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
    # AttentionWeightsAnalysis.plot_tsne(list_of_examples_per_cancer=list_of_examples_per_cancer,
        #                                   list_of_cancer_names=list_of_cancer_names,
        #                                   trainer_model=trainer_model,
        #                                   plot_for_all_cancer=True,
        #                                   plot_for_all_examples_for_all_cancer=False,
        #                                   output_path=output_path,
        #                                   fig_name='all_cancer_tsne')
    AttentionWeightsAnalysis.plot_tsne(list_of_examples_per_cancer=list_of_examples_per_cancer,
                                      list_of_cancer_names=list_of_cancer_names,
                                      trainer_model=trainer_model,
                                      plot_for_all_cancer=False,
                                      plot_for_all_examples_for_all_cancer=True,
                                      output_path=output_path,
                                      fig_name=f'all_examples_tsne_{exp_type}')


if __name__ == "__main__":
    # to run this: salloc --time=03:00:00 --nodes=1  --ntasks-per-node=16 --mem=64G --account=rrg-corbeilj-ac
    main_plot(config_file=best_config_file_path_normal_normal_2000, 
            algo_type='normal', 
            exp_type='normal', 
            output_path=best_config_file_path_normal_normal_2000[:-12], 
            data_size=2000)

    main_plot(config_file=best_config_file_path_normal_data_aug_2000, 
            algo_type='normal', 
            exp_type='data_aug', 
            output_path=best_config_file_path_normal_data_aug_2000[:-12], 
            data_size=2000)

    # main_plot(config_file=best_config_file_path_multimodal_data_aug_2000, 
    #         algo_type='multimodal', 
    #         exp_type='data_aug', 
    #         output_path=best_config_file_path_multimodal_data_aug_2000[:-12], 
    #         data_size=2000)