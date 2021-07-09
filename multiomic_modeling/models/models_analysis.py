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
        
def get_attention_weights(trainer, inputs_list):
    """
    Args:
        trainer, MultiomicTrainer or MultiomicTrainerMultimodal
        inputs_list, list of [[tensor_of_the_views, tensor_of_the_random_mask, tensor_of_the_originall_mask], [], ..., []]
        basically this is a list of all the examples per cancer
    Return:
        output with this shape [number_of_layer * batch_size * number_of_views * number_of_views]
    """
    inputs_list_copy = deepcopy(inputs_list)
    example_part = torch.Tensor([np.asarray(el[0]) for el in inputs_list_copy]).float()
    random_mask_part = torch.Tensor([np.asarray(el[1]) for el in inputs_list_copy]).bool()
    original_mask_part = torch.Tensor([np.asarray(el[2]) for el in inputs_list_copy]).bool()
    inputs = [example_part, random_mask_part, original_mask_part]
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

def plot_attentions_weights_per_batch(batch_weights, output_path='./', bidir_op="sum", fig_name='batch_'):
    list_ws = []
    for i in range(batch_weights.shape[0]): # la derniere batch est en mode 20 * 4 *4 donc ca plante mais bon on verra ca apres
        weights = batch_weights[i]
        ws = (weights - weights.min()) / ((weights.max() - weights.min()) + 1e-8)
        ws = (ws + ws.T)  if bidir_op == "sum" else (ws * ws.T)
        ws[ws < (torch.mean(ws) + 1.645 * torch.std(ws))] = 0 # for 95%  do mean + 1.645*std # pk 1.645???
        list_ws.append(ws)
        
        # faire lla moyenne sur tous les patients par type de cancer et enlever le filtering de zero
    
    columns_names = ['cnv', 'methyl_450', 'mirna', 'rna_iso'] # 4 views c'est pour ca
    total_number = batch_weights.shape[0]
    cpt = 0
    while total_number >= 4:
        i_range = 2; j_range = 2
        fig, axes = plt.subplots(nrows=i_range, ncols=j_range, figsize=(11.69, 8.27))
        sub_list = list_ws[:4]
        list_ws = list(set(list_ws) - set(sub_list))
        ws_arrays = np.asanyarray(sub_list).reshape(i_range, j_range)
        for i in range(i_range):
            for j in range(j_range): 
                sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1, 
                            xticklabels=columns_names, yticklabels=columns_names, ax=axes[i, j])
        fig.savefig(f'{output_path}/{fig_name}_batch_{cpt}.pdf')
        plt.close(fig)
        cpt += 1
        total_number -= 4        
    if total_number != 0:
        if total_number == 1:
            fig, axes = plt.subplots(figsize=(11.69, 8.27))
            sns.heatmap(to_numpy(list_ws[0]), vmin=0, vmax=1, annot=True, linewidths=0.1, 
                            xticklabels=columns_names, yticklabels=columns_names)
            fig.savefig(f'{output_path}/{fig_name}_batch_{cpt}.pdf')
            plt.close(fig)
        else: 
            j_range = total_number
            fig, axes = plt.subplots(nrows=j_range, figsize=(11.69, 8.27))
            for j in range(j_range):
                sns.heatmap(to_numpy(list_ws[j]), vmin=0, vmax=1, annot=True, linewidths=0.1, 
                            xticklabels=columns_names, yticklabels=columns_names, ax=axes[j])
            fig.savefig(f'{output_path}/{fig_name}_batch_{cpt}.pdf')
            plt.close(fig)
    
def build_examples_per_cancer(data_size=2000):
    dataset = MultiomicDataset(data_size=int(data_size), views_to_consider='all')
    _, test, _ = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
    test_data_loader =  multiomic_dataset_loader(dataset=test, batch_size=32, nb_cpus=2)
    list_of_cancer_names = [test.dataset.label_encoder.inverse_transform([i])[0] for i in range(33)]
    list_of_examples = [[] for _ in range(33)] # chaque sous liste va contenir les examples de ce type de cancer  
    for idx, test_data in enumerate(iter(test_data_loader)):
        batch_examples = test_data[0]
        batch_examples_labels = test_data[1]
        for sub_idx, cancer_label in enumerate(batch_examples_labels):
            list_of_examples[int(cancer_label)].append([batch_examples[0][sub_idx], batch_examples[1][sub_idx], batch_examples[2][sub_idx]])
            # chaque sous liste contient les examples sous cette forme [[tensor_of_the_views, tensor_of_the_random_mask, tensor_of_the_originall_mask], [], ..., []]
    return list_of_examples, list_of_cancer_names

def main_plot(config_file, algo_type='normal', output_path='./', data_size=2000):
    list_of_examples_per_cancer, list_of_cancer_names = build_examples_per_cancer(data_size=int(data_size)) # length of 33
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
        attention_weights_per_layer_for_cancer_list = get_attention_weights(trainer=trainer_model, inputs_list=cancer_list)
        batch_examples_weigths_for_cancer_list = torch.sum(attention_weights_per_layer_for_cancer_list, dim=0) # matrice [len(cancer_list) * 4 * 4]
        print(cancer_name)
        print(batch_examples_weigths_for_cancer_list.shape)
        plot_attentions_weights_per_batch(batch_weights=batch_examples_weigths_for_cancer_list, 
                                          output_path=output_path, bidir_op="sum", fig_name=cancer_name,)
        
def test_trainer_models_on_different_views(config_file, algo_type='normal', data_size=2000, save_file_name='naive_scores'):
    # views_to_consider_list = ['all', 'cnv', 'methyl', 'rna_iso', 'cnv_methyl_rna', 'cnv_methyl_mirna', 
    #                           'methyl_mirna_rna', 'cnv_mirna_rna', 'cnv_mirna', 'cnv_rna', 'cnv_methyl', 
    #                           'mirna_rna', 'methyl_mirna', 'methyl_rna'
    #                           ] # mirna match pas a cause de la taille du dataset (on doit probablement pad les 743 pour atteindre les 2000/5000/10000)
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

best_config_file_path_all_2000_data = '/home/maoss2/scratch/previous_results/optuna_test_output_2000/0d5978314e7252b03bbd8515afa42b7c8c1621fc/config.json'
best_config_file_path_all_5000_data = '/home/maoss2/scratch/previous_results/optuna_test_output_5000/7ba5aef75ab91cd2d985129435911522df496730/config.json'
best_config_file_path_all_10000_data = '/home/maoss2/scratch/previous_results/optuna_test_output_10000/5e5bccac73c3be92f11dd69011a15f2a4df03ca4/config.json'

best_config_file_path_all_2000_data_augmentation = '/home/maoss2/scratch/optuna_test_output_2000/f3af0f144fd1e2afee48276156d57d3f85703384/config.json'
best_config_file_path_all_5000_data_augmentation = '/home/maoss2/scratch/optuna_test_output_5000/cbd211752368f75fc6cf2984bd9dd3474c25929a/config.json'
best_config_file_path_all_10000_data_augmentation = '/home/maoss2/scratch/optuna_test_output_10000/2a8d6486786b0776edcad273876d96e22968ead3/config.json'

best_config_file_path_all_2000_data_augmentation_multimodal = '/home/maoss2/scratch/optuna_multimodal_test_output_2000/46ab0c165431876b737d36804bb926e834801517/config.json'
best_config_file_path_all_5000_data_augmentation_multimodal = '/home/maoss2/scratch/optuna_multimodal_test_output_5000/2bbf7f409f49bc0abacb7838cd5516dcc52d65f0/config.json'
best_config_file_path_all_10000_data_augmentation_multimodal = '/home/maoss2/scratch/optuna_multimodal_test_output_10000/76c336aee9bdeee6ca988f31124e75b21a8f0715/config.json'

best_config_file_path_cnv_2000 = '/home/maoss2/scratch/optuna_test_output_cnv/8c16f093126f68ffa4816991113ffbd21a8b54f8/config.json'
best_config_file_path_methyl_2000 = '/home/maoss2/scratch/optuna_test_output_methyl/dec24f31fbfb218043e2fb94071cdf4705b87242/config.json'
best_config_file_path_mirna_2000 = '/home/maoss2/scratch/optuna_test_output_mirna/360c29ed6981010e0aab7607c35f20df90199524/config.json'
best_config_file_path_rna_iso_2000 = '/home/maoss2/scratch/optuna_test_output_rna_iso/0474d2129f9fe1667a427d16aadd7fd21a8b2cb9/config.json'

main_plot(config_file=best_config_file_path_all_2000_data, algo_type='normal', output_path=best_config_file_path_all_2000_data[:-12], data_size=2000)
main_plot(config_file=best_config_file_path_all_5000_data, algo_type='normal', output_path=best_config_file_path_all_5000_data[:-12], data_size=5000)
main_plot(config_file=best_config_file_path_all_10000_data, algo_type='normal', output_path=best_config_file_path_all_10000_data[:-12], data_size=10000)

main_plot(config_file=best_config_file_path_all_2000_data_augmentation, algo_type='normal', 
          output_path=best_config_file_path_all_2000_data_augmentation[:-12], data_size=2000)
main_plot(config_file=best_config_file_path_all_5000_data_augmentation, algo_type='normal', 
          output_path=best_config_file_path_all_5000_data_augmentation[:-12], data_size=5000)
main_plot(config_file=best_config_file_path_all_10000_data_augmentation, algo_type='normal', 
          output_path=best_config_file_path_all_10000_data_augmentation[:-12], data_size=10000)

main_plot(config_file=best_config_file_path_all_2000_data_augmentation_multimodal, algo_type='multimodal', 
          output_path=best_config_file_path_all_2000_data_augmentation_multimodal[:-12], data_size=2000)
main_plot(config_file=best_config_file_path_all_5000_data_augmentation_multimodal, algo_type='multimodal', 
          output_path=best_config_file_path_all_5000_data_augmentation_multimodal[:-12], data_size=5000)
main_plot(config_file=best_config_file_path_all_10000_data_augmentation_multimodal, algo_type='multimodal', 
          output_path=best_config_file_path_all_10000_data_augmentation_multimodal[:-12], data_size=10000)

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
