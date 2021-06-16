import argparse
import sys
from multiomic_modeling.models.trainer import *
from multiomic_modeling.data.data_loader import MultiomicDataset, multiomic_dataset_builder, multiomic_dataset_loader
from multiomic_modeling.torch_utils import to_numpy
import matplotlib.pyplot as plt
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
        
def get_attention_weights(trainer, inputs):
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

def plot_attentions_weights_per_batch(batch_weights, output_path='./',bidir_op="sum", fig_name='batch_', batch_number=0):
    list_ws = []
    for i in range(batch_weights.shape[0]): # la derniere batch est en mode 20 * 4 *4 donc ca plante mais bon on verra ca apres
        weights = batch_weights[i]
        ws = (weights - weights.min()) / ((weights.max() - weights.min()) + 1e-8)
        ws = (ws + ws.T)  if bidir_op == "sum" else (ws * ws.T)
        ws[ws < (torch.mean(ws) + 1.645 * torch.std(ws))] = 0 # for 95%  do mean + 1.645*std # pk 1.645???
        list_ws.append(ws)
    
    columns_names = ['cnv', 'methyl_450', 'mirna', 'rna_iso'] # 4 views c'est pour ca
    fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(11.69, 8.27))
    if batch_weights.shape[0] != 32:
        i_range = 5; j_range = 4
        ws_arrays = np.asanyarray(list_ws).reshape(i_range, j_range)
        for i in range(i_range):
            for j in range(j_range):
                if j == 0:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1, 
                                yticklabels=columns_names, ax=axes[i, j])
                elif i == 4:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1,
                                xticklabels=columns_names, ax=axes[i, j])
                elif j == 0 and i == 4:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1,
                                xticklabels=columns_names, yticklabels=columns_names, ax=axes[i, j])
                else:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1, ax=axes[i, j]) 
    else:
        i_range = 8; j_range = 4
        ws_arrays = np.asanyarray(list_ws).reshape(i_range, j_range)
        for i in range(i_range):
            for j in range(j_range):
                if j == 0:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1, 
                                yticklabels=columns_names, ax=axes[i, j])
                elif i == 7:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1,
                                xticklabels=columns_names, ax=axes[i, j])
                elif j == 0 and i == 7:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1,
                                xticklabels=columns_names, yticklabels=columns_names, ax=axes[i, j])
                else:
                    sns.heatmap(to_numpy(ws_arrays[i,j]), vmin=0, vmax=1, annot=True, linewidths=0.1, ax=axes[i, j]) 
    fig.savefig(f'{output_path}/{fig_name}_{batch_number}.pdf')
    plt.close(fig)

def main_plot(config_file, output_path='./', data_size=2000):
    dataset = MultiomicDataset(data_size=int(data_size), views_to_consider='all', type_of_model='transformer', complete_dataset=False)
    _, test, _ = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
    test_data_loader =  multiomic_dataset_loader(dataset=test, batch_size=32, nb_cpus=2)
    with open(config_file, 'r') as f:
        all_params = json.load(f)
    random.seed(all_params['seed'])
    np.random.seed(all_params['seed'])
    torch.manual_seed(all_params['seed'])
    trainer_model = MultiomicTrainer(Namespace(**all_params['model_params']))
    for idx, test_data in enumerate(iter(test_data_loader)):
        # batch_examples = next(iter(test_data))[0]
        batch_examples = test_data[0]
        attention_weights_per_layer = get_attention_weights(trainer=trainer_model, inputs=batch_examples)
        batch_examples_weigths = torch.sum(attention_weights_per_layer, dim=0) # matrice [32 * 4 * 4]
        plot_attentions_weights_per_batch(batch_weights=batch_examples_weigths, output_path=output_path, 
                                          bidir_op="sum", fig_name='batch_', batch_number=idx)

def test_trainer_models_on_different_views(config_file, data_size=2000, save_file_name='naive_scores'):
    views_to_consider_list = [ 'all', 'cnv', 'methyl', 'rna_iso', 'cnv_methyl_rna', 'cnv_methyl_mirna', 
                              'methyl_mirna_rna', 'cnv_mirna_rna', 'cnv_mirna', 'cnv_rna', 'cnv_methyl', 'mirna_rna', 'methyl_mirna', 'methyl_rna'
                              ] # mirna match pas a cause de la taille du dataset (on doit probablement pad les 743 pour atteindre les 2000/5000/10000)
    for views_to_consider in views_to_consider_list:
        dataset = MultiomicDataset(data_size=int(data_size), views_to_consider=views_to_consider, type_of_model='transformer', complete_dataset=False)
        _, test, _ = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
        test_data_loader =  multiomic_dataset_loader(dataset=test, batch_size=32, nb_cpus=2)
        with open(config_file, 'r') as f:
            all_params = json.load(f)
        random.seed(all_params['seed'])
        np.random.seed(all_params['seed'])
        torch.manual_seed(all_params['seed'])
        trainer_model = MultiomicTrainer(Namespace(**all_params['model_params']))
        # scores_fname = os.path.join(all_params['fit_params']['output_path'], all_params['predict_params'].get('scores_fname', "naive_scores.txt"))
        scores_fname = os.path.join(all_params['fit_params']['output_path'], f'{save_file_name}_{views_to_consider}.txt')
        scores = trainer_model.score(dataset=test, artifact_dir=all_params['fit_params']['output_path'], nb_ckpts=all_params['predict_params'].get('nb_ckpts', 1), scores_fname=scores_fname)    

best_config_file_path_2000 = '/home/maoss2/scratch/optuna_test_output_2000/0d5978314e7252b03bbd8515afa42b7c8c1621fc/config.json'
best_config_file_path_5000 = '/home/maoss2/scratch/optuna_test_output_5000/7ba5aef75ab91cd2d985129435911522df496730/config.json'
best_config_file_path_10000 = '/home/maoss2/scratch/optuna_test_output_10000/5e5bccac73c3be92f11dd69011a15f2a4df03ca4/config.json'

# main_plot(config_file=best_config_file_path_2000, output_path='/home/maoss2/scratch/optuna_test_output_2000/0d5978314e7252b03bbd8515afa42b7c8c1621fc', data_size=2000)
# main_plot(config_file=best_config_file_path_5000, output_path='/home/maoss2/scratch/optuna_test_output_5000/7ba5aef75ab91cd2d985129435911522df496730', data_size=5000)
# main_plot(config_file=best_config_file_path_5000, output_path='/home/maoss2/scratch/optuna_test_output_10000/5e5bccac73c3be92f11dd69011a15f2a4df03ca4', data_size=5000)

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




# transposer dimension 1/0
# l'idee est de faire la somme a travers tous les layers (bon j'ai vu on pouvais aussi generer par layer huh)
# weights doit etre sum sur tous les layers de l'attention (donc weight c'est matrice carré 4 * 4)
# def draw_reactants_5(weights, bidir_op="sum"):
#     # ws = (weights - weights.min(keepdims=True)) / ((weights.max(keepdims=True) - weights.min(keepdims=True)) + 1e-8)
#     # ws = (ws + ws.transpose())  if bidir_op == "sum" else (ws * ws.transpose())
#     ws = (weights - weights.min()) / ((weights.max() - weights.min()) + 1e-8)
#     ws = (ws + ws.T)  if bidir_op == "sum" else (ws * ws.T)
#     atom_ws = ws.sum(-1)
#     # afficher ici comme une matrice et mettre le nom des collones et des vues
#     ws[ws < (np.mean(ws) + 1.645 * np.std(ws))] = 0 # for 95%  do mean + 1.645*std
#     atom_threshold = np.mean(atom_ws) + 1.645 * np.std(atom_ws)
#     attention_plots = []
#     for pos, idx in enumerate([i for i in np.argsort(atom_ws)[::-1] if atom_ws[i] > atom_threshold]):
#         top_i = __draw_mol_with_att_highlights__(reactants, idx, ws[idx])
#         attention_plots.append(top_i)
#         __draw_rxn_from_imgs__([top_i], [product_image], directory + f"__top{pos+1}_attention.png")
