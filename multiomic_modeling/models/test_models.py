from multiomic_modeling.models.trainer import *
import json
local_run_path = '/Users/maoss2/PycharmProjects/multiomic_predictions/results/local_runs'
path_to_repertory = f'{local_run_path}/9c8ed60cb8b256b8842b72d8a2defbb05b734c4b'
config_file_path = f'{path_to_repertory}/config.json'
dataset = MultiomicDataset(views_to_consider='all')
train, valid, test = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)

with open(config_file_path, 'r') as f:
    all_params = json.load(f)

random.seed(all_params['seed'])
np.random.seed(all_params['seed'])
torch.manual_seed(all_params['seed'])
model = MultiomicTrainer(Namespace(**all_params['model_params']))
scores_fname = os.path.join(all_params['fit_params']['output_path'], "naive_scores.txt")
scores = model.score(dataset=test, artifact_dir=all_params['fit_params']['output_path'], nb_ckpts=all_params['predict_params'].get('nb_ckpts', 1), scores_fname=scores_fname)