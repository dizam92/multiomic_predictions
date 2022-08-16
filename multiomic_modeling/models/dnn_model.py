import torch.nn as nn
import pickle
import logging
import os
from collections import defaultdict

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from torch.utils.data import Dataset, random_split, Subset, DataLoader, SubsetRandomSampler
from multiomic_modeling.loss_and_metrics import ClfMetrics
from multiomic_modeling.data.data_loader import MultiomicDatasetDataAug, MultiomicDatasetNormal, MultiomicDatasetBuilder, SubsetRandomSampler
from multiomic_modeling.models.base import Model, CustomModelCheckpoint
from multiomic_modeling.models.trainer import *
from multiomic_modeling.models.classification_models import BaseAlgoTemplate
from multiomic_modeling.torch_utils import get_activation

class DNNDatasetBuilder:
    @staticmethod
    def dnn_dataset_builder(dataset, test_size=0.2, valid_size=0.1):
        n = len(dataset)
        idxs = np.arange(n)
        train_idx = idxs[:dataset._len_train]
        valid_idx = idxs[dataset._len_train:dataset._len_train+dataset._len_valid]
        test_idx = idxs[dataset._len_train+dataset._len_valid:]
        train_dataset = Subset(dataset, indices=train_idx)
        test_dataset = Subset(dataset, indices=test_idx)
        valid_dataset =  Subset(dataset, indices=valid_idx)
        return train_dataset, test_dataset, valid_dataset
    
class DNNDataset(Dataset):
    def __init__(self, 
                 data_size: int = 2000, 
                 views_to_consider: str = 'all', 
                 random_state: int = 42):
        super().__init__()
        x_train, y_train, x_test, y_test, feature_names = BaseAlgoTemplate.reload_dataset(data_size=data_size, 
                                                                                          dataset_views_to_consider=views_to_consider,
                                                                                          random_state=random_state)
        self.data = np.vstack((x_train, x_test))
        self.labels = self.all_patient_labels = np.hstack((y_train, y_test))
        self._len_train = 8820
        self._len_test = 2451
        self._len_valid = 981
        self.random_state = random_state
        
    def __getitem__(self, idx): 
        return self.data[idx], self.all_patient_labels[idx]
    
    def __len__(self):
        return len(self.data)
    
class DNN(Model):
    def __init__(self, 
                 input_size: int, 
                 class_weights: list,
                 output_size: int = 33, 
                 # nb_layers: int = 3, 
                 hidden_sizes: list = [512, 128, 64],
                 activation: str = 'relu',
                 dropout: float = 0.1, 
                 loss: str = 'ce', 
                 batch_norm: bool = True
                 ):  
        super(DNN, self).__init__()
        self._params = locals()
        self.nb_layers = len(hidden_sizes)
        init_layer = nn.Linear(input_size, hidden_sizes[0])
        _layers = [nn.Linear(hidden_sizes[idx], hidden_sizes[idx + 1]) for idx, _ in enumerate(hidden_sizes[:-1])]
        output_layer = nn.Linear(hidden_sizes[-1], output_size)
        _layers.insert(0, init_layer)
        self.__output_dim = output_size
        self.activation = get_activation(activation=activation)
        self._dropout = nn.Dropout(p=dropout)
        if loss.lower() == 'ce':
            if class_weights == [] or class_weights is None:
                class_weights = torch.Tensor(np.ones(output_size))
            assert len(class_weights) == output_size, 'They must be a weights per class_weights'
            self.__loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        else:
            raise f'The error {loss} is not supported yet'
        if batch_norm:
            # self._layers = [nn.Sequential(_layers[idx], self.activation, self._dropout, self._batch_norm[idx]) if idx!=0 else nn.Sequential(_layers[idx], self.activation, self._dropout) for idx in range(self.nb_layers)]
            self._batch_norm = [nn.BatchNorm1d(el) for el in hidden_sizes]
            self._layers = [nn.Sequential(_layers[idx], self.activation, self._dropout, self._batch_norm[idx]) for idx in range(self.nb_layers)]
            self._layers.append(output_layer)
            self.dnn = nn.Sequential(*self._layers).float()
        else:
            self._layers = [nn.Sequential(_layers[idx], self.activation, self._dropout) for idx in range(self.nb_layers)]
            self._layers.append(output_layer)
            self.dnn = nn.Sequential(*self._layers).float()
        
    @property
    def output_dim(self):
        return self.__output_dim
       
    def forward(self, inputs) -> torch.Tensor:
        output = self.dnn(inputs).float()
        return output
    
    def predict(self, inputs):
        return self(inputs)
            
    def compute_loss_metrics(self, preds, targets):
        return {'ce': self.__loss(preds, targets),
                'multi_acc': self.compute_multi_acc_metrics(preds=preds, targets=targets)
        }
    
    def compute_multi_acc_metrics(self, preds, targets):
        y_pred_softmax = torch.log_softmax(preds, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
        correct_pred = (y_pred_tags == targets).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc

class DNNTrainer(MultiomicTrainer):
    name_map = dict(
        mo_model = DNN
    )
    def init_network(self, hparams):
        self.network = DNN(**hparams).float()
            
    def train_val_step(self, batch, optimizer_idx=0, train=True):
        xs, ys = batch
        ys_pred = self.network(xs.float())
        loss_metrics = self.network.compute_loss_metrics(ys_pred, ys)
        prefix = 'train_' if train else 'val_'
        for key, value in loss_metrics.items():
            self.log(prefix+key, value, prog_bar=True)
        return loss_metrics.get('ce')
    
    def train_dataloader(self):
        bs = self.hparams.batch_size
        data_sampler = SubsetRandomSampler(np.arange(len(self._train_dataset)))
        res = DataLoader(self._train_dataset, batch_size=bs, sampler=data_sampler, collate_fn=c_collate, num_workers=4)
        self.number_of_steps_per_epoch = len(res)
        return res
    
    def val_dataloader(self):
        bs = self.hparams.batch_size
        data_sampler = SubsetRandomSampler(np.arange(len(self._valid_dataset)))
        return DataLoader(self._valid_dataset, batch_size=bs, sampler=data_sampler, collate_fn=c_collate, num_workers=4)

    def load_average_weights(self, file_paths) -> None:
        state = {}
        for file_path in file_paths:
            state_new = DNNTrainer.load_from_checkpoint(file_path, map_location=self.device).state_dict()
            keys = state.keys()

            if len(keys) == 0:
                state = state_new
            else:
                for key in keys:
                    state[key] += state_new[key]

        num_weights = len(file_paths)
        for key in state.keys():
            state[key] = state[key] / num_weights
        self.load_state_dict(state)
        
    def score(self, dataset, artifact_dir=None, nb_ckpts=1, scores_fname=None):
        ckpt_path = os.path.join(artifact_dir, 'checkpoints')
        ckpt_fnames = natsort.natsorted([os.path.join(ckpt_path, x) for x in os.listdir(ckpt_path)
                                         if x.endswith('.ckpt')])
        print(*ckpt_fnames)
        ckpt_fnames = ckpt_fnames[:nb_ckpts]
        self.load_average_weights(ckpt_fnames)
        batch_size = self.hparams.batch_size  
        ploader = DataLoader(dataset, collate_fn=c_collate, batch_size=batch_size, shuffle=False)
        res = [(patient_label, torch.argmax(self.network.predict(inputs=x.float()), dim=1))
                for i, (x, patient_label) in tqdm(enumerate(ploader))] # classification multiclasse d'ou le argmax
        target_data, preds = map(list, zip(*res))
        target_data = to_numpy(target_data)
        preds = to_numpy(preds)
        new_preds = []
        for pred_batch in preds:
            new_preds.extend(pred_batch)
        new_target_data = []
        for target_data_batch in target_data:
            new_target_data.extend(target_data_batch)
        scores = ClfMetrics().score(y_test=new_target_data, y_pred=new_preds)
        clf_report = ClfMetrics().classif_report(y_test=new_target_data, y_pred=new_preds)
        
        if scores_fname is not None:
            clf_report_fname = f'{scores_fname[:-5]}_clf_report.json'
            # print(scores)
            # print(clf_report)
            with open(scores_fname, 'w') as fd:
                json.dump(scores, fd)
            with open(clf_report_fname, 'w') as fd:
                json.dump(clf_report, fd)
        return scores
    
    @staticmethod
    def run_experiment(model_params: dict, 
                       fit_params: dict, 
                       predict_params: dict, 
                       data_size: int, 
                       dataset_views_to_consider: str,
                       seed: int, 
                       output_path: str, 
                       outfmt_keys=None, **kwargs):
        all_params = locals()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        keys = ['output_path', 'outfmt_keys', 'outfmt', 'save_task_specific_models', 'ckpfmt']
        for k in keys:
            if k in all_params: del all_params[k]

        print('>>> Training configuration : ')
        print(json.dumps(all_params, sort_keys=True, indent=2))
        bare_prefix = params_to_hash(all_params) if outfmt_keys is None else expt_params_formatter(all_params, outfmt_keys)
        out_prefix = os.path.join(output_path, bare_prefix)
        os.makedirs(out_prefix, exist_ok=True)
        fit_params.update(output_path=out_prefix, artifact_dir=out_prefix)
        with open(os.path.join(out_prefix, 'config.json'), 'w') as fd:
            json.dump(all_params, fd, sort_keys=True, indent=2)
        # data_size = 2000; dataset_views_to_consider = 'all'
        dataset = DNNDataset(data_size=data_size, views_to_consider=dataset_views_to_consider, random_state=seed)
        train, test, valid = DNNDatasetBuilder.dnn_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
        logger.info("Training")
        model = DNNTrainer(Namespace(**model_params))
        model.fit(train_dataset=train, valid_dataset=valid, **fit_params)
        logger.info("Testing....")
        preds_fname = os.path.join(out_prefix, "naive_predictions.txt")
        scores_fname = os.path.join(out_prefix, predict_params.get('scores_fname', "naive_scores.txt"))
        scores = model.score(dataset=test, artifact_dir=out_prefix, nb_ckpts=predict_params.get('nb_ckpts', 1), scores_fname=scores_fname)
        
        return model

