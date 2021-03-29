import os
import json
import torch
import random
import natsort
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from multiomic_modeling.models.base import BaseTrainer
from multiomic_modeling.data.data_loader import MultiomicDataset, SubsetRandomSampler, multiomic_dataset_builder
from multiomic_modeling.models.models import MultiomicPredictionModel
from multiomic_modeling.models.utils import expt_params_formatter, c_collate
from multiomic_modeling.utilities import params_to_hash
from multiomic_modeling.torch_utils import to_numpy, get_optimizer
from multiomic_modeling import logging
from torch.utils.data import DataLoader
from pytorch_lightning.core.step_result import EvalResult, TrainResult

logger = logging.create_logger(__name__)

class MultiomicTrainer(BaseTrainer):
    name_map = dict(
        mo_model = MultiomicPredictionModel
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def configure_optimizers(self):
        if hasattr(self.network, 'configure_optimizers'):
            return self.network.configure_optimizers()
        opt = get_optimizer(self.opt, filter(lambda p: p.requires_grad, self.network.parameters()),
                            lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.1, patience=10, 
                                                                   threshold=0.0001, threshold_mode='rel', 
                                                                   cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        else:
            raise Exception("Unexpected lr_scheduler")
        return {'optimizer': opt, 'lr_scheduler': scheduler, "monitor": "train_loss"}
    
    def init_network(self, hparams):
        model_name = hparams.pop('model_name')
        if model_name not in self.name_map:
            raise Exception(f'Unhandled model "{model_name}". The name of '
                            f'the model should be one of those: {list(self.name_map.keys())}')
        modelclass = self.name_map[model_name.lower()]
        self.network = modelclass(**hparams).float()
        self.model_name = model_name

    def init_metrics(self):
        self.metrics = ()
    
    def train_val_step(self, batch, optimizer_idx=0, train=True):
        xs, ys = batch
        ys_pred = self.network(xs, ys)
        loss_metrics = self.network.compute_loss_metrics(ys_pred, ys)

        prefix = 'train_' if train else 'val_'
        for key, value in loss_metrics.items():
            self.log(prefix+key, value, prog_bar=True)
        return loss_metrics.get('loss')
    
    def train_dataloader(self):
        bs = self.hparams.batch_size
        data_sampler = SubsetRandomSampler(np.arange(len(self._train_dataset)))
        return DataLoader(self._train_dataset, sampler=data_sampler, collate_fn=c_collate, num_workers=4)

    def val_dataloader(self):
        bs = self.hparams.batch_size
        data_sampler = SubsetRandomSampler(np.arange(len(self._valid_dataset)))
        return DataLoader(self._valid_dataset, sampler=data_sampler, collate_fn=c_collate, num_workers=4)

    def load_average_weights(self, file_paths) -> None:
        state = {}
        for file_path in file_paths:
            state_new = MultiomicTrainer.load_from_checkpoint(file_path, map_location=self.device).state_dict()
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
        
    # Je comprends pas ta fonction score! Et c'est tu nécessaire? Yeah c'est nécessaire mais je comprends que dalle. 
    # Je odis debug this pone
    def score(self, dataset: MultiomicDataset, artifact_dir=None, nb_ckpts=10, preds_fname=None, scores_fname=None):
        ckpt_path = os.path.join(artifact_dir, 'checkpoints')
        ckpt_fnames = natsort.natsorted([os.path.join(ckpt_path, x) for x in os.listdir(ckpt_path)
                                         if x.endswith('.ckpt')])
        print(*ckpt_fnames)
        ckpt_fnames = ckpt_fnames[:nb_ckpts]
        self.load_average_weights(ckpt_fnames)

        batch_size = self.hparams.batch_size  
        ploader = DataLoader(dataset, collate_fn=c_collate, batch_size=batch_size, shuffle=False)
        res = [(data, mask, patient_label, *self.network.predict(inputs=data))
                for i, (data, mask, patient_label) in tqdm(enumerate(ploader))]

        input_data, mask_data, target_data, preds = map(list, zip(*res))
        acc = (preds == target_data)

        res = [{k: to_numpy(v) for k, v in self.network.compute_loss_metrics(pred, y).items()}
               for pred, y in zip(preds, ys)]
        scores = {k: np.mean([el.get(k) for el in res]).item() for k in res[0].keys()}
        scores.update(vals)
        scores.update(accs)

        save_smis = np.concatenate((input_smis, target_smis, pred_smis), axis=1)
        if preds_fname is not None:
            np.savetxt(preds_fname, save_smis, fmt='%s')

        if scores_fname is not None:
            print(scores)
            with open(scores_fname, 'w') as fd:
                json.dump(scores, fd)

        dataset.dataset.return_smiles_on_iteration = False
        return scores
    
    @staticmethod
    def run_experiment(model_params, dataset_params, fit_params, predict_params,
                       seed, output_path, outfmt_keys=None, **kwargs):
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

        dataset = MultiomicDataset()
        train, valid, test = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)

        logger.info("Training")
        model = MultiomicTrainer(Namespace(**model_params))
        model.fit(train_dataset=train, valid_dataset=valid, **fit_params)

        logger.info("Naive decoding....")
        preds_fname = os.path.join(out_prefix, "naive_predictions.txt")
        scores_fname = os.path.join(out_prefix, "naive_scores.txt")
        scores = model.score(dataset=test, artifact_dir=out_prefix, nb_ckpts=predict_params.get('nb_ckpts', 1),
                             beam_size=None, preds_fname=preds_fname, scores_fname=scores_fname)


