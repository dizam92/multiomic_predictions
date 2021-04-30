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
from multiomic_modeling.models.mlp_models import MultiomicFCModel
from multiomic_modeling.models.utils import expt_params_formatter, c_collate
from multiomic_modeling.loss_and_metrics import ClfMetrics
from multiomic_modeling.utilities import params_to_hash
from multiomic_modeling.torch_utils import to_numpy, get_optimizer
from multiomic_modeling import logging
from torch.utils.data import DataLoader
# from pytorch_lightning.core.step_result import EvalResult, TrainResult
from transformers.optimization import Adafactor, AdamW, \
    get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
    
logger = logging.create_logger(__name__)

class MultiomicFCTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def configure_optimizers(self):
        if hasattr(self.network, 'configure_optimizers'):
            return self.network.configure_optimizers()
        opt = get_optimizer(self.opt, filter(lambda p: p.requires_grad, self.network.parameters()),
                            lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "cosine_with_restarts":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                opt, num_warmup_steps=1000, num_training_steps=int(1e6), num_cycles=self.n_epochs)
        elif self.lr_scheduler == "cosine_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=1000, num_training_steps=int(1e6))
        else:
            raise Exception("Unexpected lr_scheduler")
        
        return {'optimizer': opt, 'lr_scheduler': scheduler, "monitor": "train_loss"}
    
    def init_network(self, hparams):
        self.network = MultiomicFCModel(**hparams).float()
        # self.network = MulticlassClassification(**hparams).float()

    def init_metrics(self):
        self.metrics = ()
    
    def train_val_step(self, batch, optimizer_idx=0, train=True):
        xs, ys = batch
        ys_pred = self.network(xs)
        loss_metrics = self.network.compute_loss_metrics(ys_pred, ys)
        prefix = 'train_' if train else 'val_'
        for key, value in loss_metrics.items():
            self.log(prefix+key, value, prog_bar=True)
        return loss_metrics.get('ce')
    
    def train_dataloader(self):
        bs = self.hparams.batch_size
        data_sampler = SubsetRandomSampler(np.arange(len(self._train_dataset)))
        res = DataLoader(self._train_dataset, batch_size=bs, sampler=data_sampler, collate_fn=c_collate, num_workers=4)
        return res
    
    def val_dataloader(self):
        bs = self.hparams.batch_size
        data_sampler = SubsetRandomSampler(np.arange(len(self._valid_dataset)))
        return DataLoader(self._valid_dataset, batch_size=bs, sampler=data_sampler, collate_fn=c_collate, num_workers=4)

    def load_average_weights(self, file_paths) -> None:
        state = {}
        for file_path in file_paths:
            state_new = MultiomicFCTrainer.load_from_checkpoint(file_path, map_location=self.device).state_dict()
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
        
    def score(self, dataset: MultiomicDataset, artifact_dir=None, nb_ckpts=1, scores_fname=None):
        ckpt_path = os.path.join(artifact_dir, 'checkpoints')
        ckpt_fnames = natsort.natsorted([os.path.join(ckpt_path, x) for x in os.listdir(ckpt_path)
                                         if x.endswith('.ckpt')])
        print(*ckpt_fnames)
        ckpt_fnames = ckpt_fnames[:nb_ckpts]
        self.load_average_weights(ckpt_fnames)

        batch_size = self.hparams.batch_size  
        ploader = DataLoader(dataset, collate_fn=c_collate, batch_size=batch_size, shuffle=False)
        res = [(patient_label, torch.argmax(self.network.predict(inputs=x), dim=1))
                for i, (x, patient_label) in tqdm(enumerate(ploader))] # classification multiclasse d'ou le argmax
        target_data, preds = map(list, zip(*res))
        target_data = to_numpy(target_data)
        preds = to_numpy(preds)
        clf_metrics = ClfMetrics()
        new_preds = []
        for pred_batch in preds:
            new_preds.extend(pred_batch)
        new_target_data = []
        for target_data_batch in target_data:
            new_target_data.extend(target_data_batch)
        scores = clf_metrics.score(y_test=new_target_data, y_pred=new_preds)
        clf_report = clf_metrics.classif_report(y_test=new_target_data, y_pred=new_preds)
        
        if scores_fname is not None:
            clf_report_fname = f'{scores_fname[:-5]}_clf_report.json'
            print(scores)
            print(clf_report)
            with open(scores_fname, 'w') as fd:
                json.dump(scores, fd)
            with open(clf_report_fname, 'w') as fd:
                json.dump(clf_report, fd)
        return scores

    @staticmethod
    def run_experiment(model_params, fit_params, predict_params, dataset_views_to_consider, type_of_model,
                       complete_dataset, seed, output_path, outfmt_keys=None, **kwargs):
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

        dataset = MultiomicDataset(views_to_consider=dataset_views_to_consider, 
                                   type_of_model=type_of_model, 
                                   complete_dataset=complete_dataset)
        train, valid, test = multiomic_dataset_builder(dataset=dataset, test_size=0.2, valid_size=0.1)
        logger.info("Training")
        model = MultiomicFCTrainer(Namespace(**model_params))
        model.fit(train_dataset=train, valid_dataset=valid, **fit_params)

        logger.info("Testing....")
        preds_fname = os.path.join(out_prefix, "naive_predictions.txt")
        scores_fname = os.path.join(out_prefix, predict_params.get('scores_fname', "naive_scores.txt"))
        scores = model.score(dataset=test, artifact_dir=out_prefix, nb_ckpts=predict_params.get('nb_ckpts', 1), scores_fname=scores_fname)
