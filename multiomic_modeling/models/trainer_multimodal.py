import os
import json
import torch
import random
import natsort
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from multiomic_modeling.models.base_multimodal import BaseMultiModalTrainer
from multiomic_modeling.data.data_loader import MultiomicDatasetDataAug, MultiomicDatasetNormal, MultiomicDatasetBuilder, SubsetRandomSampler
from multiomic_modeling.models.models import MultiomicPredictionModelMultiModal
from multiomic_modeling.models.utils import expt_params_formatter, c_collate
from multiomic_modeling.loss_and_metrics import ClfMetrics, NumpyEncoder, RegMetrics
from multiomic_modeling.utilities import params_to_hash
from multiomic_modeling.torch_utils import to_numpy, totensor, get_optimizer
from multiomic_modeling import logging
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, AdamW, \
    get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

 

logger = logging.create_logger(__name__)

# TODO: NEXT PAPER EXTENSION 
class MultiomicTrainerMultiModal(BaseMultiModalTrainer):
    name_map = dict(
        mo_model = MultiomicPredictionModelMultiModal
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def configure_optimizers(self):
        if hasattr(self.network, 'configure_optimizers'):
            return self.network.configure_optimizers()
        opt = get_optimizer(self.opt, filter(lambda p: p.requires_grad, self.network.parameters()),
                            lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "cosine_with_restarts":
            # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            #     opt, num_warmup_steps=self.number_of_steps_per_epoch*2, num_training_steps=int(1e6), num_cycles=self.n_epochs)
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                opt, num_warmup_steps=1000, num_training_steps=int(1e6), num_cycles=self.n_epochs)
        elif self.lr_scheduler == "cosine_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=1000, num_training_steps=int(1e6))
        else:
            raise Exception("Unexpected lr_scheduler")
        
        return {'optimizer': opt, 'lr_scheduler': scheduler, "monitor": "train_loss"}
    
    def init_network(self, hparams):
        self.network = MultiomicPredictionModelMultiModal(**hparams).float()

    def init_metrics(self):
        self.metrics = ()
    
    def train_val_step(self, batch, optimizer_idx=0, train=True): 
        # target views c'est les views qui ont été éteinte par notre mask donc on fait bien de retourner le bail original: 
        # utiliser le mask pour simplifier l'acces à la vue cible et tout
        xs, ys, _  = batch # xs: (data_augmentation, mask, original_data, original_mask), ys: patient_label, patient_name
        ys_pred, views_pred = self.network(xs)
        if train == True:
            mask_cible = to_numpy(xs[3]).astype(int) -  to_numpy(xs[1]).astype(int)
            mask_cible = totensor(mask_cible, gpu=True, device=None)
            mask_cible = mask_cible.cuda()
            loss_metrics = self.network.compute_loss_metrics(preds=ys_pred, 
                                                             targets=ys, 
                                                             preds_views=views_pred, 
                                                             targets_views=xs[2],
                                                             mask_cible=mask_cible)
        else:
            loss_metrics = self.network.compute_loss_metrics(preds=ys_pred, 
                                                             targets=ys, 
                                                             preds_views=views_pred, 
                                                             targets_views=xs[0],
                                                             mask_cible=xs[1])
        prefix = 'train_' if train else 'val_'
        for key, value in loss_metrics.items():
            self.log(prefix+key, value, prog_bar=True)
        return loss_metrics.get('combined_loss') # pour afficher les 2 autres loss, peut etre les retourner toutes ici une à une
    
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
            state_new = MultiomicTrainerMultiModal.load_from_checkpoint(file_path, map_location=self.device).state_dict()
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
        # Classification part: on the 1st part of the return of the predict
        res = [(patient_label, torch.argmax(self.network.predict(inputs=x)[0], dim=1))
                for i, (x, patient_label, patient_name) in tqdm(enumerate(ploader))] # classification multiclasse d'ou le argmax
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
        confusion_matrix = ClfMetrics().confusion_matric_report(y_test=new_target_data, y_pred=new_preds)
        confusion_matrix_dumped = json.dumps(confusion_matrix, cls=NumpyEncoder)
        if scores_fname is not None:
            clf_report_fname = f'{scores_fname[:-5]}_clf_report.json'
            confusion_matrix_fname = f'{scores_fname[:-5]}_confusion_report.json'
            # print(scores)
            # print(clf_report)
            with open(scores_fname, 'w') as fd:
                json.dump(scores, fd)
            with open(clf_report_fname, 'w') as fd:
                json.dump(clf_report, fd)
            with open(confusion_matrix_fname, 'w') as fd:
                json.dump(confusion_matrix_dumped, fd)
        
        # TODO: Regression part: on the 2nd part of the return of the predict. Hum normalement recuperer le x et partir de là mais would it work? 
#         reg_res = [(x[0], self.network.predict(inputs=x)[1])
#               for i, (x, patient_label, patient_name) in tqdm(enumerate(ploader))] 
#         reg_target_data, reg_preds = map(list, zip(*reg_res))
#         reg_target_data = to_numpy(reg_target_data)
#         reg_preds = to_numpy(reg_preds)
#         new_reg_preds = []
#         for reg_pred_batch in reg_preds:
#             new_reg_preds.extend(reg_pred_batch)
#         new_reg_target_data = []
#         for reg_target_data_batch in reg_target_data:
#             new_reg_target_data.extend(reg_target_data_batch)
#         reg_scores = RegMetrics().score(y_test=new_target_data, y_pred=new_preds)
#         if scores_fname is not None:
#             reg_report_fname = f'{scores_fname[:-5]}_reg_report.json'
#             with open(scores_fname, 'w') as fd:
#                 json.dump(reg_scores, fd)
#         print(reg_scores)
        return scores
    
    @staticmethod
    def run_experiment(model_params: dict, 
                       fit_params: dict, 
                       predict_params: dict, 
                       data_size: int, 
                       dataset_views_to_consider: str,
                       exp_type: str, 
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
        # data_size = 2000; dataset_views_to_consider = 'all'; seed = 42
        if exp_type == 'normal':
            dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider=dataset_views_to_consider)
            train, test, valid = MultiomicDatasetBuilder().multiomic_data_normal_builder(dataset=dataset, 
                                                                                         test_size=0.2, 
                                                                                         valid_size=0.1, 
                                                                                         random_state=seed)
        elif exp_type == 'data_aug':          
            dataset = MultiomicDatasetNormal(data_size=data_size, views_to_consider=dataset_views_to_consider)
            train, test, valid = MultiomicDatasetBuilder().multiomic_data_normal_builder(dataset=dataset, 
                                                                                         test_size=0.2, 
                                                                                         valid_size=0.1, 
                                                                                         random_state=seed)
            dataset_augmented = MultiomicDatasetDataAug(train_dataset=train, data_size=data_size, views_to_consider=dataset_views_to_consider)
            train = MultiomicDatasetBuilder.multiomic_data_aug_builder(augmented_dataset=dataset_augmented)
        else: 
            raise ValueError(f'The experiment type {exp_type} is not a valid option: choose between [normal and data_aug]')
        logger.info("Training")
        model = MultiomicTrainerMultiModal(Namespace(**model_params))
        model.fit(train_dataset=train, valid_dataset=valid, **fit_params)
        logger.info("Testing....")
        preds_fname = os.path.join(out_prefix, "naive_predictions.txt")
        scores_fname = os.path.join(out_prefix, predict_params.get('scores_fname', "naive_scores.txt"))
        scores = model.score(dataset=test, artifact_dir=out_prefix, nb_ckpts=predict_params.get('nb_ckpts', 1), scores_fname=scores_fname)
        
        return model
