import os
import torch
import shutil
import numpy as np
from argparse import Namespace

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts

from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer, LightningModule
# from pytorch_lightning.core.step_result import EvalResult, TrainResult
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from multiomic_modeling import logging
from multiomic_modeling.torch_utils import totensor, get_optimizer
from multiomic_modeling.models.utils import c_collate
from multiomic_modeling.loss_and_metrics import SeqCrossEntropyLoss, SeqLabelSmoothingLoss, _adjust_shapes
from multiomic_modeling.data.structs import Sequence


logger = logging.create_logger(__name__)


class Model(nn.Module):
    def predict(self, inputs, target=None):
        """Will be used for training and testing.
        Return: (preds, reals, loss, outputs, target)
        """
        raise NotImplementedError

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, file_path: str) -> None:
        """Save the model weights to disk."""
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        """Load the model weights from disk."""
        self.load_state_dict(torch.load(file_path, map_location=self.device))

    def load_average_weights(self, file_paths) -> None:
        state = {}
        for file_path in file_paths:
            state_new = torch.load(file_path, map_location=self.device)
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


class CustomModelCheckpoint(ModelCheckpoint):
    def on_validation_epoch_end(self, trainer, pl_module):
        print("In on_validation_epoch_end")
        self.save_checkpoint(trainer, pl_module)


class BaseTrainer(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (dict, Namespace)):
            hparams = args[0]
        elif len(args) == 0 and len(kwargs) > 0:
            hparams = kwargs.pop('hparams', kwargs)
        else:
            raise Exception("Expection a dict or Namespace as inputs")

        if isinstance(hparams, dict):
            self.batch_size = hparams.pop('batch_size', 32)
            self.hparams = Namespace(**hparams)

        if isinstance(hparams, Namespace):
            self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 32
            self.hparams = hparams
            hparams = vars(hparams)

        self.lr = hparams.pop('lr', 1e-3)
        self.opt = hparams.pop('optimizer', 'Adam')
        self.lr_scheduler = hparams.pop('lr_scheduler', None)
        self.n_epochs = hparams.pop('n_epochs', 100)
        self.precision = hparams.pop('precision', 32)
        self.weight_decay = hparams.pop('weight_decay', 0)
        self.valid_size = hparams.pop('valid_size', 0.2)
        self.hparams.batch_size = hparams.pop('batch_size', 32)
        self.early_stopping = hparams.pop('early_stopping', False)
        self.auto_scale_batch_size = hparams.pop('auto_scale_batch_size', None)
        self.accumulate_grad_batches = hparams.pop('accumulate_grad_batches', 1)
        self.amp_backend = hparams.pop('amp_backend', 'native')
        self.amp_level = hparams.pop('amp_level', '02')
        self.auto_lr_find = hparams.pop('auto_lr_find', False)
        self.min_batch_size = hparams.pop('min_batch_size', 32)
        self.max_batch_size = hparams.pop('max_batch_size', 2048)
        self.min_lr = hparams.pop('min_lr', 1e-6)
        self.max_lr = hparams.pop('max_lr', 1)
        self.fitted = False
        self.best_val_loss = None
        self.best_train_loss = None
        self.init_network(hparams)

    def init_network(self, hparams):
        pass

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def configure_optimizers(self):
        if hasattr(self.network, 'configure_optimizers'):
            return self.network.configure_optimizers()
        opt = get_optimizer(self.opt, filter(lambda p: p.requires_grad, self.network.parameters()),
                            lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler is None:
            scheduler = None
        elif self.lr_scheduler == "on_plateau":
            scheduler = ReduceLROnPlateau(opt, patience=3, factor=0.5, min_lr=self.lr / 1000)
        elif self.lr_scheduler == "cyclic":
            scheduler = CyclicLR(opt, base_lr=self.lr, max_lr=self.max_lr, )
        elif self.lr_scheduler == "cos_w_restart":
            scheduler = CosineAnnealingWarmRestarts(opt, T_0=8000, eta_min=self.lr/10000)
        else:
            raise Exception("Unexpected lr_scheduler")
        return {'optimizer': opt, 'lr_scheduler': scheduler, "monitor": "train_loss"}

    def train_val_step(self, batch, optimizer_idx=0, train=True):
        if hasattr(self.network, 'train_val_step'):
            return self.network.train_val_step(batch, optimizer_idx)
        if len(batch) == 2:
            xs, ys = batch
        elif len(batch) == 3:
            xs, ys, groups = batch
        else:
            raise Exception("Was expecting a list or tuple of 2 or 3 elements")

        ys_pred = self.network(xs)
        loss_metrics = self.network.compute_loss_metrics(ys_pred, ys)
        prefix = 'train_' if train else 'val_'
        for key, value in loss_metrics.items():
            self.log(prefix+key, value, prog_bar=True)
        # res = {(prefix + 'loss'): loss_metrics.get('loss'), 'log': loss_metrics, 'progress_bar': loss_metrics}

        return loss_metrics.get('loss')

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        res = self.train_val_step(batch, train=True)
        result = dict(loss=res)
        return result

    def validation_step(self, batch, *args, **kwargs):
        result = self.train_val_step(batch, train=False)
        self.log('checkpoint_on', result)
        self.log('early_stop_on', result)
        # result = EvalResult(checkpoint_on=result, early_stop_on=result)
        # print(result)
        return result

    def train_dataloader(self):
        bs = self.batch_size
        return DataLoader(self._train_dataset, batch_size=bs, shuffle=True, collate_fn=c_collate, num_workers=4)

    def val_dataloader(self):
        bs = self.batch_size
        return DataLoader(self._valid_dataset, batch_size=bs, shuffle=True, collate_fn=c_collate, num_workers=4)

    def fit(self, train_dataset=None, valid_dataset=None, artifact_dir=None, nb_ckpts=1, verbose=0, **kwargs):
        self._train_dataset, self._valid_dataset = train_dataset, valid_dataset

        def get_trainer():
            callbacks = [EarlyStopping(patience=10)] if self.early_stopping else []
            if artifact_dir is not None:
                logger = TestTubeLogger(save_dir=artifact_dir, name='logs', version=1)
                checkpoint = ModelCheckpoint(filename='{epoch}--{val_loss:.2f}', monitor="checkpoint_on",
                                             dirpath=os.path.join(artifact_dir, 'checkpoints'),
                                             verbose=False, mode='min', save_top_k=nb_ckpts, prefix='', save_last=False)
                callbacks.append(checkpoint)
            else:
                logger = verbose > 0
            res = Trainer(gpus=(1 if torch.cuda.is_available() else 0),
                          max_epochs=self.n_epochs,
                          # profiler="advanced",
                          logger=logger,
                          default_root_dir=artifact_dir,
                          progress_bar_refresh_rate=int(verbose > 0),
                          accumulate_grad_batches=self.accumulate_grad_batches,
                          # checkpoint_callback=checkpoint,
                          callbacks=callbacks,
                          auto_scale_batch_size=self.auto_scale_batch_size,
                          auto_lr_find=self.auto_lr_find,
                          amp_backend=self.amp_backend,
                          amp_level=self.amp_level,
                          precision=(self.precision if torch.cuda.is_available() else 32),
                          )
            return res

        trainer = get_trainer()
        tuner = Tuner(trainer)
        if (self.auto_scale_batch_size is not None) and self.auto_scale_batch_size:
            self.hparams.batch_size = tuner.scale_batch_size(self, steps_per_trial=5, init_val=self.min_batch_size,
                                                             max_trials=int(np.log2(self.max_batch_size/self.min_batch_size)))

        if self.hparams.get('auto_lr_find', False):
            lr_finder_res = tuner.lr_find(self, min_lr=self.hparams.get('min_lr', 1e-6),
                                          max_lr=self.hparams.get('max_lr', 1e-1),
                                          num_training=50, early_stop_threshold=None)
            print(lr_finder_res.results)

        trainer = get_trainer()
        trainer.fit(self)
        self.fitted = True
        return self

    def predict(self, dataset=None):
        ploader = DataLoader(dataset, collate_fn=c_collate, batch_size=32)
        res = [self.network.predict(x[0]).data.numpy() for x in ploader]    # supposing that the first
        return np.concatenate(res, axis=0)
