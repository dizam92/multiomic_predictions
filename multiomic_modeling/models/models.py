from multiomic_modeling.models.base import Model, CustomModelCheckpoint
from multiomic_modeling.models.encoder import TorchSeqTransformerEncoder
from multiomic_modeling.models.decoder import TorchSeqTransformerDecoder, TorchSeqTransformerDecoderViews
import torch
import numpy as np
from multiomic_modeling.torch_utils import to_numpy
torch.autograd.set_detect_anomaly(True)
class MultiomicPredictionModel(Model):
    def __init__(self, d_input_enc, nb_classes_dec, class_weights, d_model_enc_dec=1024, d_ff_enc_dec=1024, 
                 n_heads_enc_dec=16, n_layers_enc=2, n_layers_dec=2, activation="relu", dropout=0.1, loss: str = 'ce'):
        super(MultiomicPredictionModel, self).__init__()
        self.encoder = TorchSeqTransformerEncoder(d_input=d_input_enc, d_model=d_model_enc_dec, d_ff=d_ff_enc_dec, 
                                                  n_heads=n_heads_enc_dec, n_layers=n_layers_enc, dropout=dropout)
        self.decoder = TorchSeqTransformerDecoder(nb_classes=nb_classes_dec, d_model=d_model_enc_dec, d_ff=d_ff_enc_dec, 
                                                  n_heads=n_heads_enc_dec, n_layers=n_layers_dec, dropout=dropout, activation=activation)
        if loss.lower() == 'ce':
            if class_weights == [] or class_weights is None:
                class_weights = torch.Tensor(np.ones(nb_classes_dec))
            assert len(class_weights) == nb_classes_dec, 'They must be a weights per class_weights'
            self.__loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        else:
            raise f'The error {loss} is not supported yet'
        
    def forward(self, inputs) -> torch.Tensor:
        enc_res = self.encoder(inputs)
        output = self.decoder(enc_res)
        return output
    
    def predict(self, inputs):
        return self(inputs)
            
    def attention_scores(self, inputs):
        return self.encoder(inputs).attention_scores

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
    
torch.autograd.set_detect_anomaly(True)
class MultiomicPredictionModelMultiModal(Model):
    def __init__(self, d_input_enc, nb_classes_dec, class_weights, d_model_enc_dec=1024, d_ff_enc_dec=1024, 
                 n_heads_enc_dec=16, n_layers_enc=2, n_layers_dec=2, activation="relu", dropout=0.1, loss: str = 'ce'):
        super(MultiomicPredictionModelMultiModal, self).__init__()
        self.encoder = TorchSeqTransformerEncoder(d_input=d_input_enc, d_model=d_model_enc_dec, d_ff=d_ff_enc_dec, 
                                                  n_heads=n_heads_enc_dec, n_layers=n_layers_enc, dropout=dropout)
        self.decoder = TorchSeqTransformerDecoder(nb_classes=nb_classes_dec, d_model=d_model_enc_dec, d_ff=d_ff_enc_dec, 
                                                  n_heads=n_heads_enc_dec, n_layers=n_layers_dec, dropout=dropout, activation=activation)
        self.decoder_views = TorchSeqTransformerDecoderViews(d_input=d_input_enc, d_model=d_model_enc_dec, d_ff=d_ff_enc_dec, 
                                                  n_heads=n_heads_enc_dec, n_layers=n_layers_dec, dropout=dropout, activation=activation)
        if loss.lower() == 'ce':
            if class_weights == [] or class_weights is None:
                class_weights = torch.Tensor(np.ones(nb_classes_dec))
            assert len(class_weights) == nb_classes_dec, 'They must be a weights per class_weights'
            self.__loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        else:
            raise f'The error {loss} is not supported yet'
        
    def forward(self, inputs) -> torch.Tensor:
        enc_res = self.encoder(inputs)
        output = self.decoder(enc_res)
        output_views = self.decoder_views(enc_res)
        return output, output_views
    
    def predict(self, inputs):
        return self(inputs)
            
    def attention_scores(self, inputs):
        return self.encoder(inputs).attention_scores

    def compute_loss_metrics(self, preds, targets, preds_views, targets_views, mask_cible):
        
        ce_loss = self.__loss(preds, targets)
        preds_views_shape = preds_views.shape
        preds_views = preds_views.reshape(preds_views_shape[1], preds_views_shape[0], -1) 
        preds_views = preds_views * ~mask_cible.reshape(mask_cible.shape + (1,))
        targets_views = targets_views * ~mask_cible.reshape(mask_cible.shape + (1,))
        mse_loss = torch.nn.functional.mse_loss(preds_views.float(), targets_views.float()) 
        combined_loss = ce_loss + mse_loss  
        
        return {'ce': ce_loss,
                'mse': mse_loss,
                'combined_loss': combined_loss,
                'multi_acc': self.compute_multi_acc_metrics(preds=preds, targets=targets)
        }
    
    def compute_multi_acc_metrics(self, preds, targets):
        y_pred_softmax = torch.log_softmax(preds, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
        correct_pred = (y_pred_tags == targets).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc