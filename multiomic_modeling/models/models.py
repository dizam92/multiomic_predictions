from multiomic_modeling.models.base import Model, CustomModelCheckpoint
from multiomic_modeling.models.encoder import TorchSeqTransformerEncoder
from multiomic_modeling.models.decoder import TorchSeqTransformerDecoder
import torch
import numpy as np
from multiomic_modeling.torch_utils import to_numpy
torch.autograd.set_detect_anomaly(True)
class MultiomicPredictionModel(Model):
    def __init__(self, d_input_enc, nb_classes_dec, class_weights, 
                 d_model_enc=1024, d_ff_enc=1024, n_heads_enc=16, n_layers_enc=2,
                 d_model_dec=1024, d_ff_dec=1024, n_heads_dec=16, n_layers_dec=2,
                 activation="relu", dropout=0.1, loss: str = 'ce'):
        super(MultiomicPredictionModel, self).__init__()
        self.encoder = TorchSeqTransformerEncoder(d_input=d_input_enc, d_model=d_model_enc, d_ff=d_ff_enc, 
                                                  n_heads=n_heads_enc, n_layers=n_layers_enc, dropout=dropout)
        self.decoder = TorchSeqTransformerDecoder(nb_classes=nb_classes_dec, d_model=d_model_dec, d_ff=d_ff_dec, 
                                                  n_heads=n_heads_dec, n_layers=n_layers_dec, dropout=dropout, activation=activation)
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
        return self.__loss(preds, targets)
    