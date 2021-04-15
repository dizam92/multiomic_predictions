import torch
from torch import nn
from multiomic_modeling.models.utils.embedding import Embeddings, PositionalEncoding
from multiomic_modeling.models.utils import init_params_xavier_uniform, init_params_xavier_normal,  EncoderState, generate_padding_mask
from multiomic_modeling import logging
from multiomic_modeling.data.structs import Sequence

logger = logging.create_logger(__name__)


class TorchSeqTransformerDecoder(nn.Module):
    def __init__(self, nb_classes, d_model=1024, d_ff=1024, n_heads=16, n_layers=2, dropout=0.1, activation="relu"): #dff = 4 * dmodel
        super(TorchSeqTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.nb_classes = nb_classes
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_layers = n_layers
        self.activation = activation
        
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.n_layers, norm=decoder_norm)

        self.output = nn.Linear(d_model, self.nb_classes)

        init_params_xavier_uniform(self)
        # init_params_xavier_normal(self)
        
    def forward(self, enc_state: EncoderState):
        target = torch.zeros((1, enc_state.memory.shape[1], self.d_model), device=enc_state.memory.device)

        x = self.decoder(
            target,
            enc_state.memory,
            memory_mask=enc_state.mask_x,
            memory_key_padding_mask=enc_state.mask_padding_x,
        )
        x = self.output(x)[0]
        return x