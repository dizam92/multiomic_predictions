import torch
from torch import nn
from multiomic_modeling.models.utils.embedding import Embeddings, PositionalEncoding
from multiomic_modeling.models.utils import init_params_xavier_uniform, init_params_xavier_normal, EncoderState
from multiomic_modeling.data.structs import Sequence
from multiomic_modeling import logging

logger = logging.create_logger(__name__)


class TorchSeqTransformerEncoder(nn.Module):
    def __init__(self, d_input, d_model=1024, d_ff=1024, n_heads=16, n_layers=2, dropout=0.1):
        super(TorchSeqTransformerEncoder, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_layers = n_layers
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Linear(self.d_input, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout, activation="relu")
        encoder_norm = nn.LayerNorm(d_model)
        self.net = nn.TransformerEncoder(encoder_layer, self.n_layers, encoder_norm)

        init_params_xavier_uniform(self)

    def forward(self, inputs) -> EncoderState:
        mask_padding_x = ~inputs[1]
        inputs = inputs[0].float()
        
        x = self.embedding(inputs)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        # print(x.device, self.embedding.lut.weight.device)

        memory = self.net(x, src_key_padding_mask=mask_padding_x)

        return EncoderState(memory=memory, mask_padding_x=mask_padding_x)