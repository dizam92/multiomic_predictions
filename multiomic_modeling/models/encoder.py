import torch
from torch import nn
from multiomic_modeling.models.utils.embedding import Embeddings, PositionalEncoding
from multiomic_modeling.models.utils import init_params_xavier_uniform,  EncoderState
from multiomic_modeling.data.structs import Sequence
from multiomic_modeling import logging


logger = logging.create_logger(__name__)


class TorchSeqTransformerEncoder(nn.Module):
    def __init__(self, token_encoder, d_model=1024, d_ff=1024, n_heads=16, n_layers=2,
                 dropout=0.1,  embedding_dropout=0):
        super().__init__()
        self.token_encoder = token_encoder
        self.vocab_size = len(token_encoder)
        self.pad_token = token_encoder.TOKEN_PAD_INT

        self.pos_encoding = PositionalEncoding(d_model, dropout)
        # print(self.pad_token, print())
        # exit(66)
        self.embedding = Embeddings(d_model, self.vocab_size, embedding_dropout, self.pad_token)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="relu")
        encoder_norm = nn.LayerNorm(d_model)
        self.net = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        init_params_xavier_uniform(self)

    def forward(self, inputs: Sequence) -> EncoderState:
        inputs = inputs.data
        # print(inputs.device)
        mask_padding_x = (inputs == self.pad_token).to(inputs.device)

        x = self.embedding(inputs)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        # print(x.device, self.embedding.lut.weight.device)

        memory = self.net(x, src_key_padding_mask=mask_padding_x)

        return EncoderState(memory=memory, mask_padding_x=mask_padding_x)