import torch
from torch import nn
from multiomic_modeling.models.utils.embedding import Embeddings, PositionalEncoding
from multiomic_modeling.models.utils import init_params_xavier_uniform, EncoderState, generate_padding_mask
from multiomic_modeling import logging
from multiomic_modeling.data.structs import Sequence

logger = logging.create_logger(__name__)


class TorchSeqTransformerDecoder(nn.Module):
    def __init__(self, token_encoder, d_model=1024, d_ff=1024, n_heads=16, n_layers=2,
                 dropout=0.1, activation="relu", pos_encoding=None, embedding=None, embedding_dropout=0):
        super().__init__()
        self.token_encoder = token_encoder
        self.vocab_size = len(token_encoder)
        self.pad_token = token_encoder.TOKEN_PAD_INT

        if pos_encoding is None:
            self.pos_encoding = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoding = pos_encoding

        if embedding is None:
            self.embedding = Embeddings(
                d_model, self.vocab_size, embedding_dropout, self.pad_token,
            )
        else:
            self.embedding = embedding

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers, norm=decoder_norm)

        self.output = nn.Linear(d_model, self.vocab_size)

        init_params_xavier_uniform(self)

    def forward(self, target: Sequence, enc_state: EncoderState):
        target = target.data
        mask_subsequent_y = self.generate_square_subsequent_mask(target.shape[1], target.device)
        mask_padding_y = generate_padding_mask(target, self.pad_token)

        target = self.embedding(target)
        target = target.transpose(0, 1)
        target = self.pos_encoding(target)

        # print(enc_state.memory)
        # print(enc_state.mask_padding_x)
        x = self.decoder(
            target,
            enc_state.memory,
            tgt_mask=mask_subsequent_y,
            memory_mask=enc_state.mask_x,
            tgt_key_padding_mask=mask_padding_y,
            memory_key_padding_mask=enc_state.mask_padding_x,
        )

        x = x.transpose(0, 1)
        x = self.output(x)

        # print(torch.argmax(x, dim=-1)[0])
        # exit(222)
        return x

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
