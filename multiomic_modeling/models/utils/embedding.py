import math
import torch
from torch import nn

class LinearEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, dropout):
        super(LinearEmbeddings, self).__init__()
        self.lut = nn.Linear(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, dropout, pad_token=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src):
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dim (int): embedding size
       dropout (float): dropout parameter
    """

    def __init__(self, dim, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(dim)
            )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        seq_length = emb.shape[0]
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:seq_length]  # type: ignore
        emb = self.dropout(emb)

        return emb
