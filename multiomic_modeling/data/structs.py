import torch
from pytorch_lightning.utilities.apply_func import TransferableDataType

class Sequence(TransferableDataType):
    def __init__(self, data, padding_token=0):
        self.data = data
        self.padding_token = padding_token

    def squeeze(self, *args, **kwargs):
        if len(self.data.shape) == 1:
            return self

        assert self.data.shape[0] == 1, \
            "There is more than one sequences in the object or no batch dimension, so its impossible to squeeze"
        return Sequence(self.data.squeeze(), self.padding_token)

    def __getitem__(self, item):
        assert self.batch_size != 0, "We don't handle single graph indexing yet"
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, slice):
            item = list(range(*item.indices(self.batch_size)))

        res = Sequence(self.data[item], self.padding_token)
        if isinstance(item, int):
            res = res.squeeze()
        return res

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype):
        # necessary to avoid infinite recursion
        raise RuntimeError('Cannot set the dtype explicitly. Please use module.to(new_dtype).')

    def to(self, device, **kwargs):
        self.data = self.data.to(device, **kwargs)
        return self

    def cuda(self):
        return self.to('cuda:0')

    def cpu(self):
        return self.to('cpu')

    def type(self, dst_type):
        """Casts all parameters to :attr:`dst_type`."""
        self._dtype = dst_type
        self.data = self.data.type(dst_type)
        return self

    def float(self):
        return self.type(torch.float)

    def double(self):
        return self.type(torch.double)

    def half(self):
        return self.type(torch.half)

    @property
    def batch_size(self):
        return 0 if len(self.data.shape) == 1 else self.data.shape[0]

    def __len__(self):
        return self.data.shape[-1]

    @classmethod
    def collate_fn(cls, seq_list):
        assert all([s.batch_size == 0 for s in seq_list]), "Can't collate batches together, use the cat function"
        b = len(seq_list)
        n = max([len(s) for s in seq_list])

        first = seq_list[0]
        params = dict(dtype=first.data.dtype, device=first.data.device, requires_grad=first.data.requires_grad)
        seqs = torch.zeros((b, n), **params).type_as(first.data) + first.padding_token

        for i, s in enumerate(seq_list):
            k = len(s)
            seqs[i, :k] = s.data
        return Sequence(seqs, first.padding_token)

    @classmethod
    def stack(cls, seq_list):
        return cls.collate_fn(seq_list)

    @classmethod
    def cat(cls, seq_batches_list):
        assert all([s.batch_size != 0 for s in seq_batches_list]), "Can only concatenate batches together"
        l = len(seq_batches_list)
        n = max([len(s) for s in seq_batches_list])
        b = max([s.batch_size for s in seq_batches_list])

        first = seq_batches_list[0]
        params = dict(dtype=first.data.dtype, device=first.data.device, requires_grad=first.data.requires_grad)
        seqs = torch.zeros((l, b, n), **params).type_as(first.data) + first.padding_token

        for i, s in enumerate(seq_batches_list):
            m, k = s.batch_size, len(s)
            seqs[i, :m, :k] = s.data

        seqs = seqs.reshape(-1, n)

        return Sequence(seqs, first.padding_token)

    @classmethod
    def split(cls, seqs, batch_size):
        for i in range(0, seqs.batch_size, batch_size):
            yield seqs[i:i+batch_size]


if __name__ == '__main__':
    pass