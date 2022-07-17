import re
import math
import torch
import hashlib
from multiomic_modeling.data.structs import Sequence
from multiomic_modeling.torch_utils import get_activation
from multiomic_modeling import logging
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
# from torch._six import int_classes, string_classes, container_abcs
from torch._six import string_classes
int_classes = int
import collections.abc as container_abcs

from multiomic_modeling.utilities import flatten_dict

logger = logging.create_logger(__name__)


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class EncoderState:
    def __init__(self, inputs=None, lengths=None, embeddings=None,
                 memory=None, mask_x=None, mask_padding_x=None, attention_scores=None):
        self.inputs = inputs
        self.lengths = lengths
        self.embeddings = embeddings
        self.memory = memory
        self.mask_x = mask_x
        self.mask_padding_x = mask_padding_x
        self.attention_scores = attention_scores


class GraphAttentionOutput:
    def __init__(self, node_features=None, edge_features=None, attention_scores=None):
        self.node_features = node_features
        self.edge_features = edge_features
        self.attention_scores = attention_scores


def init_params_xavier_uniform(model: nn.Module):
    for param in model.parameters():
        if param.dim() > 1:
            xavier_uniform_(param)

def init_params_xavier_normal(model: nn.Module):
    for param in model.parameters():
        if param.dim() > 1:
            xavier_normal_(param)


class Model(nn.Module):
    def predict(self, inputs, target, loss_fn, training=True):
        """Will be used for training and testing.
        Return: (preds, reals, loss, outputs, target)
        """
        raise NotImplementedError("Will be used for training and testing but not implemented ")

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


def generate_max_seq_length(inputs: torch.Tensor, max_seq_length: int, factor=1) -> int:
    """Calculate the max seq length.
    The value is based on the inputs and the global max_seq_length.
    """
    dynamic_max_seq = math.ceil(inputs.shape[1] * factor)

    if dynamic_max_seq > max_seq_length:
        return max_seq_length

    return dynamic_max_seq


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x) + x


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer. Should be one supported by :func:`ivbase.nn.commons.get_activation`.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{node_feats_dim}}`
            (Default value = None)
    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None):
        super(FCLayer, self).__init__()
        # Although I disagree with this it is simple enough and robust
        # if we trust the user base
        self._params = locals()
        self.in_size = in_size
        self.out_size = out_size
        activation = get_activation(activation)
        linear = nn.Linear(in_size, out_size, bias=bias)
        if init_fn:
            init_fn(linear)
        layers = [linear, activation]
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        if b_norm:
            layers.append(nn.BatchNorm1d(out_size))
        self.net = nn.Sequential(*layers)

    @property
    def output_dim(self):
        return self.out_size

    @property
    def out_features(self):
        return self.out_size

    @property
    def in_features(self):
        return self.in_size

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    r"""
    Feature extractor using a Fully Connected Neural Network
    Arguments
    ----------
        input_size: int
            size of the input
        hidden_sizes: int list or int
            size of the hidden layers
        activation: str or callable
            activation function. Should be supported by :func:`ivbase.nn.commons.get_activation`
            (Default value = 'relu')
        b_norm: bool, optional):
            Whether batch norm is used or not.
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)
    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, input_size, hidden_sizes, activation='ReLU', b_norm=False, l_norm=False, dropout=0.0):
        super(MLP, self).__init__()
        self._params = locals()
        layers = []
        in_ = input_size
        if l_norm:
            layers.append(nn.LayerNorm(input_size))
        for i, out_ in enumerate(hidden_sizes):
            layer = FCLayer(in_, out_, activation=activation,
                            b_norm=b_norm and (i == (len(hidden_sizes) - 1)),
                            dropout=dropout)
            layers.append(layer)
            in_ = out_

        self.__output_dim = in_
        self.extractor = nn.Sequential(*layers)

    @property
    def output_dim(self):
        return self.__output_dim

    def forward(self, x):
        res = self.extractor(x)
        return res


def generate_padding_mask(seq, pad_token):
    return seq == pad_token


def expt_params_formatter(all_params, key_params):
    flatp = flatten_dict(all_params)
    ckpt = flatp.pop("model_params.pretrained_ckpt", None)
    if ckpt is None:
        ckpt = flatp.pop("model_params.cv_params.encoder_model_path", None)

    if isinstance(ckpt, (list, tuple)):
        ckpt = '---'.join(ckpt)

    res = '_'.join([str(flatp.get(k)) for k in key_params if k in flatp])
    if ckpt is not None:
        uid = hashlib.sha1(ckpt.encode()).hexdigest()
        res += '_' + uid

    res = res.replace(' ', '').replace('[', '').replace(']', '').replace(',', '-')
    return res


def c_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        _all_same_shape_ = all([el.shape == batch[0].shape for el in batch])
        _all_same_dtype_ = all([el.dtype == batch[0].dtype for el in batch])
        if _all_same_shape_:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        else:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return c_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, Sequence):
        return Sequence.collate_fn(batch)
    elif isinstance(elem, float):
        return torch.Tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.Tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: c_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(c_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [c_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))