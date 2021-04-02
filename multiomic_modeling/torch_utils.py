import re
import six
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module
from multiomic_modeling.utilities import is_callable
from torch.nn.functional import mse_loss, cross_entropy, binary_cross_entropy, \
    binary_cross_entropy_with_logits, l1_loss
from sklearn.metrics import accuracy_score, median_absolute_error, f1_score, r2_score, roc_auc_score, mean_squared_error
from scipy.stats import pearsonr

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh',
                            'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}

OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'sparseadam': torch.optim.SparseAdam,
    'asgd': torch.optim.ASGD,
    'sgd': torch.optim.SGD,
    'rprop': torch.optim.Rprop,
    'rmsprop': torch.optim.RMSprop,
    'optimizer': torch.optim.Optimizer,
    'lbfgs': torch.optim.LBFGS
}

LOSSES_METRICS = {
    'mse': mse_loss,
    'bce': binary_cross_entropy,
    'bce_logits': binary_cross_entropy_with_logits,
    'xent': cross_entropy,
    'cross_entropy': cross_entropy,
    'l1': l1_loss,
    'rmse': mean_squared_error,
    'pearsonr': pearsonr,
    #'pearsonr_squared': pearsonr_squared,
    'f1_score': f1_score,
    'accuracy': accuracy_score,
    'roc': roc_auc_score,
    'roc_auc_score': roc_auc_score,
    'r2_score': r2_score,
    'mae': median_absolute_error
}


CLF_LOSSES = ['bce', 'bce_logits', 'xent', 'cross_entropy']
RGR_LOSSES = ['mae', 'mse', 'l1']


class GlobalMaxPool1d(nn.Module):
    # see stackoverflow
    # https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
    def __init__(self, dim=1):
        super(GlobalMaxPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class GlobalAvgPool1d(nn.Module):
    def __init__(self, dim=1):
        super(GlobalAvgPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class GlobalSumPool1d(nn.Module):
    def __init__(self, dim=1):
        super(GlobalSumPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        print(x.shape)
        res = torch.sum(x, dim=self.dim)
        print(res.shape)
        return res


class Transpose(Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
        # return x.view(x.size(0), x.size(2), x.size(1))


class ResidualBlockMaker(Module):
    def __init__(self, base_module, downsample=None):
        super(ResidualBlockMaker, self).__init__()
        self.base_module = base_module
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.base_module(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def totensor(x, gpu=False, dtype=None, device='cpu'):
    """convert a np array to tensor
       gpu option is deprecated
    """
    if isinstance(x, (int, float)):
        x = torch.Tensor([x])[0]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, list):
        x = [totensor(el) for el in x]
    elif isinstance(x, tuple):
        x = tuple([totensor(el) for el in x])
    elif isinstance(x, dict):
        x = type(x)({k: totensor(v) for k, v in x.items()})
    else:
        if dtype is not None:
            x = x.type(dtype)
        if torch.cuda.is_available() and gpu:
            if device is None:
                x = x.cuda()
            else:
                x = x.to(device=device)
    return x


def get_activation(activation):
    if is_callable(activation):
        return activation
    activation = [
        x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) > 0 and isinstance(activation[0], six.string_types), \
        'Unhandled activation function'
    if activation[0].lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation[0]]()


def get_pooling(pooling, **kwargs):
    if is_callable(pooling):
        return pooling
    # there is a reason for this to not be outside
    POOLING_MAP = {"max": GlobalMaxPool1d, "avg": GlobalAvgPool1d,
                   "sum": GlobalSumPool1d, "mean": GlobalAvgPool1d}
    return POOLING_MAP[pooling.lower()](**kwargs)


def get_optimizer(optimizer, *args, **kwargs):
    r"""
    Get an optimizer by name. cUstom optimizer, need to be subclasses of :class:`torch.optim.Optimizer`.
    Arguments
    ----------
        optimizer: :class:`torch.optim.Optimizer` or str
            A class (not an object) or a valid pytorch Optimizer name
    Returns
    -------
        optm `torch.optim.Optimizer`
            Class that should be initialized to get an optimizer.s
    """
    OPTIMIZERS = {k.lower(): v for k, v in vars(torch.optim).items()
                  if not k.startswith('__')}
    if not isinstance(optimizer, six.string_types) and issubclass(optimizer.__class__, torch.optim.Optimizer):
        return optimizer
    return OPTIMIZERS[optimizer.lower()](*args, **kwargs)


def get_loss(loss):
    assert loss.lower() in LOSSES_METRICS, f"Unknown loss '{loss}'. Options are {list(LOSSES_METRICS.keys())}"
    return LOSSES_METRICS[loss.lower()]


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.data.cpu().numpy()
    elif isinstance(x, list):
        x = [to_numpy(el) for el in x]
    elif isinstance(x, tuple):
        x = tuple([to_numpy(el) for el in x])
    elif isinstance(x, dict):
        x = {k: to_numpy(v) for k, v in x.items()}

    if isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()

    return x


if __name__ == '__main__':
    pass
