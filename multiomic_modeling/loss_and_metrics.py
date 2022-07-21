import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, classification_report, matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score

def _adjust_shapes(pred, real, pad_token):
    if real.shape[1] > pred.shape[1]:
        new_shape = (pred.shape[0], real.shape[1], pred.shape[2])
        new_tensor = torch.zeros(new_shape, device=real.device) + pad_token
        new_tensor[:, : pred.shape[1], :] = pred
        pred = new_tensor
    elif real.shape[1] < pred.shape[1]:
        new_shape = (real.shape[0], pred.shape[1])
        new_tensor = torch.zeros(new_shape, dtype=torch.long, device=real.device) + pad_token
        new_tensor[:, : real.shape[1]] = real
        real = new_tensor

    return pred, real


class SequenceLoss(object):
    """Loss.
    This is a wrapper over the loss function from pytorch.
    It deals with different sequence length.
    Note:: For predictions that don't have the same length as the
           real value, some padding will be added enabling calculating
           the loss for validation and testing.
    """

    def __init__(self, pad_token: int, loss_module: nn.Module):
        self.pad_token = pad_token
        self._loss_module = loss_module

    def __call__(self, pred, real, weight=None):
        # print(pred.shape, real.shape)
        pred, real = _adjust_shapes(pred, real, self.pad_token)
        # print(pred.shape, real.shape)
        self._loss_module.weight = weight
        return self._loss_module(pred.transpose(-1, -2), real) / torch.sum(real != self.pad_token)


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing, pad_token):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_token = pad_token

    def forward(self, output, target):
        batch_size = output.shape[0]
        vocab_size = output.shape[1]

        target = self._squeeze_all(target)
        output = self._keep_one(output, dim=1)
        output = F.log_softmax(output, dim=1)

        mask = target == self.pad_token
        target = smooth_one_hot(target, vocab_size, self.smoothing)

        loss = F.kl_div(output, target, reduction="none")
        loss = loss.masked_fill(mask.unsqueeze(1), 0)

        return loss.sum() / batch_size

    def _squeeze_all(self, tensor):
        shape = list(tensor.shape)
        squeezed = 1
        for s in shape:
            squeezed *= s

        return tensor.reshape((squeezed))

    def _keep_one(self, tensor, dim=1):
        shape = list(tensor.shape)
        if len(shape) == 1:
            return tensor

        if dim != len(shape):
            tensor = tensor.transpose(dim, -1)

        kept = shape[dim]
        squeezed = 1
        for i in range(len(shape)):
            if i != dim:
                squeezed *= shape[i]

        return tensor.reshape((squeezed, kept))


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


class SeqCrossEntropyLoss(SequenceLoss):
    def __init__(self, pad_token: int):
        super(SeqCrossEntropyLoss, self).__init__(
            pad_token,
            torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=pad_token),
        )


class SeqLabelSmoothingLoss(SequenceLoss):
    def __init__(self, pad_token: int, smoothing: float):
        super().__init__(
            pad_token, LabelSmoothLoss(smoothing, pad_token),
        )
        
class ClfMetrics:
    @staticmethod
    def score(y_test, y_pred):
        return {
            'acc': np.round(accuracy_score(y_test, y_pred) * 100, 3),
            'prec': np.round(precision_score(y_test, y_pred, average='weighted') * 100, 3),
            'rec': np.round(recall_score(y_test, y_pred, average='weighted') * 100, 3),
            'f1_score': np.round(f1_score(y_test, y_pred, average='weighted') * 100, 3),
            'mcc_score': np.round(matthews_corrcoef(y_test, y_pred) * 100, 3)
        }
        
    @staticmethod  
    def classif_report(y_test, y_pred):
        return classification_report(y_test, y_pred)
    
    @staticmethod
    def confusion_matric_report(y_test, y_pred):
        return confusion_matrix(y_true=y_test, y_pred=y_pred)

class RegMetrics:
    @staticmethod
    def score(y_test, y_pred):
        return {
            'r2': np.round(r2_score(y_test, y_pred) * 100, 3),
            'mse': np.round(mean_squared_error(y_test, y_pred) * 100, 3),
            'mae': np.round(mean_absolute_error(y_test, y_pred) * 100, 3)
        }
    
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
