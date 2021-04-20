import torch
import numpy as np
from torch import nn
from multiomic_modeling.models.utils import FCLayer,Model,  MLP, init_params_xavier_uniform
from multiomic_modeling.torch_utils import get_activation

torch.autograd.set_detect_anomaly(True)
class MultiomicFCModel(Model):
    def __init__(self, 
                 input_size, 
                 hidden_sizes, 
                 nb_classes,
                 class_weights, 
                 activation='ReLU', 
                 loss="ce",
                 dropout=0.0):
        super(MultiomicFCModel, self).__init__()
        layers = []
        in_ = input_size
        for i, out_ in enumerate(hidden_sizes):
            self.layer = nn.Linear(in_, out_)
            self.batchnorm = nn.BatchNorm1d(out_)
            self.activation = get_activation(activation)
            self.dropout = nn.Dropout(p=dropout)
            in_ = out_
            layers.append(self.layer)
            layers.append(self.batchnorm)
            layers.append(self.activation)
            layers.append(self.dropout)
        self.layer_out = nn.Linear(in_, nb_classes) 
        layers.append(self.layer_out)
        if loss.lower() == 'ce':
            if class_weights == [] or class_weights is None:
                class_weights = torch.Tensor(np.ones(nb_classes))
            assert len(class_weights) == nb_classes, 'They must be a weights per class_weights'
            self.__loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        else:
            raise f'The error {loss} is not supported yet'

        self.net = nn.Sequential(*layers)
        
    def forward(self, inputs) -> torch.Tensor:
        res = self.net(inputs.float())
        return res
    
    def predict(self, inputs):
        return self(inputs.float())
    
    def compute_loss_metrics(self, preds, targets):
        return {'ce': self.__loss(preds, targets),
                'multi_acc': self.compute_multi_acc_metrics(preds=preds, targets=targets)
        }
    
    def compute_multi_acc_metrics(self, preds, targets):
        y_pred_softmax = torch.log_softmax(preds, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
        correct_pred = (y_pred_tags == targets).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc
    
# class MulticlassClassification(nn.Module):
#     def __init__(self, num_feature, num_class, class_weights, loss="ce"):
#         super(MulticlassClassification, self).__init__()
        
#         self.layer_1 = nn.Linear(num_feature, 512)
#         self.layer_2 = nn.Linear(512, 128)
#         self.layer_3 = nn.Linear(128, 64)
#         self.layer_out = nn.Linear(64, num_class) 
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.2)
#         self.batchnorm1 = nn.BatchNorm1d(512)
#         self.batchnorm2 = nn.BatchNorm1d(128)
#         self.batchnorm3 = nn.BatchNorm1d(64)
#         if loss.lower() == 'ce':
#             if class_weights == [] or class_weights is None:
#                 class_weights = torch.Tensor(np.ones(num_class))
#             assert len(class_weights) == num_class, 'They must be a weights per class_weights'
#             self.__loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
#         else:
#             raise f'The error {loss} is not supported yet'
#         layers = [self.layer_1, self.batchnorm1, self.relu,  
#                   self.layer_2, self.batchnorm2, self.relu, self.dropout,
#                   self.layer_3, self.batchnorm3, self.relu, self.dropout,
#                   self.layer_out]
#         print(layers)
#         self.net = nn.Sequential(*layers)
        
#     def forward(self, x):
#         # x = self.layer_1(x.float())
#         # x = self.batchnorm1(x)
#         # x = self.relu(x)
        
#         # x = self.layer_2(x)
#         # x = self.batchnorm2(x)
#         # x = self.relu(x)
#         # x = self.dropout(x)
        
#         # x = self.layer_3(x)
#         # x = self.batchnorm3(x)
#         # x = self.relu(x)
#         # x = self.dropout(x)
        
#         # x = self.layer_out(x)
        
#         # return x
#         res = self.net(x.float())
#         return res

#     def predict(self, inputs):
#         return self(inputs.float())
    
#     def compute_loss_metrics(self, preds, targets):
#         return {'ce': self.__loss(preds, targets),
#                 'multi_acc': self.compute_multi_acc_metrics(preds=preds, targets=targets)
#         }
    
#     def compute_multi_acc_metrics(self, preds, targets):
#         y_pred_softmax = torch.log_softmax(preds, dim = 1)
#         _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
#         correct_pred = (y_pred_tags == targets).float()
#         acc = correct_pred.sum() / len(correct_pred)
#         acc = torch.round(acc * 100)
#         return acc