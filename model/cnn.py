import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight = None, reduction = 'mean', smoothing = 0.0, pos_weight = None):
        super().__init__(weight = weight, reduction = reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing = 0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight, pos_weight = self.pos_weight)
        
        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        
        return loss

class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype = torch.float), 
            'y' : torch.tensor(self.targets[idx, :], dtype = torch.float)            
        }
        return dct

class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype = torch.float)
        }
        return dct

class Model(nn.Module):
    """
    This defines the CNN model. Images have three channels 1,2,3. 
    Features and targets get reshaped into the three channels 
    on the basis of a pre-specified hidden size hyper-parameter.
    Batch-norm and dropout layers.  
    
    Args:
            params: (Params) contains num_features, num_targets, hidden_size
    """
    def __init__(self, params):
        super(Model, self).__init__()
        num_features = params.num_features 
        num_targets = params.num_targets 
        hidden_size = params.hidden_size
        # prob_dropout = params.prob_dropout
        cha_1 = 256 # three channels
        cha_2 = 512
        cha_3 = 512

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1, cha_2, kernel_size = 5, stride = 1, padding = 2,  bias = False), dim = None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size = 3, stride = 1, padding = 1, bias = True), dim = None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size = 3, stride = 1, padding = 1, bias = True), dim = None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_3, kernel_size = 5, stride = 1, padding = 2, bias = True), dim = None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size = 4, stride = 2, padding = 1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x) if x.size()[0] > 1 else x 
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha = 0.06)

        x = x.reshape(x.shape[0], self.cha_1, 
                      self.cha_1_reshape)

        x = self.batch_norm_c1(x) if x.size()[0] > 1 else x 
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x) if x.size()[0] > 1 else x 
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x) if x.size()[0] > 1 else x 
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x) if x.size()[0] > 1 else x 
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x) if x.size()[0] > 1 else x 
        x = self.dropout3(x)
        x = self.dense3(x)

        return x
