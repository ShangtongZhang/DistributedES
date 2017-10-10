import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel:
    def get_weight(self):
        weight = []
        for param in self.parameters():
            weight.append(param.data.numpy().flatten())
        weight = np.concatenate(weight, 0)
        return weight

    def set_weight(self, solution):
        offset = 0
        for param in self.parameters():
            param_shape = param.data.numpy().shape
            param_size = np.prod(param_shape)
            src_param = solution[offset: offset + param_size]
            if len(param_shape) > 1:
                src_param = src_param.reshape(param_shape)
            param.data = torch.FloatTensor(src_param)
            offset += param_size
        assert offset == len(solution)

class StandardFCNet(nn.Module, BaseModel):
    def __init__(self, state_dim, action_dim):
        super(StandardFCNet, self).__init__()
        hidden_size = 64

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = Variable(torch.FloatTensor(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class SingleHiddenLayerNet(nn.Module, BaseModel):
    def __init__(self, state_dim, action_dim):
        super(SingleHiddenLayerNet, self).__init__()

        hidden_size = 200
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = Variable(torch.FloatTensor(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
