import torch.nn as nn
import torch


class MeteoModel(nn.Module):
    def __init__(self, config):
        super(MeteoModel, self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=config['hidden_units'])
        self.fc2 = nn.Linear(config['hidden_units'], 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(p=config['dropout_rate'])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return torch.softmax(x, dim=1)
