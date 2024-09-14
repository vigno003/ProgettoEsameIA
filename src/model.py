import torch.nn as nn
import torch


class MeteoModel(nn.Module):
    def __init__(self):
        super(MeteoModel, self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)
