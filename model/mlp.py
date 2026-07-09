import torch.nn as nn
import torch.nn.functional as F


class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int = 310, hidden_1: int = 64, hidden_2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x
