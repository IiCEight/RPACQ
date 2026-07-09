import torch.nn as nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, in_feature: int, num_class: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, in_feature)
        self.fc2 = nn.Linear(in_feature, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
