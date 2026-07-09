import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from model.residual_gcn import MulipleResidualGCN
from utils.graph_construction import get_adj_from_standard
from utils.graph_construction_deap import get_deap_adj_from_standard


class GCNFeatureExtractor(nn.Module):
    def __init__(self, num_electrodes: int, num_feature: int, layers: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.chan_num = num_electrodes
        self.band_num = num_feature
        self.hidden_dim = hidden_dim

        logger.info("Using distance-based adjacency matrix for GCN.")
        if num_electrodes == 32:
            _adj_init = get_deap_adj_from_standard()
        else:
            _adj_init = get_adj_from_standard()
        self.adj = nn.Parameter(torch.tensor(_adj_init).float(), requires_grad=True)

        self.data_bn = nn.BatchNorm1d(num_feature)
        self.mrgcn = MulipleResidualGCN(layers, num_electrodes, num_feature)

        mrgcn_out_channels = (layers + 1) * num_feature
        flatten_dim = num_electrodes * mrgcn_out_channels

        self.fc1 = nn.Linear(flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), self.band_num, self.chan_num)
        x = self.data_bn(x)
        x = x.unsqueeze(2)

        g_feat, _ = self.mrgcn(x, self.adj)

        out = self.fc1(g_feat.reshape(g_feat.size(0), -1))
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        return out
