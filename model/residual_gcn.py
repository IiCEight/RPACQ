import torch
import torch.nn as nn
from einops import einsum, rearrange


class SampleAdaptiveAdj(nn.Module):
    """
    Per-sample adjacency modulation via Q·K^T attention.
    A_final[b] = A_global + α × tanh(Q[b]·K[b]^T / √d)
    α starts at 0 (pure global) and learns how much sample-specificity helps.
    """

    def __init__(self, in_features: int, num_electrodes: int, proj_dim: int = 4):
        super().__init__()
        self.query = nn.Conv2d(in_features, proj_dim, kernel_size=1, bias=False)
        self.key   = nn.Conv2d(in_features, proj_dim, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj_global):
        """
        x:          [B, C, 1, N]
        adj_global: [N, N]
        returns:    [B, N, N]
        """
        Q = self.query(x).squeeze(2)           # [B, d, N]
        K = self.key(x).squeeze(2)             # [B, d, N]
        scale = Q.size(1) ** 0.5
        A_sample = torch.tanh(Q.permute(0, 2, 1) @ K / scale)  # [B, N, N]
        adj = adj_global.unsqueeze(0) + self.alpha * A_sample
        return torch.relu(adj)                  # [B, N, N]


class ResidualGCN(nn.Module):
    def __init__(self, feature_num: int):
        super().__init__()
        self.GConv1 = nn.Conv2d(feature_num, feature_num, kernel_size=(1, 3),
                                stride=(1, 1), padding=(0, 0),
                                groups=feature_num, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_num)
        self.GConv2 = nn.Conv2d(feature_num, feature_num, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 1),
                                groups=feature_num, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_num)
        self.ELU = nn.ELU(inplace=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, adj_batch):
        """
        x:         [B, C, 1, N]
        adj_batch: [B, N, N]
        """
        deg = adj_batch.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        adj_normalized = adj_batch / deg
        residual = x
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        y = torch.einsum('bcjk,bkp->bcjp', x, adj_normalized)
        return self.ELU(torch.add(y, residual))


class MulipleResidualGCN(nn.Module):
    def __init__(self, layers: int, chan_num: int, feature_num: int):
        super().__init__()
        self.chan_num = chan_num
        self.feature_num = feature_num
        self.adaptive_adj = SampleAdaptiveAdj(
            in_features=feature_num,
            num_electrodes=chan_num,
            proj_dim=4,
        )
        self.residual_gcn_layers = nn.ModuleList(
            [ResidualGCN(feature_num=feature_num) for _ in range(layers)]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, adj=None):
        adj_batch = self.adaptive_adj(x, adj)  # [B, N, N]
        outputs = [x]
        for layer in self.residual_gcn_layers:
            x = layer(outputs[0], adj_batch)
            outputs.append(x)
        return torch.cat(outputs, dim=1), adj_batch
