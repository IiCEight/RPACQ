import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.adversarial import DomainAdversarialLoss
from model.classifier import Discriminator
from model.mlp import MLPFeatureExtractor
from model.gcn import GCNFeatureExtractor


class _LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.0005):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        weight = logits.new_ones(logits.size()) * self.epsilon / (logits.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), 1.0 - self.epsilon)
        return (-weight * log_prob).sum(dim=-1).mean()


def source_prototypes(f_s: torch.Tensor, y_s_oh: torch.Tensor) -> torch.Tensor:
    """Per-batch L2-normalized class centroids.

    Args:
        f_s:    Source features, shape [B, D].
        y_s_oh: One-hot source labels, shape [B, C].

    Returns:
        M_s: Class prototype matrix, shape [C, D], each row on the unit sphere.
    """
    counts = y_s_oh.sum(0).clamp(min=1.0).unsqueeze(1)  # [C, 1]
    M_s = (y_s_oh.t() @ f_s) / counts                   # [C, D]
    return F.normalize(M_s, dim=1)


def sinkhorn_assignments(
    Q: torch.Tensor, M_s: torch.Tensor, lam: float = 0.05, n_iter: int = 3
) -> torch.Tensor:
    cost = (Q @ M_s.t()) / lam   # [N, C]
    P = cost.exp()
    for _ in range(n_iter):
        P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
        P = P / (P.sum(dim=0, keepdim=True) + 1e-8)
    return P


def target_prototypes(
    Q: torch.Tensor,
    M_s: torch.Tensor,
    lam: float = 0.05,
    n_iter: int = 3,
    detach_assignments: bool = True,
) -> torch.Tensor:
    if Q.size(0) < 3:
        return M_s.clone()
    Q_in = Q.detach() if detach_assignments else Q
    M_s_in = M_s.detach() if detach_assignments else M_s
    P = sinkhorn_assignments(Q_in, M_s_in, lam, n_iter)
    if detach_assignments:
        P = P.detach()
    counts = P.sum(dim=0, keepdim=True).t().clamp(min=1.0)  # [C, 1]
    M_t = (P.t() @ Q) / counts
    return F.normalize(M_t, dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, feat_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f_n = F.normalize(f, dim=1)
        W_n = F.normalize(self.W, dim=1)
        tau = self.log_tau.exp().clamp(min=1e-3)
        return (f_n @ W_n.t()) / tau


class FIFOPool:
    def __init__(self, capacity: int, feat_dim: int, device):
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.buf = torch.zeros(capacity, feat_dim, device=device)
        self.size = 0
        self.ptr = 0

    def push(self, x: torch.Tensor) -> None:
        k = x.size(0)
        for i in range(k):
            self.buf[self.ptr] = x[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def view(self) -> torch.Tensor:
        return self.buf[: self.size]


class RPACQ(nn.Module):
    def __init__(
        self,
        num_electrodes: int = 62,
        in_features: int = 5,
        num_classes: int = 3,
        use_gcn: bool = False,
        num_layers: int = 2,
        max_iter: int = 1000,
        pool_capacity: int = 256,
        sinkhorn_lambda: float = 0.05,
        sinkhorn_iters: int = 3,
        triangulation_margin: float = 0.5,
        sinkhorn_warmup_epochs: int = 1000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_gcn = use_gcn
        self.max_iter = max_iter
        self.pool_capacity = pool_capacity
        self.sinkhorn_lambda = sinkhorn_lambda
        self.sinkhorn_iters = sinkhorn_iters
        self.triangulation_margin = triangulation_margin
        self.sinkhorn_warmup_epochs = sinkhorn_warmup_epochs
        self.feat_dim = 64

        if use_gcn:
            self.feature_extractor = GCNFeatureExtractor(
                num_electrodes=num_electrodes,
                num_feature=in_features,
                layers=num_layers,
                hidden_dim=self.feat_dim,
            )
        else:
            self.feature_extractor = MLPFeatureExtractor(
                input_dim=num_electrodes * in_features,
                hidden_1=self.feat_dim,
                hidden_2=self.feat_dim,
            )

        self.classifier = CosineClassifier(feat_dim=self.feat_dim, num_classes=num_classes)
        self.domain_disc = Discriminator(in_feature=self.feat_dim, num_class=1)
        self.dann = DomainAdversarialLoss(self.domain_disc, max_iter=max_iter)
        self.smooth_ce = _LabelSmoothingCE(num_classes=num_classes)

        # Pool is allocated lazily on first forward (device unknown until then).
        self.pool: FIFOPool | None = None

        self.register_buffer("_last_M_s", torch.zeros(num_classes, self.feat_dim), persistent=False)

    def _ensure_pool(self, device) -> None:
        if self.pool is None:
            self.pool = FIFOPool(self.pool_capacity, self.feat_dim, device)

    def _quantile_schedule(self, epoch: int, max_iter: int) -> float:
        p = epoch / max(1, max_iter)
        return 0.20 + 0.30 * (2.0 / (1.0 + math.exp(-5.0 * p)) - 1.0)

    @staticmethod
    def _offdiag_max(M: torch.Tensor) -> float:
        sim = M @ M.t()
        sim = sim - torch.eye(M.size(0), device=M.device) * 2.0
        return sim.max().item()

    def _hinge(self, M: torch.Tensor) -> torch.Tensor:
        margin = self.triangulation_margin
        sim = M @ M.t()
        dist = (2.0 - 2.0 * sim).clamp_min(0.0).add(1e-8).sqrt()
        off = ~torch.eye(M.size(0), dtype=torch.bool, device=M.device)
        return F.relu(margin - dist)[off].pow(2).mean()

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_label_oh: torch.Tensor,
        epoch: int,
        max_iter: int,
    ):
        self._ensure_pool(source.device)
        f_s = self.feature_extractor(source)
        f_t = self.feature_extractor(target)
        B = f_s.size(0)

        M_s = source_prototypes(f_s, source_label_oh)
        self._last_M_s.copy_(M_s.detach())

        logits_s = self.classifier(f_s)
        src_label = source_label_oh.argmax(dim=1)
        loss_src_ce = self.smooth_ce(logits_s, src_label)

        noise = 0.005
        loss_dann = self.dann(
            f_s + noise * torch.randn_like(f_s),
            f_t + noise * torch.randn_like(f_t),
        )

        logits_t = self.classifier(f_t)
        with torch.no_grad():
            tau_no_grad = self.classifier.log_tau.exp().clamp(min=1e-3)
            p_cls = F.softmax(logits_t, dim=1)
            p_proto = F.softmax((F.normalize(f_t, dim=1) @ M_s.t()) / tau_no_grad, dim=1)
            agree = (p_cls.argmax(dim=1) == p_proto.argmax(dim=1)).float()
            geom = (p_cls * p_proto).clamp_min(1e-12).sqrt()
            d_logits = self.domain_disc(f_t)
            d_score = torch.sigmoid(d_logits).squeeze(-1)
            c_full = agree.unsqueeze(1) * geom * d_score.unsqueeze(1)
            c_score, pseudo_label = c_full.max(dim=1)
            agree_rate = agree.mean().item()

        q_pct = self._quantile_schedule(epoch, max_iter)
        k = max(1, int(q_pct * B))
        topk_idx = c_score.topk(k).indices
        self.pool.push(F.normalize(f_t[topk_idx], dim=1).detach())

        M_t = target_prototypes(
            self.pool.view(), M_s, lam=self.sinkhorn_lambda, n_iter=self.sinkhorn_iters,
            detach_assignments=(epoch < self.sinkhorn_warmup_epochs),
        )

        tau = self.classifier.log_tau.exp().clamp(min=1e-3)
        proto_logits_t = (F.normalize(f_t, dim=1) @ M_t.t()) / tau
        ce_per_sample = F.cross_entropy(proto_logits_t, pseudo_label, reduction="none")
        loss_tgt_ce = (c_score * ce_per_sample).mean()

        align = ((M_s - M_t) ** 2).sum(dim=1).mean()
        loss_tri = align + (self._hinge(M_s) + self._hinge(M_t))
        xconf_logits = (F.normalize(f_s, dim=1) @ M_t.t()) / tau
        loss_xconf = self.smooth_ce(xconf_logits, src_label)

        losses = {
            "src_ce": loss_src_ce,
            "dann": loss_dann,
            "tgt_ce": loss_tgt_ce,
            "tri": loss_tri,
            "xconf": loss_xconf,
        }
        diag = {
            "pool_size": float(self.pool.size),
            "agree_rate": agree_rate,
            "M_t_offdiag_max": self._offdiag_max(M_t),
        }
        return losses, diag

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        f = self.feature_extractor(x)
        M_s = self._last_M_s
        if self.pool is None or self.pool.size < 3:
            M = M_s
        else:
            M_t = target_prototypes(
                self.pool.view(), M_s, lam=self.sinkhorn_lambda, n_iter=self.sinkhorn_iters,
                detach_assignments=True,
            )
            M = M_s if self._offdiag_max(M_t) > 0.95 else M_t
        f_n = F.normalize(f, dim=1)
        return (f_n @ M.t()).argmax(dim=1)

    def get_parameters(self):
        return [
            {"params": self.feature_extractor.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
            {"params": self.domain_disc.parameters(), "lr_mult": 1},
        ]
