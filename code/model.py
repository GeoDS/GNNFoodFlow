"""
model.py — MT-LocalizedGCN model architectures (Multi-Type Food Flow).

A single model predicts all 7 SCTG food commodity flows simultaneously via
seven task-specific hurdle heads over a shared backbone.

Architectures:
  MTEdgeMLP      — shared MLP backbone + 7 independent HurdleHeads (non-spatial baseline)
  MTLocalizedGCN — shared GCN encoder over a distance-based k-NN (localized)
                   graph + 7 independent HurdleHeads (the paper's model)

Notes:
  - forward() returns per-edge (logit, value) of shape (E, n_tasks).
  - The localized k-NN graph is stored internally as the persistent buffer
    `sparse_edge_index` (name kept for checkpoint compatibility).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class HurdleHead(nn.Module):
    """
    Hurdle model output head for one task.
    Input : (B, hidden_dim)
    Output: logit (B, 1), value (B, 1)
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.classifier(x), self.regressor(x)


# ─────────────────────────────────────────────────────────────
# Model 1 — MTEdgeMLP
# ─────────────────────────────────────────────────────────────

class MTEdgeMLP(nn.Module):
    """
    Shared MLP backbone with 7 independent task heads.

    For each directed edge (i → j):
      input = [x_i, x_j, edge_feat]   shape: (node_dim*2 + edge_dim,)
      → shared MLP → shared_repr
      → head_k(shared_repr) → (logit_k, value_k)  for k in 1..n_tasks

    forward() signature is identical to EdgeMLP.
    Output shape changes: (E,1) → (E, n_tasks).
    """

    def __init__(
        self,
        node_dim:  int,
        edge_dim:  int,
        hidden:    int   = 128,
        n_tasks:   int   = 7,
        dropout:   float = 0.2,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        in_dim = node_dim * 2 + edge_dim

        # Shared backbone — same depth as EdgeMLP
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.01),
        )

        # One independent HurdleHead per task
        self.heads = nn.ModuleList([
            HurdleHead(hidden // 2, dropout) for _ in range(n_tasks)
        ])

        # Kendall uncertainty weighting: log(σ²) per task, init=0 → σ=1 (no reweighting)
        self.log_sigma2 = nn.Parameter(torch.zeros(n_tasks))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr):
        """
        x          : (N, node_dim)
        edge_index : (2, E)
        edge_attr  : (E, edge_dim)

        Returns
        -------
        logit : (E, n_tasks)
        value : (E, n_tasks)
        """
        src, dst = edge_index
        h = torch.cat([x[src], x[dst], edge_attr], dim=1)
        shared = self.backbone(h)                           # (E, hidden//2)

        logits = []
        values = []
        for head in self.heads:
            lg, vl = head(shared)
            logits.append(lg)   # (E, 1)
            values.append(vl)   # (E, 1)

        logit = torch.cat(logits, dim=1)   # (E, n_tasks)
        value = torch.cat(values, dim=1)   # (E, n_tasks)
        return logit, value


# ─────────────────────────────────────────────────────────────
# Model 2 — MTLocalizedGCN
# ─────────────────────────────────────────────────────────────

class MTLocalizedGCN(nn.Module):
    """
    Shared GCN backbone (k-NN sparse graph) with 7 independent task heads.

    forward() signature is identical to LocalizedGCN.
    Output shape changes: (E,1) → (E, n_tasks).
    """

    def __init__(
        self,
        node_dim:  int,
        edge_dim:  int,
        hidden:    int   = 64,
        n_tasks:   int   = 7,
        dropout:   float = 0.2,
        sparse_edge_index = None,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        # Store k-NN graph as a non-trainable buffer so it's saved with the checkpoint
        # and forward() doesn't need it as an argument.
        # If None, register a placeholder — load_state_dict will overwrite it.
        if sparse_edge_index is None:
            sparse_edge_index = torch.zeros(2, 0, dtype=torch.long)
        self.register_buffer("sparse_edge_index", sparse_edge_index)

        # Shared node encoder — two GCN layers
        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden * 2)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden * 2)

        # Shared edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.LeakyReLU(0.01),
        )

        # Shared edge MLP after combining node + edge embeddings
        combined_dim = hidden * 2 * 2 + hidden   # h_orig + h_dest + edge_enc
        self.edge_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden * 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.01),
        )

        # One independent HurdleHead per task
        self.heads = nn.ModuleList([
            HurdleHead(hidden, dropout) for _ in range(n_tasks)
        ])

        # Kendall uncertainty weighting: log(σ²) per task, init=0 → σ=1 (no reweighting)
        self.log_sigma2 = nn.Parameter(torch.zeros(n_tasks))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr):
        """
        x          : (N, node_dim)
        edge_index : (2, E)   — prediction OD pairs
        edge_attr  : (E, edge_dim)

        k-NN sparse graph for GCN is stored as self.sparse_edge_index (buffer).

        Returns
        -------
        logit : (E, n_tasks)
        value : (E, n_tasks)
        """
        # Shared node embeddings via sparse GCN (k-NN graph stored as buffer)
        sei = self.sparse_edge_index
        h = F.leaky_relu(self.bn1(self.conv1(x, sei)), 0.01)
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.leaky_relu(self.bn2(self.conv2(h, sei)), 0.01)

        # Shared edge representation
        src, dst = edge_index
        edge_enc = self.edge_encoder(edge_attr)
        combined = torch.cat([h[src], h[dst], edge_enc], dim=1)
        shared   = self.edge_mlp(combined)                  # (E, hidden)

        logits = []
        values = []
        for head in self.heads:
            lg, vl = head(shared)
            logits.append(lg)
            values.append(vl)

        logit = torch.cat(logits, dim=1)   # (E, n_tasks)
        value = torch.cat(values, dim=1)   # (E, n_tasks)
        return logit, value
