"""Graph autoencoder anomaly head — architecture doc sec 4.3's "Anomaly" row:
"Graph autoencoder; reconstruction error z-scored against population".

Shares argus.models.encoder's TemporalHeteroGATEncoder rather than training a
bespoke model of its own — its first real consumer, and the architecture
doc's explicit intent (sec 4.3: sharing one encoder "forces anomaly/pattern/
risk signals to reason over the SAME learned representation").

TRAINING: a standard autoencoder — encode each node, then a per-node-type
linear decoder reconstructs that node's own input features (already
z-scored by argus.models.sage's build_hetero_data, reused here via
argus.models.encoder.build_temporal_hetero_data), trained via MSE. A node
whose local neighborhood makes it hard to reconstruct (high error) is
structurally or behaviorally unlike the rest of the graph — no labels
needed, matching this head's unsupervised design in the architecture doc.

SCORING: reconstruction error is z-scored WITHIN each node type (a Wallet's
error is only meaningful relative to other Wallets — IPs/ASNs have entirely
different feature semantics, see argus.features.build's fill-strategy
docstring), then squashed to [0,1] via a sigmoid, matching every other
score head's contracted range (docs/contracts.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import igraph as ig
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from argus.models.encoder import EMBEDDING_DIM, HIDDEN_DIM, TemporalHeteroGATEncoder, build_temporal_hetero_data

MAX_EPOCHS = 30
LEARNING_RATE = 0.01


@dataclass
class AnomalyScore:
    node_id: str
    node_type: str
    score: float  # sigmoid(z_score), in [0,1]
    z_score: float
    raw_error: float
    population_mean_error: float
    population_std_error: float


class _GraphAutoencoder(nn.Module):
    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        temporal_edge_types: set[tuple[str, str, str]],
        feature_dims: dict[str, int],
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = TemporalHeteroGATEncoder(
            edge_types, temporal_edge_types, hidden_dim=hidden_dim, out_dim=embedding_dim
        )
        self.decoders = nn.ModuleDict({nt: nn.Linear(embedding_dim, dim) for nt, dim in feature_dims.items()})

    def forward(self, data) -> dict[str, torch.Tensor]:
        embeddings = self.encoder(data)
        return {nt: self.decoders[nt](emb) for nt, emb in embeddings.items() if nt in self.decoders}


def train_and_score_anomalies(
    g: ig.Graph,
    node_features: pd.DataFrame,
    hidden_dim: int = HIDDEN_DIM,
    embedding_dim: int = EMBEDDING_DIM,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LEARNING_RATE,
    seed: int = 0,
) -> list[AnomalyScore]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, node_ids = build_temporal_hetero_data(g, node_features)
    data = data.to(device)

    temporal_edge_types = {et for et in data.edge_types if et[1] in ("BROADCAST_VIA", "rev_BROADCAST_VIA")}
    feature_dims = {nt: data[nt].x.size(-1) for nt in data.node_types if data[nt].x.size(0) > 0}

    if not feature_dims or not data.edge_types:
        # No node types with any nodes, or no edges anywhere (degenerate/
        # empty graph) — nothing to encode or reconstruct.
        return []

    model = _GraphAutoencoder(data.edge_types, temporal_edge_types, feature_dims, hidden_dim, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        recon = model(data)
        loss = sum(F.mse_loss(recon[nt], data[nt].x) for nt in recon)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(data)

    scores: list[AnomalyScore] = []
    for node_type, ids in node_ids.items():
        if node_type not in recon or not ids:
            continue
        error = ((recon[node_type] - data[node_type].x) ** 2).mean(dim=-1)  # per-node MSE, shape (n,)
        mean = error.mean()
        std = error.std(unbiased=False)
        std_safe = std if std > 1e-8 else torch.ones_like(std)  # constant-error population: z=0 for everyone
        z = (error - mean) / std_safe
        score = torch.sigmoid(z)

        for i, node_id in enumerate(ids):
            scores.append(
                AnomalyScore(
                    node_id=node_id,
                    node_type=node_type,
                    score=float(score[i]),
                    z_score=float(z[i]),
                    raw_error=float(error[i]),
                    population_mean_error=float(mean),
                    population_std_error=float(std),
                )
            )
    return scores


def anomaly_score_rows(scores: list[AnomalyScore]) -> list[dict]:
    rows = []
    for s in scores:
        reason_code = f"ANOMALY_ZSCORE={s.z_score:.2f}"
        evidence_json = json.dumps(
            {
                "z_score": s.z_score,
                "raw_reconstruction_error": s.raw_error,
                "node_type": s.node_type,
                "population_mean_error": s.population_mean_error,
                "population_std_error": s.population_std_error,
            }
        )
        rows.append({"node_id": s.node_id, "score": s.score, "reason_code": reason_code, "evidence_json": evidence_json})
    return rows
