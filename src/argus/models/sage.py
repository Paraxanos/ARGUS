"""Heterogeneous GraphSAGE embeddings for wallets — ER pass 2's input.

Architecture doc sec 4.2 (pass 2): wallets are embedded jointly with their
surrounding graph context — including their IP/ASN broadcast context via
BROADCAST_VIA/RESOLVES_TO edges — so that two wallets which never co-spend
but consistently share network infrastructure end up with similar
embeddings. This is what lets ER pass 2 (argus.er.embed_cluster) catch
entities pass 1's structural heuristics cannot see.

TRAINING OBJECTIVE: self-supervised link prediction via negative sampling —
for each real edge (u, v), the model is trained to score it higher than a
random (u, v') pair of the same node types. This is a standard, simplified
substitute for GraphSAGE's original random-walk co-occurrence objective
(Hamilton et al. 2017): same "no labels needed" property, far less code.
ponytail: deliberate simplification — a proper random-walk or triplet-margin
objective is the upgrade path if these embeddings prove insufficiently
discriminative for pass 2's clustering.

The model is heterogeneous per docs/contracts.md's node/edge vocabulary
(Wallet/Transaction/IP/ASN; FUNDS/PAYS/BROADCAST_VIA/RESOLVES_TO/CO_SPEND) —
each relation gets its own SAGEConv weights via PyG's HeteroConv.
ToUndirected() adds reverse edges for message passing only (never used as
extra training supervision — that would double-count a single relation).

An edge type absent from the current graph (currently: CO_SPEND, which ER
pass 1 produces zero of on this dataset — see docs/WRITEUP.md's Phase 2 ER
diagnosis) is simply absent from HeteroData's relation set; nothing here
needs to change once pass 1 (or a re-run after pass 2 adds SAME_ENTITY)
produces real same-wallet-pair edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import igraph as ig
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected

HIDDEN_DIM = 32
EMBEDDING_DIM = 16
MAX_EPOCHS = 30
LEARNING_RATE = 0.01
# Subsamples dense relations so a full-batch epoch stays cheap at ~200k-tx
# scale (PAYS/FUNDS can have hundreds of thousands of edges) — every relation
# still contributes every epoch, just via a fresh random subset each time.
MAX_EDGES_PER_RELATION_PER_EPOCH = 4096


@dataclass
class TrainingResult:
    embeddings: dict[str, np.ndarray]  # node_type -> (n, embedding_dim)
    node_ids: dict[str, list[str]]  # node_type -> ordered ids matching embeddings' rows
    losses: list[float] = field(default_factory=list)


class HeteroSAGE(nn.Module):
    def __init__(self, edge_types: list[tuple[str, str, str]], hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), out_dim) for et in edge_types}, aggr="sum")

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return self.conv2(x_dict, edge_index_dict)


def build_hetero_data(g: ig.Graph, node_features: pd.DataFrame) -> tuple[HeteroData, dict[str, list[str]]]:
    """Builds a PyG HeteroData from the typed igraph + node_features.parquet's
    f_* columns, reused directly as input features (already one row per
    node, uniform schema across types, 0-filled where inapplicable — see
    argus.features.build's fill-strategy docstring).

    Returns the HeteroData plus node_type -> ordered node_id list, needed to
    map embedding rows back to wallet_id/txid/ip/asn strings afterward.
    """
    f_cols = [c for c in node_features.columns if c.startswith("f_")]
    node_ids: dict[str, list[str]] = {}
    local_index: dict[str, dict[str, int]] = {}

    data = HeteroData()
    for node_type, group in node_features.groupby("node_type"):
        ids = group["node_id"].tolist()
        node_ids[node_type] = ids
        local_index[node_type] = {nid: i for i, nid in enumerate(ids)}
        x = torch.tensor(group[f_cols].to_numpy(dtype=np.float32))
        mean = x.mean(dim=0, keepdim=True)
        # unbiased=False: population std, defined (0.0) even for a single-row
        # group (e.g. a run with exactly one ASN) rather than NaN.
        std = x.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))  # avoid /0 on constant columns
        data[node_type].x = (x - mean) / std

    type_by_name: dict[str, str] = {nid: node_type for node_type, ids in node_ids.items() for nid in ids}

    edgelist = g.get_edgelist()
    names = g.vs["name"]
    edge_types = g.es["type"]

    by_relation: dict[str, list[tuple[str, str]]] = {}
    for (s, t), et in zip(edgelist, edge_types):
        by_relation.setdefault(et, []).append((names[s], names[t]))

    for et, pairs in by_relation.items():
        if not pairs:
            continue
        src_type = type_by_name[pairs[0][0]]
        dst_type = type_by_name[pairs[0][1]]
        src_idx = [local_index[src_type][s] for s, _ in pairs]
        dst_idx = [local_index[dst_type][t] for _, t in pairs]
        data[src_type, et, dst_type].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    return ToUndirected()(data), node_ids


def _sample_negative(num_nodes: int, batch_size: int, exclude: torch.Tensor) -> torch.Tensor:
    # Approximate negative sampling — no full exclusion-set filtering beyond
    # nudging away an accidental collision with the true positive. Standard
    # and cheap; adequate at this node count, where a random collision with
    # the true positive is rare.
    neg = torch.randint(0, num_nodes, (batch_size,), device=exclude.device)
    collision = neg == exclude
    if collision.any():
        neg = torch.where(collision, (neg + 1) % num_nodes, neg)
    return neg


def train_wallet_embeddings(
    g: ig.Graph,
    node_features: pd.DataFrame,
    hidden_dim: int = HIDDEN_DIM,
    embedding_dim: int = EMBEDDING_DIM,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LEARNING_RATE,
    seed: int = 0,
) -> TrainingResult:
    """Trains the heterogeneous GraphSAGE encoder via self-supervised link
    prediction (see module docstring) and returns embeddings for every node
    type — callers needing only wallets should read
    ``result.embeddings["Wallet"]`` (or use ``wallet_embeddings_frame``).
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, node_ids = build_hetero_data(g, node_features)
    data = data.to(device)
    # Built manually rather than via data.edge_index_dict: that property
    # raises KeyError when the HeteroData has zero edge-store entries of any
    # kind (a graph with nodes but literally no edges yet) instead of
    # returning {} — verified directly. dict comprehension over
    # data.edge_types (which IS always a safe, possibly-empty list) sidesteps
    # that entirely.
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    # Supervise on the ORIGINAL relations only — ToUndirected()'s added
    # "rev_*" relations (or symmetrized same-type relations) exist purely for
    # message passing, not as extra supervision.
    original_relations = [et for et in data.edge_types if not et[1].startswith("rev_")]

    model = HeteroSAGE(data.edge_types, hidden_dim, embedding_dim).to(device)
    result = TrainingResult(embeddings={}, node_ids=node_ids)

    trainable_relations = [et for et in original_relations if data[et].edge_index.size(1) > 0]
    if not trainable_relations:
        # No edges anywhere (degenerate/empty graph) — nothing to train on;
        # still return a contract-valid, correctly-shaped (untrained) output
        # rather than crashing.
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, edge_index_dict)
        result.embeddings = {k: v.cpu().numpy() for k, v in out.items()}
        return result

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, edge_index_dict)

        total_loss = torch.zeros((), device=device)
        for src_type, rel, dst_type in trainable_relations:
            edge_index = data[src_type, rel, dst_type].edge_index
            num_edges = edge_index.size(1)
            if num_edges > MAX_EDGES_PER_RELATION_PER_EPOCH:
                perm = torch.randperm(num_edges, device=device)[:MAX_EDGES_PER_RELATION_PER_EPOCH]
                edge_index = edge_index[:, perm]
                num_edges = edge_index.size(1)

            src, dst = edge_index[0], edge_index[1]
            neg_dst = _sample_negative(out[dst_type].size(0), num_edges, dst)

            pos_score = (out[src_type][src] * out[dst_type][dst]).sum(dim=-1)
            neg_score = (out[src_type][src] * out[dst_type][neg_dst]).sum(dim=-1)

            labels = torch.cat([torch.ones(num_edges, device=device), torch.zeros(num_edges, device=device)])
            scores = torch.cat([pos_score, neg_score])
            total_loss = total_loss + F.binary_cross_entropy_with_logits(scores, labels)

        total_loss.backward()
        optimizer.step()
        result.losses.append(float(total_loss.item()))

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, edge_index_dict)
    result.embeddings = {k: v.cpu().numpy() for k, v in out.items()}
    return result


def wallet_embeddings_frame(result: TrainingResult) -> pd.DataFrame:
    """Wallet-only embeddings as a flat DataFrame: node_id, emb_0..emb_{d-1}."""
    ids = result.node_ids.get("Wallet", [])
    emb = result.embeddings.get("Wallet")
    if emb is None or len(ids) == 0:
        return pd.DataFrame(columns=["node_id"])
    cols = {f"emb_{i}": emb[:, i] for i in range(emb.shape[1])}
    return pd.DataFrame({"node_id": ids, **cols})
