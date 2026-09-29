"""Shared temporal heterogeneous GAT-v2 encoder — architecture doc sec 4.3's
"Multi-Task Detection Core". Distinct from models/sage.py's GraphSAGE encoder
(sec 4.2, ER pass 2 only): this one is designed to be SHARED across the
anomaly/pattern/risk detection heads, so they reason over one learned
representation instead of three disconnected models (models/anomaly.py is
its first consumer, Phase 2 — see that module).

Heterogeneous: one GATv2Conv per relation via PyG's HeteroConv, same pattern
as models/sage.py. Temporal: BROADCAST_VIA edges (the only relation carrying
a real timestamp per docs/contracts.md) get a Time2Vec-encoded edge feature
(Kazemi & Poole 2019 — one learnable linear term + learnable-frequency
periodic terms) that GATv2Conv's attention conditions on via edge_dim;
Time2Vec's own parameters are learned end-to-end with the rest of the model,
not precomputed — that learnable-frequency property is what distinguishes it
from a fixed sinusoidal positional encoding. Relations without a timestamp
get a plain GATv2Conv (no edge_dim).
"""
from __future__ import annotations

import igraph as ig
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

from argus.models.sage import build_hetero_data

HIDDEN_DIM = 32
EMBEDDING_DIM = 16
HEADS = 2
TIME2VEC_DIM = 8

TEMPORAL_RELATION = "BROADCAST_VIA"


class Time2Vec(nn.Module):
    """t -> [w0*t + b0, sin(w1*t + b1), ..., sin(w_{k-1}*t + b_{k-1})].
    w, b are learned, not fixed — the property that distinguishes this from
    a plain sinusoidal positional encoding (Kazemi & Poole, 2019).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t * self.weight + self.bias  # (E, 1) * (dim,) broadcasts to (E, dim)
        return torch.cat([angles[:, :1], torch.sin(angles[:, 1:])], dim=-1)


def build_temporal_hetero_data(g: ig.Graph, node_features: pd.DataFrame) -> tuple[HeteroData, dict[str, list[str]]]:
    """Reuses argus.models.sage.build_hetero_data for the base structure,
    then attaches a raw (not yet Time2Vec-encoded — that happens inside the
    encoder's forward pass, see module docstring) normalized-time edge_attr
    to the BROADCAST_VIA relation.

    build_hetero_data already applies ToUndirected() internally, so the
    reverse relation (rev_BROADCAST_VIA) exists by the time this function
    sees `data` but was never given an edge_attr of its own — ToUndirected
    only propagates edge_attr set BEFORE it runs. Fixed here by attaching
    the identical tensor to both relations directly: verified empirically
    that ToUndirected's reverse relation preserves the forward relation's
    edge order exactly (column i's (src, dst) becomes column i's (dst,
    src)), so the same per-edge time value is valid for both.
    """
    data, node_ids, edge_order = build_hetero_data(g, node_features)

    broadcast_key = next((et for et in data.edge_types if et[1] == TEMPORAL_RELATION), None)
    pairs = edge_order.get(TEMPORAL_RELATION, [])
    if broadcast_key is None or not pairs:
        return data, node_ids  # no timestamped edges in this graph -- nothing to attach

    tx_to_ts: dict[str, float] = {
        g.vs[e.source]["name"]: pd.Timestamp(e["timestamp"]).timestamp() for e in g.es if e["type"] == TEMPORAL_RELATION
    }
    raw_ts = torch.tensor([tx_to_ts[tx] for tx, _ in pairs], dtype=torch.float32)
    span = (raw_ts.max() - raw_ts.min()).clamp(min=1.0)  # avoid /0 when every timestamp is identical
    t_norm = ((raw_ts - raw_ts.min()) / span).view(-1, 1)

    data[broadcast_key].edge_attr = t_norm
    reverse_key = (broadcast_key[2], f"rev_{broadcast_key[1]}", broadcast_key[0])
    if reverse_key in data.edge_types:
        data[reverse_key].edge_attr = t_norm
    return data, node_ids


class TemporalHeteroGATEncoder(nn.Module):
    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        temporal_edge_types: set[tuple[str, str, str]],
        hidden_dim: int = HIDDEN_DIM,
        out_dim: int = EMBEDDING_DIM,
        heads: int = HEADS,
        time2vec_dim: int = TIME2VEC_DIM,
    ) -> None:
        super().__init__()
        self.temporal_edge_types = temporal_edge_types
        self.time2vec = Time2Vec(time2vec_dim) if temporal_edge_types else None

        def make_conv(out_channels: int, concat: bool) -> HeteroConv:
            convs = {}
            for et in edge_types:
                kwargs = dict(heads=heads, concat=concat, add_self_loops=False)
                if et in temporal_edge_types:
                    kwargs["edge_dim"] = time2vec_dim
                convs[et] = GATv2Conv((-1, -1), out_channels, **kwargs)
            return HeteroConv(convs, aggr="sum")

        # concat=True on the hidden layer (standard multi-head GAT practice —
        # each head sees a different attention pattern, concatenation keeps
        # all of them); concat=False on the output layer so out_dim is
        # exactly EMBEDDING_DIM regardless of head count, not heads*out_dim.
        self.conv1 = make_conv(hidden_dim, concat=True)
        self.conv2 = make_conv(out_dim, concat=False)

    def _edge_attr_dict(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        if self.time2vec is None:
            return {}
        return {et: self.time2vec(data[et].edge_attr) for et in self.temporal_edge_types if "edge_attr" in data[et]}

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        # Built manually rather than via data.edge_index_dict: that property
        # raises KeyError when the HeteroData has zero edge-store entries of
        # any kind, instead of returning {} — see argus.models.sage's
        # train_wallet_embeddings for the same fix, verified there directly.
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        edge_attr_dict = self._edge_attr_dict(data)
        x_dict = self.conv1(data.x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        return self.conv2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

    def forward_with_attention(
        self, data: HeteroData
    ) -> tuple[dict[str, torch.Tensor], dict[int, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]]]:
        """Same computation as forward(), plus per-layer, per-relation
        attention weights (fusion/evidence.py's "top attention-weighted
        edges" per architecture doc sec 4.4) — a second, side pass calling
        each relation's underlying GATv2Conv directly with
        return_attention_weights=True, rather than threading that flag
        through HeteroConv (which does not expose it). Verified empirically
        to reproduce HeteroConv's own per-relation output bit-for-bit when
        called with the same (x_src, x_dst) bipartite input it uses
        internally — see the Phase 4 commit message for that check. This
        doubles the encoder's forward-pass cost, but only at evidence-
        extraction time (a handful of calls for flagged nodes), never during
        training.

        Returns (embeddings, {1: {relation: (edge_index, attention)}, 2: {...}})
        — attention is averaged across heads to one scalar weight per edge.
        """
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        edge_attr_dict = self._edge_attr_dict(data)

        def layer_attention(hetero_conv: HeteroConv, x_dict: dict[str, torch.Tensor]) -> dict:
            attn = {}
            for et, conv in hetero_conv.convs.items():
                src_type, _, dst_type = et
                edge_attr = edge_attr_dict.get(et)
                kwargs = {"edge_attr": edge_attr} if edge_attr is not None else {}
                _, (edge_index, alpha) = conv(
                    (x_dict[src_type], x_dict[dst_type]), edge_index_dict[et], return_attention_weights=True, **kwargs
                )
                attn[et] = (edge_index, alpha.mean(dim=-1))
            return attn

        attn_layer1 = layer_attention(self.conv1, data.x_dict)
        x_dict = self.conv1(data.x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        attn_layer2 = layer_attention(self.conv2, x_dict)
        out_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        return out_dict, {1: attn_layer1, 2: attn_layer2}
