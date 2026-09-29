"""argus.models.encoder: the shared temporal heterogeneous GAT-v2 encoder
(architecture doc sec 4.3). Tests the encoder/Time2Vec machinery itself
(shapes, no NaNs, temporal edge_attr wiring, gradient flow into Time2Vec's
learnable parameters) — anomaly-detection quality on real data is
argus.models.anomaly's concern, tested in test_models_anomaly.py.
"""
from __future__ import annotations

import datetime as dt

import igraph as ig
import pandas as pd
import torch

from argus.models.encoder import TemporalHeteroGATEncoder, Time2Vec, build_temporal_hetero_data


def _toy_graph_with_timestamps() -> ig.Graph:
    """Same shape as test_models_sage.py's toy graph, plus real
    BROADCAST_VIA timestamps (which that test never needed) — tx0/tx1
    broadcast an hour apart via ip0, tx2/tx3 an hour apart via ip1.
    """
    names = ["w0", "w1", "w2", "w3", "tx0", "tx1", "tx2", "tx3", "ip0", "ip1", "asn0", "asn1"]
    types = ["Wallet"] * 4 + ["Transaction"] * 4 + ["IP"] * 2 + ["ASN"] * 2
    g = ig.Graph(directed=True)
    g.add_vertices(len(names))
    g.vs["name"] = names
    g.vs["type"] = types

    def idx(name: str) -> int:
        return names.index(name)

    edges: list[tuple[int, int]] = []
    edge_types: list[str] = []
    timestamps: list[object] = []
    base = dt.datetime(2026, 1, 1)

    for i, (tx, funders, ip) in enumerate(
        [("tx0", ("w0", "w1"), "ip0"), ("tx1", ("w0", "w1"), "ip0"),
         ("tx2", ("w2", "w3"), "ip1"), ("tx3", ("w2", "w3"), "ip1")]
    ):
        for w in funders:
            edges.append((idx(w), idx(tx))); edge_types.append("FUNDS"); timestamps.append(None)
        edges.append((idx(tx), idx(funders[0]))); edge_types.append("PAYS"); timestamps.append(None)
        edges.append((idx(tx), idx(ip))); edge_types.append("BROADCAST_VIA"); timestamps.append(base + dt.timedelta(hours=i))

    edges.append((idx("ip0"), idx("asn0"))); edge_types.append("RESOLVES_TO"); timestamps.append(None)
    edges.append((idx("ip1"), idx("asn1"))); edge_types.append("RESOLVES_TO"); timestamps.append(None)

    g.add_edges(edges)
    g.es["type"] = edge_types
    g.es["timestamp"] = timestamps
    return g


def _toy_features(g: ig.Graph) -> pd.DataFrame:
    names = g.vs["name"]
    types = g.vs["type"]
    return pd.DataFrame(
        {
            "node_id": names,
            "node_type": types,
            "f_degree_total": [float(d) for d in g.degree()],
            "f_dummy": [0.1 * i for i in range(len(names))],
        }
    )


def test_build_temporal_hetero_data_attaches_matching_edge_attr_both_directions():
    g = _toy_graph_with_timestamps()
    features = _toy_features(g)
    data, node_ids = build_temporal_hetero_data(g, features)

    fwd = data["Transaction", "BROADCAST_VIA", "IP"]
    rev = data["IP", "rev_BROADCAST_VIA", "Transaction"]
    assert "edge_attr" in fwd and "edge_attr" in rev
    assert fwd.edge_attr.shape == (4, 1)
    # tx0..tx3 broadcast at hour 0,1,2,3 respectively; normalized to [0, 1]
    # over the full 3-hour span shared across all 4 BROADCAST_VIA edges.
    assert torch.allclose(fwd.edge_attr.flatten(), torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]), atol=1e-4)
    # ToUndirected preserves edge order 1:1 between forward and reverse.
    assert torch.equal(fwd.edge_attr, rev.edge_attr)


def test_time2vec_output_shape_and_learnable_params():
    t2v = Time2Vec(dim=6)
    t = torch.rand(10, 1)
    out = t2v(t)
    assert out.shape == (10, 6)
    assert not torch.isnan(out).any()
    params = list(t2v.parameters())
    assert len(params) == 2  # weight, bias
    assert all(p.requires_grad for p in params)


def test_encoder_forward_produces_valid_embeddings_for_every_node_type():
    g = _toy_graph_with_timestamps()
    features = _toy_features(g)
    data, _ = build_temporal_hetero_data(g, features)
    temporal = {et for et in data.edge_types if et[1] in ("BROADCAST_VIA", "rev_BROADCAST_VIA")}

    model = TemporalHeteroGATEncoder(data.edge_types, temporal, hidden_dim=8, out_dim=4, heads=2, time2vec_dim=6)
    out = model(data)

    assert set(out.keys()) == {"Wallet", "Transaction", "IP", "ASN"}
    for node_type, expected_n in (("Wallet", 4), ("Transaction", 4), ("IP", 2), ("ASN", 2)):
        assert out[node_type].shape == (expected_n, 4)
        assert not torch.isnan(out[node_type]).any()


def test_forward_with_attention_matches_forward_and_yields_valid_attention():
    """fusion/evidence.py's attention-based evidence extraction (Phase 4)
    depends on forward_with_attention reproducing forward()'s embeddings
    exactly (it must be reading the SAME computation, not an approximation)
    while additionally exposing per-edge attention weights for every
    relation at both layers.
    """
    g = _toy_graph_with_timestamps()
    features = _toy_features(g)
    data, _ = build_temporal_hetero_data(g, features)
    temporal = {et for et in data.edge_types if et[1] in ("BROADCAST_VIA", "rev_BROADCAST_VIA")}

    model = TemporalHeteroGATEncoder(data.edge_types, temporal, hidden_dim=8, out_dim=4, heads=2, time2vec_dim=6)
    model.eval()
    with torch.no_grad():
        plain_out = model(data)
        attn_out, attention = model.forward_with_attention(data)

    for node_type in plain_out:
        assert torch.allclose(plain_out[node_type], attn_out[node_type])

    assert set(attention.keys()) == {1, 2}
    for layer in (1, 2):
        assert set(attention[layer].keys()) == set(data.edge_types)
        for et, (edge_index, alpha) in attention[layer].items():
            assert edge_index.shape[1] == data[et].edge_index.shape[1]
            assert alpha.shape == (edge_index.shape[1],)  # averaged across heads to one scalar per edge
            assert not torch.isnan(alpha).any()


def test_gradients_flow_into_time2vec_parameters():
    """If Time2Vec's weight/bias never receive a gradient, the temporal
    signal is silently disconnected from training — this is the one thing
    that must never regress silently, since it wouldn't show up as a shape
    or NaN failure, just a permanently-untrained (effectively fixed-random)
    time encoding despite claiming to be learned.
    """
    g = _toy_graph_with_timestamps()
    features = _toy_features(g)
    data, _ = build_temporal_hetero_data(g, features)
    temporal = {et for et in data.edge_types if et[1] in ("BROADCAST_VIA", "rev_BROADCAST_VIA")}

    model = TemporalHeteroGATEncoder(data.edge_types, temporal, hidden_dim=8, out_dim=4, heads=2, time2vec_dim=6)
    out = model(data)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    assert model.time2vec.weight.grad is not None
    assert model.time2vec.weight.grad.abs().sum() > 0
