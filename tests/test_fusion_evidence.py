"""argus.fusion.evidence: attention-based evidence subgraph extraction, and
the encoder checkpoint save/load it depends on. Tests structural
correctness (hop-1 neighbors are real graph neighbors, hop-2 are neighbors
of neighbors, no duplicate node/relation pairs, checkpoint round-trips
exactly) rather than exact attention *values*, which are learned and not
independently hand-verifiable — same standard as this repo's other
ML-mechanism tests (e.g. test_models_encoder.py's own attention-shape
checks).
"""
from __future__ import annotations

import datetime as dt

import igraph as ig
import pandas as pd
import torch

from argus.fusion.evidence import (
    build_attention_context,
    extract_attention_evidence,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from argus.models.encoder import TemporalHeteroGATEncoder, build_temporal_hetero_data


def _toy_graph_with_timestamps() -> ig.Graph:
    """Same fixture shape as test_models_encoder.py's — two clearly-separate
    neighborhoods, {w0, w1} via tx0/tx1/ip0 and {w2, w3} via tx2/tx3/ip1."""
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


def _build_model():
    g = _toy_graph_with_timestamps()
    features = _toy_features(g)
    data, node_ids = build_temporal_hetero_data(g, features)
    temporal = {et for et in data.edge_types if et[1] in ("BROADCAST_VIA", "rev_BROADCAST_VIA")}
    model = TemporalHeteroGATEncoder(data.edge_types, temporal, hidden_dim=8, out_dim=4, heads=2, time2vec_dim=6)
    return model, data, node_ids, data.edge_types, temporal


def test_checkpoint_round_trip_reproduces_identical_embeddings(tmp_path):
    model, data, node_ids, edge_types, temporal = _build_model()
    model.eval()
    with torch.no_grad():
        before = model(data)

    ckpt_path = tmp_path / "encoder.pt"
    save_encoder_checkpoint(model, data, node_ids, edge_types, temporal, hidden_dim=8, embedding_dim=4, path=ckpt_path,
                             heads=2, time2vec_dim=6)
    loaded_encoder, loaded_data, loaded_node_ids = load_encoder_checkpoint(ckpt_path)

    with torch.no_grad():
        after = loaded_encoder(loaded_data)

    assert loaded_node_ids == node_ids
    for node_type in before:
        assert torch.allclose(before[node_type], after[node_type])


def test_hop1_neighbors_are_real_graph_neighbors_of_the_target(tmp_path):
    model, data, node_ids, edge_types, temporal = _build_model()
    ckpt_path = tmp_path / "encoder.pt"
    save_encoder_checkpoint(model, data, node_ids, edge_types, temporal, 8, 4, ckpt_path, heads=2, time2vec_dim=6)
    encoder, loaded_data, loaded_node_ids = load_encoder_checkpoint(ckpt_path)

    context = build_attention_context(encoder, loaded_data, loaded_node_ids)
    evidence = extract_attention_evidence(context, "w0", "Wallet", top_n=5)

    assert evidence.target_node_id == "w0"
    hop1_ids = {n.node_id for n in evidence.neighbors if n.hop == 1}
    # w0's real graph neighborhood is {tx0, tx1} (the transactions it funds/
    # is paid by) — never w2/w3's disjoint neighborhood {tx2, tx3}.
    assert hop1_ids <= {"tx0", "tx1"}
    assert hop1_ids  # at least one real neighbor found
    assert hop1_ids.isdisjoint({"tx2", "tx3"})


def test_hop2_neighbors_connect_via_an_actual_hop1_neighbor(tmp_path):
    model, data, node_ids, edge_types, temporal = _build_model()
    ckpt_path = tmp_path / "encoder.pt"
    save_encoder_checkpoint(model, data, node_ids, edge_types, temporal, 8, 4, ckpt_path, heads=2, time2vec_dim=6)
    encoder, loaded_data, loaded_node_ids = load_encoder_checkpoint(ckpt_path)

    context = build_attention_context(encoder, loaded_data, loaded_node_ids)
    evidence = extract_attention_evidence(context, "w0", "Wallet", top_n=5)
    hop1_ids = {n.node_id for n in evidence.neighbors if n.hop == 1}
    hop2 = [n for n in evidence.neighbors if n.hop == 2]

    assert hop2  # w0's transactions have their own neighbors (w1, ip0, ...)
    for n in hop2:
        assert n.connects_to in hop1_ids  # every hop-2 edge lands on a genuine hop-1 node


def test_no_duplicate_node_id_per_hop_per_target():
    """A node reachable via more than one relation (e.g. Wallet->Transaction
    via both FUNDS and ToUndirected's synthetic rev_PAYS) must occupy only
    ONE slot, not silently double-count toward top_n or get hop-2-expanded
    twice."""
    model, data, node_ids, edge_types, temporal = _build_model()
    context = build_attention_context(model, data, node_ids)
    evidence = extract_attention_evidence(context, "w0", "Wallet", top_n=5)

    hop1_ids = [n.node_id for n in evidence.neighbors if n.hop == 1]
    assert len(hop1_ids) == len(set(hop1_ids))

    hop2_by_parent: dict[str, list[str]] = {}
    for n in evidence.neighbors:
        if n.hop == 2:
            hop2_by_parent.setdefault(n.connects_to, []).append(n.node_id)
    for parent, ids in hop2_by_parent.items():
        assert len(ids) == len(set(ids)), f"duplicate hop-2 node under {parent}"


def test_unknown_target_returns_empty_evidence():
    model, data, node_ids, edge_types, temporal = _build_model()
    context = build_attention_context(model, data, node_ids)
    evidence = extract_attention_evidence(context, "does_not_exist", "Wallet", top_n=5)
    assert evidence.neighbors == []
    assert evidence.nodes == ["does_not_exist"]
    assert evidence.edges == []


def test_top_n_limits_hop1_neighbor_count():
    model, data, node_ids, edge_types, temporal = _build_model()
    context = build_attention_context(model, data, node_ids)
    evidence = extract_attention_evidence(context, "w0", "Wallet", top_n=1)
    assert len([n for n in evidence.neighbors if n.hop == 1]) <= 1


def test_one_context_serves_multiple_targets_without_recomputation():
    """The whole point of separating build_attention_context from
    extract_attention_evidence (see fusion/evidence.py's PERFORMANCE note):
    one forward pass must serve every alert, not just one target node.
    """
    model, data, node_ids, edge_types, temporal = _build_model()
    context = build_attention_context(model, data, node_ids)

    ev_w0 = extract_attention_evidence(context, "w0", "Wallet", top_n=5)
    ev_w2 = extract_attention_evidence(context, "w2", "Wallet", top_n=5)

    hop1_w0 = {n.node_id for n in ev_w0.neighbors if n.hop == 1}
    hop1_w2 = {n.node_id for n in ev_w2.neighbors if n.hop == 1}
    assert hop1_w0 and hop1_w2
    assert hop1_w0.isdisjoint(hop1_w2)  # w0/w2 are in disjoint graph neighborhoods
