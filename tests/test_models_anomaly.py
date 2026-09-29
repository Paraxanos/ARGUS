"""argus.models.anomaly: the graph autoencoder anomaly head. Tests that
scoring is actually meaningful (a planted outlier scores higher than a
uniform "normal" population), not just contract-shaped and non-crashing —
matching this repo's standard of measuring against a known-correct answer
before trusting a model's numbers (see e.g. the peeling-detector debugging
story in docs/WRITEUP.md).
"""
from __future__ import annotations

import datetime as dt
import random

import igraph as ig
import pandas as pd

from argus.models.anomaly import anomaly_score_rows, train_and_score_anomalies


def _graph_with_one_outlier_wallet(n_normal: int = 20) -> tuple[ig.Graph, pd.DataFrame]:
    """n_normal ordinary wallets (near-identical low-magnitude features) each
    funding one transaction via a shared IP, plus one "w_outlier" wallet
    with wildly different-scale features but an otherwise identical local
    graph structure — isolates the anomaly signal to the FEATURES, not the
    topology, so a real detection here can't be a topology artifact.
    """
    rng = random.Random(0)
    names, types = [], []
    for i in range(n_normal):
        names.append(f"w{i}"); types.append("Wallet")
    names.append("w_outlier"); types.append("Wallet")
    for i in range(n_normal + 1):
        names.append(f"tx{i}"); types.append("Transaction")
    names += ["ip0", "asn0"]
    types += ["IP", "ASN"]

    g = ig.Graph(directed=True)
    g.add_vertices(len(names))
    g.vs["name"] = names
    g.vs["type"] = types

    def idx(n: str) -> int:
        return names.index(n)

    edges, edge_types, timestamps = [], [], []
    base = dt.datetime(2026, 1, 1)
    for i in range(n_normal + 1):
        w = "w_outlier" if i == n_normal else f"w{i}"
        tx = f"tx{i}"
        edges.append((idx(w), idx(tx))); edge_types.append("FUNDS"); timestamps.append(None)
        edges.append((idx(tx), idx(w))); edge_types.append("PAYS"); timestamps.append(None)
        edges.append((idx(tx), idx("ip0"))); edge_types.append("BROADCAST_VIA")
        timestamps.append(base + dt.timedelta(minutes=i))
    edges.append((idx("ip0"), idx("asn0"))); edge_types.append("RESOLVES_TO"); timestamps.append(None)

    g.add_edges(edges)
    g.es["type"] = edge_types
    g.es["timestamp"] = timestamps

    rows = []
    for name, t in zip(names, types):
        if name == "w_outlier":
            rows.append({"node_id": name, "node_type": t, "f_a": 500.0, "f_b": -300.0})
        else:
            rows.append({"node_id": name, "node_type": t, "f_a": rng.uniform(0.9, 1.1), "f_b": rng.uniform(-0.1, 0.1)})
    return g, pd.DataFrame(rows)


def test_planted_outlier_scores_highest_among_wallets():
    n_normal = 20
    g, node_features = _graph_with_one_outlier_wallet(n_normal)

    result = train_and_score_anomalies(g, node_features, hidden_dim=8, embedding_dim=4, max_epochs=60, seed=0)
    by_id = {s.node_id: s for s in result.scores}

    wallet_scores = [s for s in result.scores if s.node_type == "Wallet"]
    assert by_id["w_outlier"].score == max(s.score for s in wallet_scores)
    normal_z = [by_id[f"w{i}"].z_score for i in range(n_normal)]
    assert by_id["w_outlier"].z_score > max(normal_z)


def test_scores_are_contract_shaped_and_bounded():
    g, node_features = _graph_with_one_outlier_wallet(n_normal=10)
    result = train_and_score_anomalies(g, node_features, hidden_dim=8, embedding_dim=4, max_epochs=5, seed=0)

    assert result.scores  # every node type got scored
    assert all(0.0 <= s.score <= 1.0 for s in result.scores)
    node_ids = {s.node_id for s in result.scores}
    assert node_ids == set(g.vs["name"])  # full population coverage, no node skipped

    rows = anomaly_score_rows(result.scores)
    df = pd.DataFrame(rows)
    assert list(df.columns) == ["node_id", "score", "reason_code", "evidence_json"]
    assert df["reason_code"].str.startswith("ANOMALY_ZSCORE=").all()
    assert not df.isna().any().any()


def test_result_also_returns_matching_embeddings_for_pattern_sim_reuse():
    g, node_features = _graph_with_one_outlier_wallet(n_normal=10)
    result = train_and_score_anomalies(g, node_features, hidden_dim=8, embedding_dim=4, max_epochs=5, seed=0)

    assert set(result.embeddings) == set(result.node_ids)
    for node_type, ids in result.node_ids.items():
        assert result.embeddings[node_type].shape == (len(ids), 4)


def test_empty_graph_returns_no_scores_without_crashing():
    g = ig.Graph(directed=True)
    g.add_vertices(2)
    g.vs["name"] = ["w0", "w1"]
    g.vs["type"] = ["Wallet", "Wallet"]
    g.es["type"] = []
    features = pd.DataFrame({"node_id": ["w0", "w1"], "node_type": ["Wallet", "Wallet"], "f_dummy": [0.0, 1.0]})

    result = train_and_score_anomalies(g, features, max_epochs=1)
    assert result.scores == []
