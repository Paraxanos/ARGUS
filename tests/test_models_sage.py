"""argus.models.sage: the heterogeneous GraphSAGE encoder that produces ER
pass 2's wallet embeddings. Tests the training/embedding machinery itself
(shapes, no NaNs, determinism-adjacent stability, meaningful relative
separation) — not entity-resolution quality on real data, which is
argus.er.embed_cluster's concern (and its own module docstring's MEASURED
RESULT section for why that's currently weak on the full dataset).
"""
from __future__ import annotations

import igraph as ig
import pandas as pd

from argus.models.sage import build_hetero_data, train_wallet_embeddings, wallet_embeddings_frame


def _toy_graph() -> ig.Graph:
    """Two clearly-separate transaction "neighborhoods": {w0, w1} both fund
    tx0/tx1, which both broadcast via ip0 -> asn0; {w2, w3} both fund
    tx2/tx3, which both broadcast via ip1 -> asn1. w0/w1 should end up
    embedded closer to each other than to w2/w3, and vice versa — the
    minimal case that exercises message passing through every edge type in
    docs/contracts.md (FUNDS, PAYS, BROADCAST_VIA, RESOLVES_TO).
    """
    names = ["w0", "w1", "w2", "w3", "tx0", "tx1", "tx2", "tx3", "ip0", "ip1", "asn0", "asn1"]
    types = ["Wallet"] * 4 + ["Transaction"] * 4 + ["IP"] * 2 + ["ASN"] * 2
    g = ig.Graph(directed=True)
    g.add_vertices(len(names))
    g.vs["name"] = names
    g.vs["type"] = types

    edges: list[tuple[int, int]] = []
    edge_types: list[str] = []

    def idx(name: str) -> int:
        return names.index(name)

    for tx, funders, ip in (("tx0", ("w0", "w1"), "ip0"), ("tx1", ("w0", "w1"), "ip0"),
                             ("tx2", ("w2", "w3"), "ip1"), ("tx3", ("w2", "w3"), "ip1")):
        for w in funders:
            edges.append((idx(w), idx(tx)))
            edge_types.append("FUNDS")
        edges.append((idx(tx), idx(funders[0])))
        edge_types.append("PAYS")
        edges.append((idx(tx), idx(ip)))
        edge_types.append("BROADCAST_VIA")

    edges.append((idx("ip0"), idx("asn0")))
    edge_types.append("RESOLVES_TO")
    edges.append((idx("ip1"), idx("asn1")))
    edge_types.append("RESOLVES_TO")

    g.add_edges(edges)
    g.es["type"] = edge_types
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


def test_build_hetero_data_shapes():
    g = _toy_graph()
    features = _toy_features(g)
    data, node_ids = build_hetero_data(g, features)

    assert set(node_ids["Wallet"]) == {"w0", "w1", "w2", "w3"}
    assert data["Wallet"].x.shape == (4, 2)
    assert ("Wallet", "FUNDS", "Transaction") in data.edge_types
    # ToUndirected() symmetrizes for message passing; the original directed
    # relations must still be present unchanged for training supervision.
    assert data["Wallet", "FUNDS", "Transaction"].edge_index.shape[1] == 8  # 2 funders x 4 txs


def test_training_produces_valid_wallet_embeddings():
    g = _toy_graph()
    features = _toy_features(g)
    result = train_wallet_embeddings(g, features, hidden_dim=8, embedding_dim=4, max_epochs=20, seed=0)
    frame = wallet_embeddings_frame(result)

    assert list(frame["node_id"]) == ["w0", "w1", "w2", "w3"]
    emb_cols = [c for c in frame.columns if c.startswith("emb_")]
    assert len(emb_cols) == 4
    assert not frame[emb_cols].isna().any().any()
    # Loss should have moved (not asserting a specific value — just that
    # training actually ran, not a no-op).
    assert result.losses[0] != result.losses[-1]


def test_same_neighborhood_wallets_embed_closer_than_different_neighborhoods():
    g = _toy_graph()
    features = _toy_features(g)
    result = train_wallet_embeddings(g, features, hidden_dim=8, embedding_dim=4, max_epochs=40, seed=0)
    frame = wallet_embeddings_frame(result).set_index("node_id")
    emb_cols = [c for c in frame.columns if c.startswith("emb_")]

    import numpy as np

    def dist(a: str, b: str) -> float:
        return float(np.linalg.norm(frame.loc[a, emb_cols].to_numpy() - frame.loc[b, emb_cols].to_numpy()))

    same_neighborhood = dist("w0", "w1")
    cross_neighborhood = (dist("w0", "w2") + dist("w0", "w3") + dist("w1", "w2") + dist("w1", "w3")) / 4
    assert same_neighborhood < cross_neighborhood


def test_edgeless_graph_returns_contract_valid_output_without_crashing():
    """Nodes but zero edges of any type — cannot happen in the real pipeline
    (argus.graph.build only ever creates a Wallet vertex via a FUNDS/PAYS
    edge in the first place) but must not crash if it ever did: with no
    relation for HeteroConv to route through, there is structurally no way
    to produce a per-node output, so wallet_embeddings_frame's own "no
    embeddings" fallback (an empty, contract-shaped frame) is the correct,
    documented result here — not a NaN-filled or partial one.
    """
    g = ig.Graph(directed=True)
    g.add_vertices(2)
    g.vs["name"] = ["w0", "w1"]
    g.vs["type"] = ["Wallet", "Wallet"]
    g.es["type"] = []  # declares the (empty) edge attribute set, matching a real graph's schema
    features = pd.DataFrame({"node_id": ["w0", "w1"], "node_type": ["Wallet", "Wallet"], "f_dummy": [0.0, 1.0]})

    result = train_wallet_embeddings(g, features, max_epochs=1)  # must not raise
    frame = wallet_embeddings_frame(result)
    assert list(frame.columns) == ["node_id"]
    assert frame.empty
