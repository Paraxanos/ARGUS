"""ER pass 2 (argus.er.embed_cluster): HDBSCAN + reconciliation logic,
tested against a hand-built embedding fixture where merge, split, and noise
all have a known-correct answer — see argus.models.sage's own test for the
GraphSAGE training/embedding half, and this module's MEASURED RESULT
docstring for why the real 200k-tx dataset does not (currently) exercise a
meaningful merge/split outcome (a diagnosed graph-topology limitation, not a
correctness bug in the algorithm this file tests).
"""
from __future__ import annotations

import threading

import igraph as ig
import pandas as pd
import pytest

from argus.er.embed_cluster import (
    BIG_STACK_BYTES,
    drop_high_fanin_broadcasters,
    reconcile_with_pass1,
    run_hdbscan,
    run_with_big_stack,
)


def _pass1_entities(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"wallet_id": w, "entity_id": e, "source": "pass1", "conf": 1.0, "merge_split_log_ref": None} for w, e in pairs]
    )


def test_merge_and_split_and_noise_all_correct():
    """Three tight clusters (3 points each, well separated) plus one isolated
    point:
      - cluster A: w1, w2, w1b — three DIFFERENT pass-1 singleton entities
        (E1, E2, E1b) -> HDBSCAN should MERGE all three into one entity.
      - cluster B / C: pass-1 entity E3 already groups {w3, w4, w6} (as if a
        future/improved pass 1 had merged them), but w3/w4 embed near a
        wallet from a different pass-1 entity (E3b) while w6 embeds near
        wallets from two other entities (E6b, E6c) — HDBSCAN should SPLIT E3
        into {w3, w4} and {w6}, each then merged into its own cross-entity
        group.
      - w5: alone, far from everything — HDBSCAN must call it noise, and
        pass 2 must leave it exactly as pass 1 had it (source stays "pass1").
    """
    pass1 = _pass1_entities(
        [
            ("w1", "E1"), ("w2", "E2"), ("w1b", "E1b"),
            ("w3", "E3"), ("w4", "E3"), ("w6", "E3"),
            ("w3b", "E3b"), ("w6b", "E6b"), ("w6c", "E6c"),
            ("w5", "E5"),
        ]
    )
    emb = {
        "w1": (0.0, 0.0), "w2": (0.1, 0.1), "w1b": (0.05, 0.05),
        "w3": (10.0, 10.0), "w4": (10.1, 10.1), "w3b": (10.05, 10.05),
        "w6": (20.0, 20.0), "w6b": (20.1, 20.1), "w6c": (20.05, 20.05),
        "w5": (100.0, 100.0),
    }
    wallet_embeddings = pd.DataFrame([{"node_id": w, "emb_0": xy[0], "emb_1": xy[1]} for w, xy in emb.items()])

    hdbscan_df = run_hdbscan(wallet_embeddings, min_cluster_size=3)
    result = reconcile_with_pass1(pass1, hdbscan_df)

    by_wallet = result.entities.set_index("wallet_id")

    assert by_wallet.loc["w1", "entity_id"] == by_wallet.loc["w2", "entity_id"] == by_wallet.loc["w1b", "entity_id"]
    assert by_wallet.loc["w3", "entity_id"] == by_wallet.loc["w4", "entity_id"] == by_wallet.loc["w3b", "entity_id"]
    assert by_wallet.loc["w6", "entity_id"] == by_wallet.loc["w6b", "entity_id"] == by_wallet.loc["w6c", "entity_id"]
    assert by_wallet.loc["w3", "entity_id"] != by_wallet.loc["w6", "entity_id"], "E3 should have split"

    assert by_wallet.loc["w5", "entity_id"] == "E5"
    assert by_wallet.loc["w5", "source"] == "pass1"

    for w in ("w1", "w2", "w1b", "w3", "w4", "w3b", "w6", "w6b", "w6c"):
        assert by_wallet.loc[w, "source"] == "pass2"
        assert by_wallet.loc[w, "merge_split_log_ref"] is not None

    assert (result.log["action"] == "merge").sum() == 3
    assert (result.log["action"] == "split").sum() == 1

    linked_pairs = {frozenset((a, b)) for a, b, _ in result.same_entity_links}
    assert frozenset(("w1", "w2")) in linked_pairs
    assert frozenset(("w3", "w6")) not in linked_pairs  # never same entity, must never be linked


def test_noise_only_when_too_few_points_for_min_cluster_size():
    wallet_embeddings = pd.DataFrame(
        [{"node_id": f"w{i}", "emb_0": float(i), "emb_1": float(i)} for i in range(2)]
    )
    hdbscan_df = run_hdbscan(wallet_embeddings, min_cluster_size=3)
    assert hdbscan_df.empty

    pass1 = _pass1_entities([("w0", "E0"), ("w1", "E1")])
    result = reconcile_with_pass1(pass1, hdbscan_df)
    assert set(result.entities["source"]) == {"pass1"}
    assert result.same_entity_links == []


def test_protect_multiwallet_pass1_entities_never_splits():
    """Real-data finding (ARGUS dataset Track M benchmarking report,
    2026-09-30): pass 1 measured precision 0.960 there (unlike this repo's
    own vacuous 0.0-recall synthetic result), so its multi-wallet co-spend
    clusters are usually correct — pass 2 splitting them anyway dropped
    overall ER precision to 0.209. protect_multiwallet_pass1_entities=True
    must stop the split without needing that (hidden) dataset: this fixture
    reproduces the same split mechanism the test above exercises (a
    multi-wallet pass-1 entity where one wallet lands in a real HDBSCAN
    cluster and another is noise) and confirms protection keeps it whole.
    """
    pass1 = pd.concat(
        [
            _pass1_entities([("wa", "E"), ("wb", "E"), ("wc", "E")]),
            _pass1_entities([("px", "Ex")]),
            _pass1_entities([("p1", "Ep1"), ("p2", "Ep2"), ("p3", "Ep3")]),
        ],
        ignore_index=True,
    )
    wallet_embeddings = pd.DataFrame(
        [
            {"node_id": "wa", "emb_0": 0.0, "emb_1": 0.0},
            {"node_id": "wb", "emb_0": 0.1, "emb_1": 0.1},
            {"node_id": "px", "emb_0": 0.05, "emb_1": 0.05},
            {"node_id": "wc", "emb_0": 50.0, "emb_1": 50.0},  # far away -> noise
            # padding cluster — HDBSCAN's 'leaf' method needs more than one
            # candidate cluster's worth of points to call anything non-noise.
            {"node_id": "p1", "emb_0": 10.0, "emb_1": 10.0},
            {"node_id": "p2", "emb_0": 10.1, "emb_1": 10.1},
            {"node_id": "p3", "emb_0": 10.05, "emb_1": 10.05},
        ]
    )
    hdbscan_df = run_hdbscan(wallet_embeddings, min_cluster_size=3)

    unprotected = reconcile_with_pass1(pass1, hdbscan_df)
    by_unprot = unprotected.entities.set_index("wallet_id")
    assert by_unprot.loc["wa", "entity_id"] != by_unprot.loc["wc", "entity_id"], "sanity: E splits without protection"

    protected = reconcile_with_pass1(pass1, hdbscan_df, protect_multiwallet_pass1_entities=True)
    by_prot = protected.entities.set_index("wallet_id")
    assert by_prot.loc["wa", "entity_id"] == by_prot.loc["wb", "entity_id"] == by_prot.loc["wc", "entity_id"]
    assert (protected.log["action"] == "split").sum() == 0


def test_min_merge_probability_blocks_low_confidence_merges():
    """Real-data guard: a merge should only fire when hdbscan is confidently
    not calling it noise for every wallet involved — otherwise skip it,
    treated as insufficient evidence rather than forced through. 1.1 is an
    impossible-to-satisfy floor (hdbscan_prob is always <= 1.0), so this
    doesn't depend on predicting HDBSCAN's exact probability output.
    """
    pass1 = pd.concat(
        [
            _pass1_entities([("w1", "E1"), ("w2", "E2"), ("w3", "E3")]),
            _pass1_entities([("p1", "Ep1"), ("p2", "Ep2"), ("p3", "Ep3")]),
        ],
        ignore_index=True,
    )
    wallet_embeddings = pd.DataFrame(
        [
            {"node_id": "w1", "emb_0": 0.0, "emb_1": 0.0},
            {"node_id": "w2", "emb_0": 0.05, "emb_1": 0.05},
            {"node_id": "w3", "emb_0": 0.1, "emb_1": 0.1},
            # padding cluster — same reason as the test above.
            {"node_id": "p1", "emb_0": 10.0, "emb_1": 10.0},
            {"node_id": "p2", "emb_0": 10.1, "emb_1": 10.1},
            {"node_id": "p3", "emb_0": 10.05, "emb_1": 10.05},
        ]
    )
    hdbscan_df = run_hdbscan(wallet_embeddings, min_cluster_size=3)
    assert (hdbscan_df.loc[["w1", "w2", "w3"], "hdbscan_label"] != -1).all(), "sanity: w1-w3 form one real cluster"

    permissive = reconcile_with_pass1(pass1, hdbscan_df, min_merge_probability=0.0)
    by_perm = permissive.entities.set_index("wallet_id")
    assert by_perm.loc["w1", "entity_id"] == by_perm.loc["w2", "entity_id"] == by_perm.loc["w3", "entity_id"]

    strict = reconcile_with_pass1(pass1, hdbscan_df, min_merge_probability=1.1)
    by_strict = strict.entities.set_index("wallet_id")
    assert by_strict.loc["w1", "entity_id"] != by_strict.loc["w2", "entity_id"] != by_strict.loc["w3", "entity_id"]
    assert (strict.log["action"] == "merge").sum() == 0


def _tiny_broadcast_graph(fan_ins: dict[str, int]) -> ig.Graph:
    """One IP vertex per key in fan_ins, `count` distinct Transaction
    vertices each with one BROADCAST_VIA edge into it — the minimal shape
    drop_high_fanin_broadcasters operates on."""
    names: list[str] = []
    types: list[str] = []
    for ip, count in fan_ins.items():
        names.append(ip)
        types.append("IP")
        for i in range(count):
            names.append(f"tx_{ip}_{i}")
            types.append("Transaction")

    g = ig.Graph(directed=True)
    g.add_vertices(len(names))
    g.vs["name"] = names
    g.vs["type"] = types
    index_of = {n: i for i, n in enumerate(names)}

    edges = [(index_of[f"tx_{ip}_{i}"], index_of[ip]) for ip, count in fan_ins.items() for i in range(count)]
    g.add_edges(edges)
    g.es["type"] = ["BROADCAST_VIA"] * len(edges)
    return g


def test_drop_high_fanin_broadcasters_removes_only_the_outlier_ip():
    """Real-data marker: a shared light-wallet-server IP relaying many
    unrelated owners' transactions — reproduced directly (no real data
    needed) as one IP node with a population-relative outlier fan-in among
    otherwise-ordinary IPs."""
    fan_ins = {f"ip_{i}": 1 + (i % 2) for i in range(30)}
    fan_ins["ip_hub"] = 40
    g = _tiny_broadcast_graph(fan_ins)

    filtered = drop_high_fanin_broadcasters(g)

    hub_idx = filtered.vs.find(name="ip_hub").index
    assert filtered.degree(hub_idx, mode="in") == 0

    ordinary_idx = filtered.vs.find(name="ip_0").index
    assert filtered.degree(ordinary_idx, mode="in") == fan_ins["ip_0"]
    assert filtered.ecount() == g.ecount() - fan_ins["ip_hub"]


def test_drop_high_fanin_broadcasters_is_noop_below_min_population():
    """Same fallback principle as detectors/pattern_sim.py's own adaptive
    gate: too few IPs to trust population statistics -> no-op, not a guess."""
    fan_ins = {f"ip_{i}": 1 for i in range(5)}
    fan_ins["ip_hub"] = 40
    g = _tiny_broadcast_graph(fan_ins)

    filtered = drop_high_fanin_broadcasters(g)
    assert filtered.ecount() == g.ecount()


def test_run_with_big_stack_returns_value_and_requests_bigger_stack(monkeypatch):
    """Real-data guard (ARGUS dataset Track M benchmarking report,
    2026-09-30): sklearn.cluster.HDBSCAN.fit's recursive single-linkage-tree
    construction overflowed Windows' default (1 MB) thread stack at
    real-world wallet counts. This must actually run the given function on a
    thread with the larger stack requested, return its value unchanged, and
    restore the previous stack size afterward."""
    original_stack_size = threading.stack_size
    requested_sizes = []

    def spy(size=0):
        requested_sizes.append(size)
        return original_stack_size(size)

    monkeypatch.setattr(threading, "stack_size", spy)

    result = run_with_big_stack(lambda: 21 + 21)

    assert result == 42
    assert BIG_STACK_BYTES in requested_sizes
    assert requested_sizes[-1] == 0  # restored to the default afterward


def test_run_with_big_stack_propagates_exceptions():
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_with_big_stack(boom)


def test_oversized_cluster_rejected_as_noise():
    """max_cluster_size guards against exactly the failure mode measured on
    the real dataset: a large, weakly-separated population collapsing into
    one HDBSCAN mega-cluster. 20 near-identical points with max_cluster_size
    below that must all be treated as noise, not merged into one entity.
    """
    wallet_embeddings = pd.DataFrame(
        [{"node_id": f"w{i}", "emb_0": 0.01 * i, "emb_1": 0.01 * i} for i in range(20)]
    )
    hdbscan_df = run_hdbscan(wallet_embeddings, min_cluster_size=3, max_cluster_size=5)
    assert (hdbscan_df["hdbscan_label"] == -1).all()
