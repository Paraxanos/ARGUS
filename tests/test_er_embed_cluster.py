"""ER pass 2 (argus.er.embed_cluster): HDBSCAN + reconciliation logic,
tested against a hand-built embedding fixture where merge, split, and noise
all have a known-correct answer — see argus.models.sage's own test for the
GraphSAGE training/embedding half, and this module's MEASURED RESULT
docstring for why the real 200k-tx dataset does not (currently) exercise a
meaningful merge/split outcome (a diagnosed graph-topology limitation, not a
correctness bug in the algorithm this file tests).
"""
from __future__ import annotations

import pandas as pd

from argus.er.embed_cluster import reconcile_with_pass1, run_hdbscan


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
