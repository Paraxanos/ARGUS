"""Node feature engineering over the typed graph + canonical transactions.

Produces one row per graph node (Wallet, Transaction, IP, ASN) with a uniform
set of f_* numeric columns — the SAME columns for every node type, so the
schema is stable regardless of which edge types happen to have volume in a
given run. Features that don't apply to a node's type (e.g. temporal stats on
an IP node) are 0-filled rather than NaN or a dropped row — see FILL STRATEGY
below.

FILL STRATEGY (explicit, per requirement): every f_* column defaults to 0.0
when a node has no applicable data — zero activity, zero degree, zero entropy,
zero burstiness deviation are all semantically correct "nothing observed"
values for every feature here. No row is ever dropped: the node list itself
comes from the graph's vertex set, and every feature table is left-merged onto
it, so a missing computation simply leaves that node's column at the fill
value, never removes the row.

CROSS-LAYER ENTITY FEATURES: f_entity_asn_entropy is computed by grouping
wallets via artifacts/entities.parquet (ER pass 1's resolved clusters) — never
via ground_truth/entities.parquet, which would leak the label the downstream
detectors are meant to predict. Pass 1 currently resolves every wallet to its
own singleton cluster (see the Phase 2 ER diagnosis — recall 0 on this
dataset), so today this feature is numerically equivalent to a per-wallet ASN
entropy. It is still the architecturally correct source: it improves for free
once pass 1 (or a future pass 2) produces real clusters, with no change here.
"""
from __future__ import annotations

import math
from collections import Counter

import igraph as ig
import pandas as pd

from argus.er.union_find import is_coinjoin_like

# The graph's full edge-type vocabulary (docs/contracts.md), hardcoded rather
# than derived from this run's actual edges — CO_SPEND/SAME_ENTITY may have
# zero edges in a given run (a standalone `make features` re-run against a
# graph.pkl from before ER pass 2 ran, for instance) but the column must
# still exist, and every edge type actually present in the graph MUST be
# listed here or _bulk_topology_features's per-type counter raises KeyError.
EDGE_TYPES = ["FUNDS", "PAYS", "BROADCAST_VIA", "RESOLVES_TO", "CO_SPEND", "SAME_ENTITY"]


def _bulk_topology_features(g: ig.Graph) -> pd.DataFrame:
    """Degree (total/in/out, and split per edge type) and fan-in/fan-out (count
    of DISTINCT neighbors, which differs from degree when a node has parallel
    edges — e.g. a Transaction with two FUNDS edges from the same input wallet).
    One pass over the edge list, O(E).
    """
    n = g.vcount()
    edgelist = g.get_edgelist()
    edge_types = g.es["type"]

    per_type_degree = {et: [0] * n for et in EDGE_TYPES}
    in_neighbors: list[set] = [set() for _ in range(n)]
    out_neighbors: list[set] = [set() for _ in range(n)]

    for (s, t), et in zip(edgelist, edge_types):
        per_type_degree[et][s] += 1
        per_type_degree[et][t] += 1
        out_neighbors[s].add(t)
        in_neighbors[t].add(s)

    data = {
        "node_id": g.vs["name"],
        "f_degree_total": g.degree(mode="all"),
        "f_degree_in": g.degree(mode="in"),
        "f_degree_out": g.degree(mode="out"),
        "f_fan_in": [len(s) for s in in_neighbors],
        "f_fan_out": [len(s) for s in out_neighbors],
    }
    for et in EDGE_TYPES:
        data[f"f_degree_{et.lower()}"] = per_type_degree[et]

    return pd.DataFrame(data)


def _motif_features(df: pd.DataFrame, node_ids: list[str]) -> pd.DataFrame:
    """Two documented motifs, each meaningful for both Wallet and Transaction
    nodes (0 elsewhere):

    f_motif_coinjoin_like — for a Transaction: 1 if this tx is structurally
    CoinJoin-like (see argus.er.union_find.is_coinjoin_like). For a Wallet:
    the number of such rounds it participated in as an input. This is the
    same structural check ER pass 1 uses to guard against over-merging, reused
    here as a standalone signal for downstream detectors.

    f_motif_self_change — for a Transaction: 1 if any output address equals
    one of its own input addresses (the change-address pattern). For a
    Wallet: how many of its transactions show this pattern.
    """
    coinjoin_score: dict[str, float] = dict.fromkeys(node_ids, 0.0)
    self_change_score: dict[str, float] = dict.fromkeys(node_ids, 0.0)

    for row in df.itertuples(index=False):
        input_addresses = list(row.input_addresses)
        output_addresses = list(row.output_addresses)
        output_amounts = list(row.output_amounts)
        distinct_inputs = set(input_addresses)

        if is_coinjoin_like(input_addresses, output_amounts):
            coinjoin_score[row.txid] = 1.0
            for w in distinct_inputs:
                coinjoin_score[w] = coinjoin_score.get(w, 0.0) + 1.0

        if any(addr in distinct_inputs for addr in output_addresses):
            self_change_score[row.txid] = 1.0
            for w in distinct_inputs:
                self_change_score[w] = self_change_score.get(w, 0.0) + 1.0

    return pd.DataFrame(
        {
            "node_id": node_ids,
            "f_motif_coinjoin_like": [coinjoin_score[n] for n in node_ids],
            "f_motif_self_change": [self_change_score[n] for n in node_ids],
        }
    )


def _temporal_stats(timestamps: pd.Series) -> pd.Series:
    n = len(timestamps)
    if n < 2:
        return pd.Series(
            {"f_temporal_tx_count": float(n), "f_temporal_mean_interarrival": 0.0,
             "f_temporal_std_interarrival": 0.0, "f_temporal_burstiness": 0.0}
        )
    diffs = timestamps.sort_values().diff().dropna().dt.total_seconds()
    mean = float(diffs.mean())
    std = float(diffs.std(ddof=0)) if len(diffs) > 1 else 0.0
    if math.isnan(std):
        std = 0.0
    # Goh & Barabási burstiness: -1 perfectly periodic, 0 Poisson-like, 1 bursty.
    burstiness = (std - mean) / (std + mean) if (std + mean) > 0 else 0.0
    return pd.Series(
        {"f_temporal_tx_count": float(n), "f_temporal_mean_interarrival": mean,
         "f_temporal_std_interarrival": std, "f_temporal_burstiness": burstiness}
    )


def _geo_hop_stats(countries: pd.Series) -> pd.Series:
    n = len(countries)
    if n < 2:
        return pd.Series({"f_geo_hop_count": 0.0, "f_geo_hop_rate": 0.0})
    values = countries.tolist()
    hops = sum(1 for i in range(1, len(values)) if values[i] != values[i - 1])
    return pd.Series({"f_geo_hop_count": float(hops), "f_geo_hop_rate": hops / (n - 1)})


def _wallet_cross_layer_and_temporal_features(
    df: pd.DataFrame, resolved_entities: pd.DataFrame
) -> pd.DataFrame:
    """Every field here (temporal, IP reuse, geo-hop, ASN entropy) is keyed to a
    wallet's activity as a transaction SENDER — src_ip/timestamp/geo_country/asn
    are generated per-transaction and tied to whoever broadcast it, i.e. the
    tx's own input wallet(s). A transaction with >1 distinct input wallet (only
    CoinJoin rounds, in this dataset) is attributed to all of them.
    """
    records = []
    for row in df.itertuples(index=False):
        for wallet_id in set(row.input_addresses):
            records.append((wallet_id, row.timestamp, row.src_ip, row.geo_country, row.asn))
    activity = pd.DataFrame(records, columns=["wallet_id", "timestamp", "src_ip", "geo_country", "asn"])
    activity = activity.sort_values(["wallet_id", "timestamp"])

    grouped = activity.groupby("wallet_id")
    temporal = grouped["timestamp"].apply(_temporal_stats).unstack()
    geo = grouped["geo_country"].apply(_geo_hop_stats).unstack()
    ip_stats = grouped["src_ip"].agg(
        f_ip_distinct_count=lambda s: float(s.nunique()),
        f_ip_reuse_count=lambda s: float(max(0, len(s) - s.nunique())),
    )

    wallet_to_entity = dict(zip(resolved_entities["wallet_id"], resolved_entities["entity_id"]))
    entity_asn_totals: dict[str, Counter] = {}
    for (wallet_id, asn), count in activity.groupby(["wallet_id", "asn"]).size().items():
        entity_id = wallet_to_entity.get(wallet_id)
        if entity_id is None:
            continue
        entity_asn_totals.setdefault(entity_id, Counter())[asn] += count

    entity_entropy: dict[str, float] = {}
    for entity_id, counts in entity_asn_totals.items():
        total = sum(counts.values())
        raw = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0 and total > 0)
        entity_entropy[entity_id] = max(0.0, raw)  # guards float noise (e.g. -0.0) on single-ASN entities

    asn_entropy = {
        wallet_id: entity_entropy.get(wallet_to_entity.get(wallet_id), 0.0)
        for wallet_id in activity["wallet_id"].unique()
    }

    result = temporal.join(geo, how="outer").join(ip_stats, how="outer")
    result["f_entity_asn_entropy"] = result.index.map(asn_entropy).fillna(0.0)
    result = result.reset_index().rename(columns={"wallet_id": "node_id"})
    return result


def compute_node_features(g: ig.Graph, df: pd.DataFrame, resolved_entities: pd.DataFrame) -> pd.DataFrame:
    node_ids = g.vs["name"]
    base = pd.DataFrame({"node_id": node_ids, "node_type": g.vs["type"]})

    topo = _bulk_topology_features(g)
    motifs = _motif_features(df, node_ids)
    wallet_features = _wallet_cross_layer_and_temporal_features(df, resolved_entities)

    result = base.merge(topo, on="node_id", how="left")
    result = result.merge(motifs, on="node_id", how="left")
    result = result.merge(wallet_features, on="node_id", how="left")

    f_cols = [c for c in result.columns if c.startswith("f_")]
    result[f_cols] = result[f_cols].fillna(0.0)
    return result
