"""Entity resolution pass 2: refines pass 1's Union-Find clusters using
HDBSCAN over GraphSAGE wallet embeddings (argus.models.sage). Per
ARCHITECTURE.md sec 4.2, pass 2 can SPLIT an over-merged pass-1 cluster or
MERGE two separate pass-1 clusters — never silently overriding pass 1;
every action is logged with the evidence that triggered it
(artifacts/er_pass2_log.parquet, referenced from entities.parquet's
merge_split_log_ref).

WHY THIS IS THE RIGHT NEXT STEP (see docs/WRITEUP.md's Phase 2 ER
diagnosis): pass 1 resolves every wallet on this dataset to its own
singleton cluster (0.0 recall) because the only multi-input transactions are
CoinJoin rounds (which pass 1 must skip to avoid over-merging) and the
change-address heuristic never links two DIFFERENT wallets here. Pass 2
embeds wallets jointly with their network-layer (IP/ASN broadcast) context,
so it can catch entities that never literally co-spend but consistently
share broadcast infrastructure — exactly the case pass 1 is structurally
blind to.

ALGORITHM (two steps, because a single Union-Find can only ever grow
clusters, never split them):

1. SPLIT (provisional re-keying): a wallet's provisional id is
   "{pass1_entity_id}::{hdbscan_label}" for wallets HDBSCAN placed in a real
   (non-noise) cluster, or unchanged (its pass-1 entity_id) for noise
   wallets — HDBSCAN expressing no opinion is not treated as evidence to
   move a wallet. Two wallets pass 1 merged into the same entity but
   HDBSCAN places in different clusters get different provisional ids: a
   split.
2. MERGE (Union-Find over provisional ids): for each non-noise HDBSCAN
   cluster, every wallet in it shares that hdbscan_label, so their
   provisional ids differ only in the pass1_entity_id prefix — union all
   provisional ids appearing under the same hdbscan_label. Two wallets from
   different pass-1 entities that HDBSCAN clustered together end up under
   one root: a merge.

Today, only MERGE actions can fire on the real dataset (pass 1 produces no
multi-wallet clusters to split); the split path is exercised by a hand-built
fixture in tests/test_er_embed_cluster.py and will start mattering the
moment pass 1 (or a future dataset) produces non-singleton clusters.

SCOPE LIMITATION (documented, not a bug): a split does not retroactively
remove the CO_SPEND edges pass 1 already wrote for the cluster it came from
— those remain in the graph as pass 1's own record. Only new SAME_ENTITY
edges are added here, for genuine cross-pass1-entity merges.

MEASURED RESULT AND DIAGNOSED ROOT CAUSE (verified, not hidden — same
standard as pass 1's 0.0-recall finding): on the full 200k-tx dataset, pass 2
recall stays near 0 (~0.0000-0.0002 across every min_cluster_size /
cluster_selection_method / max_cluster_size combination tried) even though
the algorithm above is correct — it reproduces 100% of a hand-built
merge/split fixture exactly (tests/test_er_embed_cluster.py). Root cause,
diagnosed directly rather than assumed: argus.graph.build gives every IP
node its literal (full /32) address, and argus.synth.transactions randomizes
the last octet per transaction while holding the first three fixed per
entity — so two transactions FROM THE SAME ENTITY almost never broadcast to
the SAME IP NODE (up to 254 distinct last-octet values per entity), even
though they share a /24 subnet. GraphSAGE's message passing sees an
essentially unique IP node per transaction; the only place same-entity
transactions actually meet in the graph is two hops out at the shared ASN
node — diluted by every other unrelated entity coincidentally assigned to
the same one of only 64 synthetic ASNs (~31 entities/ASN on average). The
/24-level affinity the architecture doc describes is real in the generator
(argus.synth.networks) but not representable through current graph topology
at the granularity this encoder can exploit.

NOT fixed here — this is a graph-schema change (e.g. an IP node keyed by
subnet prefix, or a dedicated Subnet node type, in argus.graph.build,
propagating through argus.features.build's IP-based features) spanning
modules outside this phase's scope. The MAX_CLUSTER_SIZE guard below is a
permanent safety net (independent of this diagnosis — HDBSCAN's default
`eom` selection can collapse a large, weakly-separated population into one
enormous cluster regardless of root cause; verified directly: uncapped, one
run merged 49,962 of 49,983 wallets into six clusters, precision 0.004),
not a workaround for the diagnosis above.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from argus.er.union_find import UnionFind, add_wallet_link_edges

MIN_CLUSTER_SIZE = 3
# 'leaf' (favors more, smaller, finer-grained clusters) over sklearn's own
# default 'eom' (favors fewer, larger, more "stable" clusters) — verified
# directly: 'eom' on the real dataset produces one cluster covering ~99% of
# all wallets regardless of min_cluster_size; 'leaf' degrades far less
# catastrophically (see module docstring's MEASURED RESULT section).
CLUSTER_SELECTION_METHOD = "leaf"
# A real Bitcoin/entity-resolution cluster spanning tens of thousands of
# wallets is implausible in a dataset this size (this repo's own synthetic
# ground truth tops out at 308 wallets/entity) — sklearn.cluster.HDBSCAN's
# native max_cluster_size treats an oversized cluster as noise rather than a
# real entity, a permanent safety net independent of the root-cause
# diagnosis in this module's docstring.
MAX_CLUSTER_SIZE = 500
NOISE_LABEL = -1


@dataclass
class ReconciliationResult:
    entities: pd.DataFrame  # wallet_id, entity_id, source, conf, merge_split_log_ref
    log: pd.DataFrame  # log_id, action, hdbscan_cluster, pass1_entities, wallets, evidence_json
    same_entity_links: list[tuple[str, str, float]] = field(default_factory=list)


def run_hdbscan(
    wallet_embeddings: pd.DataFrame,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    cluster_selection_method: str = CLUSTER_SELECTION_METHOD,
    max_cluster_size: int | None = MAX_CLUSTER_SIZE,
) -> pd.DataFrame:
    """wallet_id-indexed DataFrame of hdbscan_label (-1 = noise, i.e. no
    cluster — HDBSCAN's signature behavior vs. e.g. k-means, and exactly
    what lets pass 2 leave ambiguous wallets alone rather than forcing every
    wallet into a same-size-shaped cluster) and hdbscan_prob (cluster
    membership strength, used directly as this pass's per-wallet confidence).
    """
    emb_cols = [c for c in wallet_embeddings.columns if c.startswith("emb_")]
    if wallet_embeddings.empty or not emb_cols or len(wallet_embeddings) < min_cluster_size:
        return pd.DataFrame(columns=["hdbscan_label", "hdbscan_prob"])

    X = wallet_embeddings[emb_cols].to_numpy()
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        max_cluster_size=max_cluster_size,
    )
    labels = clusterer.fit_predict(X)
    probabilities = getattr(clusterer, "probabilities_", np.ones(len(labels)))
    return pd.DataFrame(
        {
            "wallet_id": wallet_embeddings["node_id"].to_numpy(),
            "hdbscan_label": labels,
            "hdbscan_prob": probabilities,
        }
    ).set_index("wallet_id")


def reconcile_with_pass1(pass1_entities: pd.DataFrame, hdbscan: pd.DataFrame) -> ReconciliationResult:
    wallet_to_pass1 = dict(zip(pass1_entities["wallet_id"], pass1_entities["entity_id"]))
    wallet_to_conf = dict(zip(pass1_entities["wallet_id"], pass1_entities["conf"]))

    def label_of(wallet_id: str) -> int:
        if wallet_id in hdbscan.index:
            return int(hdbscan.loc[wallet_id, "hdbscan_label"])
        return NOISE_LABEL

    def prob_of(wallet_id: str) -> float:
        if wallet_id in hdbscan.index:
            return float(hdbscan.loc[wallet_id, "hdbscan_prob"])
        return 0.0

    # Step 1 — provisional ids (implements split).
    provisional: dict[str, str] = {}
    for wallet_id, entity_id in wallet_to_pass1.items():
        label = label_of(wallet_id)
        provisional[wallet_id] = entity_id if label == NOISE_LABEL else f"{entity_id}::{label}"

    # Step 2 — union provisional ids sharing a non-noise hdbscan label (implements merge).
    uf = UnionFind(set(provisional.values()))
    by_label: dict[int, list[str]] = {}
    for wallet_id, entity_id in wallet_to_pass1.items():
        label = label_of(wallet_id)
        if label == NOISE_LABEL:
            continue
        by_label.setdefault(label, []).append(wallet_id)

    log_rows: list[dict] = []
    merge_log_id_by_label: dict[int, str] = {}
    for label, wallets in by_label.items():
        prov_ids = {provisional[w] for w in wallets}
        if len(prov_ids) > 1:
            anchor = min(prov_ids)
            for other in sorted(prov_ids - {anchor}):
                uf.union(anchor, other)
            log_id = f"merge_{label}"
            merge_log_id_by_label[label] = log_id
            log_rows.append(
                {
                    "log_id": log_id,
                    "action": "merge",
                    "hdbscan_cluster": int(label),
                    "pass1_entities": sorted({wallet_to_pass1[w] for w in wallets}),
                    "wallets": sorted(wallets),
                    "evidence_json": json.dumps(
                        {
                            "mean_hdbscan_probability": float(np.mean([prob_of(w) for w in wallets])),
                            "cluster_size": len(wallets),
                        }
                    ),
                }
            )

    # Split log entries — one per pass-1 entity whose wallets landed in more
    # than one provisional bucket (informational; the split is already
    # encoded in `provisional`/the final entity ids, nothing further to
    # apply here).
    pass1_groups: dict[str, list[str]] = {}
    for wallet_id, entity_id in wallet_to_pass1.items():
        pass1_groups.setdefault(entity_id, []).append(wallet_id)
    split_log_id_by_wallet: dict[str, str] = {}
    for entity_id, wallets in pass1_groups.items():
        buckets = {provisional[w] for w in wallets}
        if len(buckets) > 1:
            log_id = f"split_{entity_id}"
            for w in wallets:
                split_log_id_by_wallet[w] = log_id
            log_rows.append(
                {
                    "log_id": log_id,
                    "action": "split",
                    "hdbscan_cluster": None,
                    "pass1_entities": [entity_id],
                    "wallets": sorted(wallets),
                    "evidence_json": json.dumps({"bucket_count": len(buckets)}),
                }
            )

    # Final assignment + rows. `by_label`'s merge condition above only fires
    # when a label's wallets span >1 distinct pass1 entity (two wallets from
    # the SAME pass1 entity always share one provisional id, so they can
    # never by themselves make `len(prov_ids) > 1`) — so `label in
    # merge_log_id_by_label` is exactly "this wallet's hdbscan label caused a
    # genuine cross-entity merge", no extra bookkeeping needed.
    rows = []
    same_entity_groups: dict[str, list[str]] = {}
    for wallet_id, entity_id in wallet_to_pass1.items():
        label = label_of(wallet_id)
        final_root = uf.find(provisional[wallet_id])

        if label in merge_log_id_by_label:
            source, conf, log_ref = "pass2", prob_of(wallet_id), merge_log_id_by_label[label]
            same_entity_groups.setdefault(final_root, []).append(wallet_id)
        elif wallet_id in split_log_id_by_wallet:
            source, conf, log_ref = "pass2", wallet_to_conf[wallet_id], split_log_id_by_wallet[wallet_id]
        else:
            source, conf, log_ref = "pass1", wallet_to_conf[wallet_id], None

        rows.append(
            {
                "wallet_id": wallet_id,
                "entity_id": final_root,
                "source": source,
                "conf": conf,
                "merge_split_log_ref": log_ref,
            }
        )

    same_entity_links: list[tuple[str, str, float]] = []
    for root, wallets in same_entity_groups.items():
        if len(wallets) < 2:
            continue
        anchor = wallets[0]
        for other in wallets[1:]:
            # 0.0 is a legitimate low-confidence HDBSCAN membership score,
            # not a missing value — no truthy-fallback substitution here.
            confidence = min(prob_of(anchor), prob_of(other))
            same_entity_links.append((anchor, other, confidence))

    entities_df = pd.DataFrame(rows, columns=["wallet_id", "entity_id", "source", "conf", "merge_split_log_ref"])
    log_df = pd.DataFrame(
        log_rows, columns=["log_id", "action", "hdbscan_cluster", "pass1_entities", "wallets", "evidence_json"]
    )
    return ReconciliationResult(entities=entities_df, log=log_df, same_entity_links=same_entity_links)


def add_same_entity_edges(g: ig.Graph, links: list[tuple[str, str, float]]) -> ig.Graph:
    return add_wallet_link_edges(g, links, "SAME_ENTITY")


def write_pass2_outputs(result: ReconciliationResult, entities_path: Path, log_path: Path) -> None:
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    result.entities.to_parquet(entities_path, index=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # pass1_entities/wallets are list columns — stringify for a stable, always
    # readable parquet round-trip (avoids pyarrow inferring a variable-length
    # list type that later differs across an all-empty vs non-empty log).
    log_out = result.log.copy()
    for col in ("pass1_entities", "wallets"):
        if col in log_out.columns:
            log_out[col] = log_out[col].apply(json.dumps)
    log_out.to_parquet(log_path, index=False)
