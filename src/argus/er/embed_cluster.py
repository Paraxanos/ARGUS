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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from argus.er.union_find import UnionFind, add_wallet_link_edges

T = TypeVar("T")

# Real-data finding (ARGUS dataset Track M benchmarking report, 2026-09-30):
# sklearn.cluster.HDBSCAN.fit's recursive single-linkage-tree construction
# overflowed the stack on Windows at real-world wallet counts (150k+) --
# "no Python traceback, exit 127... Windows fatal exception: stack overflow."
# Windows' default MAIN-thread stack is 1 MB; Linux's is already 8 MB (per
# the report's own diagnosis, and matches this repo's general Windows-vs-
# Linux gotchas elsewhere -- see docs/offline_install.md's torch DLL-order
# note). 128 MB comfortably covers real-world scale without being wasteful.
BIG_STACK_BYTES = 128 * 1024 * 1024


def run_with_big_stack(fn: Callable[[], T], stack_size_bytes: int = BIG_STACK_BYTES) -> T:
    """Runs fn() (no args -- wrap with a lambda/functools.partial) on a
    thread with a much larger stack than the calling thread may have, then
    restores the previous default. threading.stack_size() affects
    subsequently-created threads process-wide, so this is set/restored
    around the one call, not left changed globally. Portable (harmless,
    just unnecessary, on platforms whose default stack is already big
    enough) rather than Windows-only, since the fix costs nothing on Linux.
    """
    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # re-raised on the caller's thread below, never swallowed
            error["exc"] = exc

    previous_stack_size = threading.stack_size(stack_size_bytes)
    try:
        worker = threading.Thread(target=_target)
        worker.start()
        worker.join()
    finally:
        threading.stack_size(previous_stack_size)

    if "exc" in error:
        raise error["exc"]
    return result["value"]

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

# Real-data finding (ARGUS dataset Track M benchmarking report, 2026-09-30):
# on real data pass 1 is NOT the vacuous singleton-only result diagnosed
# above — it measured precision 0.960 there, meaning its multi-wallet
# co-spend clusters are usually correct — yet pass 2 still split them,
# dropping overall ER precision from 0.960 to 0.209. The split PATH itself
# (tests/test_er_embed_cluster.py::test_merge_and_split_and_noise_all_correct)
# is a real, deliberately-designed capability (ARCHITECTURE.md sec 4.2: pass 2
# can correct an over-merged pass-1 cluster) and stays exercised/default-on
# here — this dataset's own pass 1 is vacuous (0 co-spend links), so it can
# never demonstrate the real-data regression either way. The two guards below
# are opt-in (default off, preserving every existing test/behavior exactly)
# for a caller that has independently confirmed pass 1 is reliable on its own
# dataset before disabling pass 2's ability to override it.
PROTECT_MULTIWALLET_PASS1_ENTITIES = False
MIN_MERGE_PROBABILITY = 0.0

# Same population-relative-outlier principle as detectors/pattern_sim.py's
# adaptive similarity gate ("don't trust one global constant to mean the same
# thing regardless of the population it's compared against") — a fixed
# absolute fan-in cutoff would mean something different at every scale;
# z-score-relative-to-this-run's-own IP population does not.
BROADCASTER_FANIN_Z_THRESHOLD = 3.0
MIN_IPS_FOR_FANIN_GATE = 30  # matches pattern_sim.py's own <30-candidates fallback threshold


@dataclass
class ReconciliationResult:
    entities: pd.DataFrame  # wallet_id, entity_id, source, conf, merge_split_log_ref
    log: pd.DataFrame  # log_id, action, hdbscan_cluster, pass1_entities, wallets, evidence_json
    same_entity_links: list[tuple[str, str, float]] = field(default_factory=list)


def drop_high_fanin_broadcasters(
    g: ig.Graph, z_threshold: float = BROADCASTER_FANIN_Z_THRESHOLD
) -> ig.Graph:
    """Real-data guard (ARGUS dataset Track M benchmarking report,
    2026-09-30): a shared broadcast endpoint (e.g. a light-wallet server
    relaying many unrelated owners' transactions) gives GraphSAGE's message
    passing a false similarity signal — every transaction routed through it
    receives a message from the same IP node, making otherwise-unrelated
    wallets look alike and over-merging them (this dataset's own generator
    gives every transaction an essentially unique IP — see this module's
    MEASURED RESULT docstring above — so it never plants this shape; the
    guard is an inert no-op here and only matters at real-data scale).

    Returns a COPY of g with BROADCAST_VIA edges removed from any IP node
    whose in-degree is a population-relative outlier (z-score over
    z_threshold among this run's own IP nodes) — never an absolute count,
    which would mean something different at every dataset scale. Below
    MIN_IPS_FOR_FANIN_GATE distinct IPs, or when every IP has the same
    fan-in (std == 0), population statistics aren't trustworthy and this is
    a no-op, same fallback principle as detectors/pattern_sim.py's own
    adaptive gate. Only affects the returned graph — callers should pass
    this filtered copy to embedding training only, never persist it in
    place of the real graph (BROADCAST_VIA edges are real data).
    """
    edgelist = g.get_edgelist()
    broadcast_eids = [i for i, et in enumerate(g.es["type"]) if et == "BROADCAST_VIA"]
    if not broadcast_eids:
        return g

    fan_in: dict[int, int] = {}
    for eid in broadcast_eids:
        _src, dst = edgelist[eid]
        fan_in[dst] = fan_in.get(dst, 0) + 1

    if len(fan_in) < MIN_IPS_FOR_FANIN_GATE:
        return g

    counts = np.array(list(fan_in.values()), dtype=float)
    mean, std = counts.mean(), counts.std()
    if std <= 1e-8:
        return g  # every IP has the same fan-in — no genuine outliers to drop

    outlier_ips = {ip for ip, count in fan_in.items() if (count - mean) / std > z_threshold}
    if not outlier_ips:
        return g

    to_delete = [eid for eid in broadcast_eids if edgelist[eid][1] in outlier_ips]
    g = g.copy()
    g.delete_edges(to_delete)
    return g


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


def reconcile_with_pass1(
    pass1_entities: pd.DataFrame,
    hdbscan: pd.DataFrame,
    protect_multiwallet_pass1_entities: bool = PROTECT_MULTIWALLET_PASS1_ENTITIES,
    min_merge_probability: float = MIN_MERGE_PROBABILITY,
) -> ReconciliationResult:
    """protect_multiwallet_pass1_entities and min_merge_probability are both
    opt-in real-data guards (see the constants' own comments above) — off by
    default, so every existing caller/test sees byte-identical behavior.

    protect_multiwallet_pass1_entities=True: any pass-1 entity with more than
    one wallet can never be SPLIT (its wallets all keep one shared provisional
    id regardless of individual hdbscan label) — but its wallets stay fully
    eligible to be MERGED into a larger cross-entity group, since that
    decision is independent of provisional-id assignment (see Step 2 below).

    min_merge_probability > 0: a merge only fires when every wallet in the
    triggering hdbscan cluster has hdbscan_prob at or above this floor —
    otherwise that cluster's merge is skipped entirely (treated as
    insufficient evidence, not forced through).
    """
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
    protected_entities: set[str] = set()
    if protect_multiwallet_pass1_entities:
        wallet_counts: dict[str, int] = {}
        for entity_id in wallet_to_pass1.values():
            wallet_counts[entity_id] = wallet_counts.get(entity_id, 0) + 1
        protected_entities = {e for e, n in wallet_counts.items() if n > 1}

    provisional: dict[str, str] = {}
    for wallet_id, entity_id in wallet_to_pass1.items():
        if entity_id in protected_entities:
            provisional[wallet_id] = entity_id  # never split — see protect_multiwallet_pass1_entities docstring
        else:
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
            if min(prob_of(w) for w in wallets) < min_merge_probability:
                continue  # insufficient evidence for this merge — real-data guard, see docstring above
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
